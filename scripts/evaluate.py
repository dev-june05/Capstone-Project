#!/usr/bin/env python3
"""
evaluate.py
-----------
Evaluation script for 4-class semantic segmentation models.

Evaluates performance on validation or test sets across key metrics:
  - Mean Intersection-over-Union (mIoU)
  - Per-class IoU (Walkable, Non-walkable, Dynamic Obstacle, Structural Obstacle)
  - Mean Dice and Per-class Dice (F1-score)
  - Obstacle Recall (Dynamic Obstacle + Structural Obstacle)
  - Missed Obstacle Rate (False Negative Rate for obstacle classes)
  - Boundary F1-Score (Contour boundary alignment quality)

Generates colored mask overlays blended with original images for inspection.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Class definitions
CLASSES = [
    "Walkable",             # Class 0
    "Non-walkable",         # Class 1
    "Dynamic Obstacle",     # Class 2
    "Structural Obstacle",  # Class 3
]
NUM_CLASSES = len(CLASSES)
OBSTACLE_CLASSES = [2, 3]  # Dynamic (2) & Structural (3)

# Color palette for mask overlay: BGR format for OpenCV
# Class 0 (Walkable): Green [0, 200, 0]
# Class 1 (Non-walkable): Gray [128, 128, 128]
# Class 2 (Dynamic Obstacle): Red [0, 0, 220]
# Class 3 (Structural Obstacle): Orange/Yellow [0, 165, 255]
COLOR_PALETTE = np.array([
    [0, 200, 0],     # Class 0: Walkable (Green)
    [128, 128, 128], # Class 1: Non-walkable (Gray)
    [0, 0, 220],     # Class 2: Dynamic Obstacle (Red)
    [0, 165, 255],   # Class 3: Structural Obstacle (Orange)
], dtype=np.uint8)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("EvaluateSegmentation")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate 4-class semantic segmentation models and compute safety metrics"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML training configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model weights (.pth checkpoint)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate ('val' or 'test')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory to save metric reports and overlay visualizations",
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        default=True,
        help="Whether to save colored prediction overlay images (default: True)",
    )
    parser.add_argument(
        "--max-vis-samples",
        type=int,
        default=50,
        help="Maximum number of overlay images to save",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class SegmentationMetrics:
    """
    Accumulator and computer for semantic segmentation metrics:
    mIoU, Per-Class IoU, Dice coefficients, Obstacle Recall,
    Missed Obstacle Rate, and Boundary F1.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.boundary_f1_scores: List[float] = []

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """
        Update confusion matrix with a batch or single image pair.

        Args:
            pred: Predicted class label array of shape [H, W] or [B, H, W].
            target: Ground truth label array of same shape.
        """
        pred = pred.flatten()
        target = target.flatten()

        valid_mask = (target != self.ignore_index) & (target < self.num_classes) & (target >= 0)
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]

        indices = self.num_classes * target_valid.astype(np.int64) + pred_valid.astype(np.int64)
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def add_boundary_f1(self, score: float) -> None:
        """Record boundary F1 score for an image."""
        if not np.isnan(score):
            self.boundary_f1_scores.append(score)

    def compute(self) -> Dict[str, Any]:
        """
        Compute final aggregated metrics from the accumulated confusion matrix.

        Returns:
            Dictionary of metrics including mIoU, class IoUs, Dice, Obstacle Recall,
            Missed Obstacle Rate, and Boundary F1.
        """
        cm = self.confusion_matrix
        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(axis=0).astype(np.float64) - tp
        fn = cm.sum(axis=1).astype(np.float64) - tp

        # Per-class IoU: TP / (TP + FP + FN)
        union = tp + fp + fn
        iou_per_class = np.zeros(self.num_classes, dtype=np.float64)
        valid_class_mask = union > 0
        iou_per_class[valid_class_mask] = tp[valid_class_mask] / union[valid_class_mask]
        miou = np.nanmean(iou_per_class[valid_class_mask]) if np.any(valid_class_mask) else 0.0

        # Per-class Dice (F1): 2*TP / (2*TP + FP + FN)
        dice_denom = 2 * tp + fp + fn
        dice_per_class = np.zeros(self.num_classes, dtype=np.float64)
        valid_dice_mask = dice_denom > 0
        dice_per_class[valid_dice_mask] = (2 * tp[valid_dice_mask]) / dice_denom[valid_dice_mask]
        mean_dice = np.nanmean(dice_per_class[valid_dice_mask]) if np.any(valid_dice_mask) else 0.0

        # Obstacle Recall & Missed Obstacle Rate (classes 2 & 3)
        obs_tp = tp[OBSTACLE_CLASSES].sum()
        obs_fn = fn[OBSTACLE_CLASSES].sum()
        obs_total = obs_tp + obs_fn

        if obs_total > 0:
            obstacle_recall = obs_tp / obs_total
            missed_obstacle_rate = obs_fn / obs_total
        else:
            obstacle_recall = 0.0
            missed_obstacle_rate = 0.0

        mean_boundary_f1 = (
            float(np.mean(self.boundary_f1_scores)) if len(self.boundary_f1_scores) > 0 else 0.0
        )

        return {
            "mIoU": float(miou),
            "iou_per_class": iou_per_class.tolist(),
            "mean_dice": float(mean_dice),
            "dice_per_class": dice_per_class.tolist(),
            "obstacle_recall": float(obstacle_recall),
            "missed_obstacle_rate": float(missed_obstacle_rate),
            "boundary_f1": float(mean_boundary_f1),
            "confusion_matrix": cm.tolist(),
        }


def compute_boundary_f1(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    bound_theta: float = 0.008,
) -> float:
    """
    Compute Boundary F1 score between predicted segmentation and ground truth.

    Extracts binary boundary contours and measures contour precision and recall
    within a pixel tolerance distance proportional to the image diagonal.

    TODO:
      - Use morphological gradient or contour distance transform to evaluate
        boundary precision and recall.
      - Default tolerance bound_theta = 0.008 (approx 8-10 pixels for 1024x512).

    Args:
        pred_mask: 2D array of predicted class labels [H, W].
        gt_mask: 2D array of ground truth labels [H, W].
        bound_theta: Boundary tolerance factor relative to image diagonal.

    Returns:
        Boundary F1 score between 0.0 and 1.0.
    """
    # TODO: Implement full contour boundary F1 distance transform calculation
    # Placeholder returning standard boundary approximation stub
    return 0.0


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert a 2D segmentation label mask [H, W] to a 3D BGR color mask [H, W, 3].

    Args:
        mask: 2D array of class indices [0..NUM_CLASSES-1].

    Returns:
        3-channel BGR uint8 image.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx in range(NUM_CLASSES):
        color_mask[mask == cls_idx] = COLOR_PALETTE[cls_idx]
    return color_mask


def create_overlay(image_bgr: np.ndarray, pred_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create a blended overlay of the segmentation predictions on top of the original image.

    Args:
        image_bgr: Original image in BGR format [H, W, 3].
        pred_mask: 2D predicted label mask [H, W].
        alpha: Blending weight for color mask (1 - alpha for original image).

    Returns:
        Blended BGR uint8 image.
    """
    color_mask = colorize_mask(pred_mask)
    overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, color_mask, alpha, 0)
    return overlay


def load_model(config: Dict[str, Any], checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Load segmentation model architecture and restore weights from checkpoint.

    Args:
        config: Full configuration dictionary.
        checkpoint_path: Path to checkpoint .pth file.
        device: PyTorch device.

    Returns:
        Model module loaded and placed in evaluation mode.
    """
    model_name = config["model"]["name"].lower()
    num_classes = config["model"]["num_classes"]

    # Connect with models.get_model
    try:
        from models import get_model
        try:
            model = get_model(name=model_name, num_classes=num_classes)
        except TypeError:
            model = get_model(name=model_name, num_classes=num_classes, auxiliary_loss=False)
    except ImportError:
        try:
            from src.models import get_model
            try:
                model = get_model(name=model_name, num_classes=num_classes)
            except TypeError:
                model = get_model(name=model_name, num_classes=num_classes, auxiliary_loss=False)
        except ImportError:
            raise NotImplementedError(
                f"Architecture '{model_name}' builder not available in models."
            )

    logger.info("Loading checkpoint from %s...", checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully.")
    return model


def build_eval_dataloader(config: Dict[str, Any], split: str) -> DataLoader:
    """
    Construct DataLoader for the evaluation split (val or test).

    TODO:
      - Instantiate Cityscapes / IDD dataset for `split`.
      - Apply validation transforms (Resize to 512x1024, Normalize).

    Args:
        config: Full configuration dictionary.
        split: 'val' or 'test'.

    Returns:
        PyTorch DataLoader.
    """
    # TODO: Implement dataset loading for validation / testing
    raise NotImplementedError(
        f"Evaluation DataLoader for split '{split}' is not yet implemented. "
        "Connect with src.data dataset implementations."
    )


def print_results_table(metrics: Dict[str, Any]) -> None:
    """
    Print formatted evaluation results table to stdout.

    Args:
        metrics: Dictionary of computed metric values.
    """
    border = "=" * 65
    divider = "-" * 65

    print("\n" + border)
    print("        4-CLASS SEMANTIC SEGMENTATION EVALUATION RESULTS")
    print(border)
    print(f"{'Class Name':<25} | {'IoU (%)':<15} | {'Dice / F1 (%)':<15}")
    print(divider)

    ious = metrics.get("iou_per_class", [0.0] * NUM_CLASSES)
    dices = metrics.get("dice_per_class", [0.0] * NUM_CLASSES)

    for i, cls_name in enumerate(CLASSES):
        iou_val = ious[i] * 100.0 if i < len(ious) else 0.0
        dice_val = dices[i] * 100.0 if i < len(dices) else 0.0
        print(f"{cls_name:<25} | {iou_val:>12.2f} % | {dice_val:>12.2f} %")

    print(divider)
    print(f"{'Mean IoU (mIoU)':<25} | {metrics.get('mIoU', 0.0) * 100.0:>12.2f} % |")
    print(f"{'Mean Dice':<25} | {metrics.get('mean_dice', 0.0) * 100.0:>12.2f} % |")
    print(divider)
    print("  CRITICAL SAFETY METRICS (Obstacle Detection & Boundary Accuracy)")
    print(divider)
    print(f"{'Obstacle Recall':<30} : {metrics.get('obstacle_recall', 0.0) * 100.0:>8.2f} %")
    print(f"{'Missed Obstacle Rate':<30} : {metrics.get('missed_obstacle_rate', 0.0) * 100.0:>8.2f} %")
    print(f"{'Boundary F1 Score':<30} : {metrics.get('boundary_f1', 0.0) * 100.0:>8.2f} %")
    print(border + "\n")


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    save_visualizations: bool = True,
    max_vis_samples: int = 50,
) -> Dict[str, Any]:
    """
    Run evaluation loop over the dataloader.

    Args:
        model: Evaluated model module.
        dataloader: PyTorch evaluation DataLoader.
        device: Execution device.
        output_dir: Directory where predictions and overlay plots are saved.
        save_visualizations: Whether to save visualization overlays.
        max_vis_samples: Cap on number of images saved.

    Returns:
        Computed metrics dictionary.
    """
    metrics_tracker = SegmentationMetrics(num_classes=NUM_CLASSES)
    vis_dir = output_dir / "overlays"
    if save_visualizations:
        vis_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Handle dictionary or tuple batch
            if isinstance(batch, dict):
                images = batch["image"]
                masks = batch["mask"]
            else:
                images, masks = batch[:2]

            images = images.to(device)

            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs.get("main", next(iter(outputs.values())))
            elif isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets = masks.numpy()

            for i in range(preds.shape[0]):
                pred_i = preds[i]
                target_i = targets[i]

                # Update metrics
                metrics_tracker.update(pred_i, target_i)

                # Compute Boundary F1
                bf1 = compute_boundary_f1(pred_i, target_i)
                metrics_tracker.add_boundary_f1(bf1)

                # Save overlay visualization
                if save_visualizations and saved_count < max_vis_samples:
                    # TODO: Use un-normalized original image for overlay
                    dummy_img = np.zeros((*pred_i.shape, 3), dtype=np.uint8)
                    overlay = create_overlay(dummy_img, pred_i)
                    out_path = vis_dir / f"pred_sample_{saved_count:04d}.png"
                    cv2.imwrite(str(out_path), overlay)
                    saved_count += 1

    return metrics_tracker.compute()


def main() -> None:
    """Main execution entry point for evaluation."""
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing model evaluation on device: %s", device)

    # Load Model
    try:
        model = load_model(config, args.checkpoint, device)
    except (NotImplementedError, FileNotFoundError) as e:
        logger.warning("Model load stub notice: %s", e)
        # Display sample format table for demonstration
        print_results_table({
            "mIoU": 0.0,
            "iou_per_class": [0.0, 0.0, 0.0, 0.0],
            "mean_dice": 0.0,
            "dice_per_class": [0.0, 0.0, 0.0, 0.0],
            "obstacle_recall": 0.0,
            "missed_obstacle_rate": 0.0,
            "boundary_f1": 0.0,
        })
        return

    # Load DataLoader
    try:
        dataloader = build_eval_dataloader(config, split=args.split)
    except NotImplementedError as e:
        logger.warning("Dataset evaluation notice: %s", e)
        return

    # Run evaluation
    results = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=output_dir,
        save_visualizations=args.save_visualizations,
        max_vis_samples=args.max_vis_samples,
    )

    # Print results table
    print_results_table(results)

    # Save metrics JSON / YAML
    metrics_file = output_dir / f"evaluation_{args.split}_metrics.yaml"
    with open(metrics_file, "w", encoding="utf-8") as f:
        yaml.dump(results, f, sort_keys=False)
    logger.info("Saved evaluation summary to %s", metrics_file)


if __name__ == "__main__":
    main()
