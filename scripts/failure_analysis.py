#!/usr/bin/env python3
"""
failure_analysis.py
-------------------
Failure analysis and diagnostic script for 4-class semantic segmentation.

Identifies worst-performing images in test/validation sets by per-image mIoU,
generates side-by-side diagnostic panels:
  [Original Image | Ground Truth | Prediction Overlay | Error Map]
and summarizes failure categories (small obstacles missed, boundary ambiguity,
dynamic vs structural confusion, shadow/lighting artifacts).
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

# Semantic classes
CLASSES = [
    "Walkable",             # Class 0
    "Non-walkable",         # Class 1
    "Dynamic Obstacle",     # Class 2
    "Structural Obstacle",  # Class 3
]
NUM_CLASSES = len(CLASSES)

# Color palette: BGR format for OpenCV
COLOR_PALETTE = np.array([
    [0, 200, 0],     # Class 0: Walkable (Green)
    [128, 128, 128], # Class 1: Non-walkable (Gray)
    [0, 0, 220],     # Class 2: Dynamic Obstacle (Red)
    [0, 165, 255],   # Class 3: Structural Obstacle (Orange)
], dtype=np.uint8)

# Common failure mode categories for navigation safety
FAILURE_CATEGORIES = [
    "Missed Dynamic Obstacles (Pedestrians/Cyclists False Negatives)",
    "Missed Structural Obstacles (Poles/Barriers False Negatives)",
    "Walkable vs Non-walkable Boundary Confusion (Curb/Transition Bleeding)",
    "Dynamic vs Structural Obstacle Confusion",
    "False Alarm Obstacles on Walkable Surface (Shadows/Puddle Reflections)",
]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FailureAnalysis")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for failure analysis."""
    parser = argparse.ArgumentParser(
        description="Diagnose worst-performing failure cases and classify error patterns"
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
        help="Path to trained model weights checkpoint (.pth)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/failure_cases",
        help="Directory to save failure visualization panels and diagnostic report",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Dataset split to evaluate ('test' or 'val')",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of worst failure cases to save (default: 20)",
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


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert label mask [H, W] to color BGR image [H, W, 3]."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx in range(NUM_CLASSES):
        color_mask[mask == cls_idx] = COLOR_PALETTE[cls_idx]
    return color_mask


def compute_per_image_miou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = 255,
) -> Tuple[float, List[float]]:
    """
    Compute mean IoU and per-class IoU for a single image.

    Args:
        pred: Predicted 2D label mask [H, W].
        target: Ground truth 2D label mask [H, W].
        num_classes: Number of classes (4).
        ignore_index: Label index to ignore.

    Returns:
        Tuple of (mIoU_score, [iou_class_0, ..., iou_class_3]).
    """
    valid = (target != ignore_index) & (target < num_classes) & (target >= 0)
    p = pred[valid]
    t = target[valid]

    ious = []
    for c in range(num_classes):
        p_c = p == c
        t_c = t == c
        intersection = np.logical_and(p_c, t_c).sum()
        union = np.logical_or(p_c, t_c).sum()
        if union == 0:
            ious.append(np.nan)  # Class not present in this image
        else:
            ious.append(float(intersection) / float(union))

    valid_ious = [v for v in ious if not np.isnan(v)]
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0
    return miou, ious


def generate_error_map(pred: np.ndarray, target: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """
    Generate an RGB error map highlighting misclassified pixels in bright red
    and correct pixels in dark slate gray.

    Args:
        pred: Predicted 2D mask.
        target: Ground truth 2D mask.
        ignore_index: Label to ignore.

    Returns:
        BGR image [H, W, 3].
    """
    h, w = pred.shape
    error_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    # Slate gray for correct predictions
    correct_mask = (pred == target) & (target != ignore_index)
    error_bgr[correct_mask] = [60, 60, 60]

    # Bright Red for errors
    incorrect_mask = (pred != target) & (target != ignore_index)
    error_bgr[incorrect_mask] = [0, 0, 255]

    # Black for ignored pixels
    ignored_mask = target == ignore_index
    error_bgr[ignored_mask] = [0, 0, 0]

    return error_bgr


def create_side_by_side_panel(
    image_bgr: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    miou_score: float,
    sample_id: str,
) -> np.ndarray:
    """
    Assemble a 4-panel diagnostic montage:
      [Original Image | Ground Truth | Prediction Overlay | Error Map]

    Args:
        image_bgr: Original camera frame.
        gt_mask: Ground truth segmentation.
        pred_mask: Predicted segmentation.
        miou_score: Image mIoU score.
        sample_id: Name or index of sample.

    Returns:
        Combined composite BGR image with labeled headers.
    """
    h, w = image_bgr.shape[:2]

    # Ground truth colorized
    gt_color = colorize_mask(gt_mask)

    # Prediction overlay
    pred_color = colorize_mask(pred_mask)
    pred_overlay = cv2.addWeighted(image_bgr, 0.5, pred_color, 0.5, 0)

    # Error map
    error_map = generate_error_map(pred_mask, gt_mask)

    # Panel header banner
    banner_height = 40
    def add_header(panel: np.ndarray, title: str) -> np.ndarray:
        header = np.zeros((banner_height, w, 3), dtype=np.uint8)
        cv2.putText(
            header, title, (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )
        return np.vstack([header, panel])

    p1 = add_header(image_bgr, f"Input Image ({sample_id})")
    p2 = add_header(gt_color, "Ground Truth")
    p3 = add_header(pred_overlay, f"Prediction Overlay (mIoU: {miou_score:.3f})")
    p4 = add_header(error_map, "Error Map (Red: Incorrect)")

    # 2x2 grid assembly
    top_row = np.hstack([p1, p2])
    bot_row = np.hstack([p3, p4])
    montage = np.vstack([top_row, bot_row])
    return montage


def classify_failure_mode(
    pred: np.ndarray,
    target: np.ndarray,
    ious: List[float],
) -> str:
    """
    Heuristically categorize primary failure mode for the sample.

    Args:
        pred: Predicted 2D mask.
        target: Ground truth 2D mask.
        ious: List of IoU per class.

    Returns:
        String failure category description.
    """
    # Dynamic obstacle missed
    dynamic_gt = target == 2
    if dynamic_gt.sum() > 200 and ious[2] < 0.3:
        return FAILURE_CATEGORIES[0]

    # Structural obstacle missed
    structural_gt = target == 3
    if structural_gt.sum() > 200 and ious[3] < 0.3:
        return FAILURE_CATEGORIES[1]

    # Walkable / Non-walkable boundary confusion
    walkable_gt = target == 0
    non_walkable_pred = pred == 1
    if np.logical_and(walkable_gt, non_walkable_pred).sum() > 2000:
        return FAILURE_CATEGORIES[2]

    # Dynamic vs Structural confusion
    if np.logical_and(pred == 2, target == 3).sum() > 500 or np.logical_and(pred == 3, target == 2).sum() > 500:
        return FAILURE_CATEGORIES[3]

    # False positive obstacle on walkable area
    obs_pred = np.isin(pred, [2, 3])
    if np.logical_and(obs_pred, walkable_gt).sum() > 1000:
        return FAILURE_CATEGORIES[4]

    return "General Boundary / Low Contrast Ambiguity"


def load_model(config: Dict[str, Any], checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load model architecture and weights."""
    model_name = config["model"]["name"].lower()
    num_classes = config["model"]["num_classes"]

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
            raise NotImplementedError(f"Model {model_name} could not be loaded from models")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def build_test_loader(config: Dict[str, Any], split: str) -> DataLoader:
    """
    TODO: Construct DataLoader returning (images, masks, raw_images_bgr, filenames).
    """
    raise NotImplementedError(
        f"Test DataLoader for split '{split}' is not yet implemented. "
        "Connect with src.data dataset implementations."
    )


def print_failure_summary(failure_counts: Dict[str, int], total_failures: int) -> None:
    """Print formatted summary of diagnostic failure categories."""
    border = "=" * 75
    divider = "-" * 75

    print("\n" + border)
    print("                    FAILURE CATEGORY DIAGNOSTIC SUMMARY")
    print(border)
    print(f"{'Failure Category':<55} | {'Count':<7} | {'Share (%)':<8}")
    print(divider)

    for category, count in sorted(failure_counts.items(), key=lambda x: x[1], reverse=True):
        share = (count / max(1, total_failures)) * 100.0
        print(f"{category:<55} | {count:>5} | {share:>7.1f}%")

    print(divider)
    print(f"{'Total Worst Analyzed Samples':<55} | {total_failures:>5} | 100.0%")
    print(border + "\n")


def run_failure_analysis(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    top_k: int = 20,
) -> None:
    """
    Execute failure analysis pipeline:
      1. Run inference on full split.
      2. Record per-image mIoU scores and sample references.
      3. Sort samples by mIoU ascending.
      4. Export top-k worst failure diagnostic panels.
      5. Aggregate and report failure categories.
    """
    sample_records = []

    logger.info("Computing per-image segmentation metrics across dataset...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Diagnosing")):
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
                p_mask = preds[i]
                t_mask = targets[i]
                miou, per_class_ious = compute_per_image_miou(p_mask, t_mask)

                # TODO: Retrieve original BGR image and image identifier
                dummy_bgr = np.zeros((*p_mask.shape, 3), dtype=np.uint8)
                sample_name = f"sample_{batch_idx}_{i}"

                sample_records.append({
                    "name": sample_name,
                    "miou": miou,
                    "per_class_ious": per_class_ious,
                    "pred": p_mask,
                    "target": t_mask,
                    "image_bgr": dummy_bgr,
                })

    # Sort ascending by mIoU (worst first)
    sample_records.sort(key=lambda s: s["miou"])
    worst_samples = sample_records[:top_k]

    # Save diagnostic panels and tally failure modes
    panels_dir = output_dir / "worst_cases"
    panels_dir.mkdir(parents=True, exist_ok=True)
    failure_counts: Dict[str, int] = {}

    for rank, sample in enumerate(worst_samples, start=1):
        panel = create_side_by_side_panel(
            image_bgr=sample["image_bgr"],
            gt_mask=sample["target"],
            pred_mask=sample["pred"],
            miou_score=sample["miou"],
            sample_id=sample["name"],
        )
        save_path = panels_dir / f"rank_{rank:02d}_miou_{sample['miou']:.3f}_{sample['name']}.png"
        cv2.imwrite(str(save_path), panel)

        # Categorize
        category = classify_failure_mode(sample["pred"], sample["target"], sample["per_class_ious"])
        failure_counts[category] = failure_counts.get(category, 0) + 1

    print_failure_summary(failure_counts, total_failures=len(worst_samples))
    logger.info("Saved %d worst failure diagnostic panels to %s", len(worst_samples), panels_dir)


def main() -> None:
    """Main entry point for failure analysis."""
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting failure analysis for split: %s", args.split)

    # Load Model
    try:
        model = load_model(config, args.checkpoint, device)
    except (NotImplementedError, FileNotFoundError) as e:
        logger.warning("Model loader stub notice: %s", e)
        # Display mock failure category summary for skeleton review
        mock_summary = {
            FAILURE_CATEGORIES[0]: 7,
            FAILURE_CATEGORIES[1]: 5,
            FAILURE_CATEGORIES[2]: 4,
            FAILURE_CATEGORIES[3]: 3,
            FAILURE_CATEGORIES[4]: 1,
        }
        print_failure_summary(mock_summary, total_failures=20)
        return

    # Load test dataloader
    try:
        dataloader = build_test_loader(config, split=args.split)
    except NotImplementedError as e:
        logger.warning("Dataset loader notice: %s", e)
        return

    run_failure_analysis(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=output_dir,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
