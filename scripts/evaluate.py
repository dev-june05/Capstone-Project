#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset import CityscapesDataset, IDDDataset, CampusDataset, CombinedDataset
from utils.metrics import SegmentationMetrics
from utils.augmentations import get_val_augmentations
from models import get_model

CLASSES = ["Walkable", "Non-walkable", "Dynamic Obstacle", "Structural Obstacle"]
NUM_CLASSES = len(CLASSES)

COLOR_PALETTE = np.array([[0, 200, 0], [128, 128, 128], [0, 0, 220], [0, 165, 255]], dtype=np.uint8)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("EvaluateSegmentation")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 4-class semantic segmentation models")
    parser.add_argument("--config", type=str, required=True, help="Path to config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--output-dir", type=str, default="eval_results")
    parser.add_argument("--save-visualizations", action="store_true", default=True)
    parser.add_argument("--max-vis-samples", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx in range(NUM_CLASSES):
        color_mask[mask == cls_idx] = COLOR_PALETTE[cls_idx]
    return color_mask

def create_overlay(image_bgr: np.ndarray, pred_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    color_mask = colorize_mask(pred_mask)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, color_mask, alpha, 0)

def load_model(config: Dict[str, Any], checkpoint_path: str, device: torch.device) -> nn.Module:
    model_name = config["model"]["name"].lower()
    num_classes = config["model"]["num_classes"]
    model = get_model(name=model_name, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def build_eval_dataloader(config: Dict[str, Any], split: str) -> DataLoader:
    h = config["model"].get("input_height", 512)
    w = config["model"].get("input_width", 1024)
    val_aug = get_val_augmentations(height=h, width=w)
    
    datasets = []
    # If the config has a 'test' split, use it, else fallback to 'val' section of config
    config_split = "test" if split == "test" and "test" in config["dataset"] else "val"
    
    for d_cfg in config["dataset"][config_split]:
        if d_cfg["name"] == "cityscapes":
            datasets.append(CityscapesDataset(d_cfg["root"], d_cfg["split"], transform=val_aug))
        elif d_cfg["name"] == "idd":
            datasets.append(IDDDataset(d_cfg["root"], d_cfg["split"], transform=val_aug))
        elif d_cfg["name"] == "campus":
            datasets.append(CampusDataset(d_cfg["root"], d_cfg["split"], transform=val_aug))

    ds = CombinedDataset(datasets) if len(datasets) > 1 else datasets[0]
    return DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=config["dataset"].get("num_workers", 4))

def print_results_table(metrics: Dict[str, Any]) -> None:
    border = "=" * 65
    divider = "-" * 65
    print("\n" + border)
    print("        4-CLASS SEMANTIC SEGMENTATION EVALUATION RESULTS")
    print(border)
    print(f"{'Class Name':<25} | {'IoU (%)':<15} | {'Dice / F1 (%)':<15}")
    print(divider)

    ious = metrics.get("per_class_iou", {})
    dices = metrics.get("per_class_dice", {})

    for cls_name in CLASSES:
        iou_val = ious.get(cls_name, 0.0) * 100.0
        dice_val = dices.get(cls_name, 0.0) * 100.0
        print(f"{cls_name:<25} | {iou_val:>12.2f} % | {dice_val:>12.2f} %")

    print(divider)
    print(f"{'Mean IoU (mIoU)':<25} | {metrics.get('mIoU', 0.0) * 100.0:>12.2f} % |")
    print(divider)
    print("  CRITICAL SAFETY METRICS (Obstacle Detection)")
    print(divider)
    
    obs = metrics.get("obstacle_recall", {})
    print(f"{'Obstacle Recall':<30} : {obs.get('combined_obstacle_recall', 0.0) * 100.0:>8.2f} %")
    print(f"{'Missed Obstacle Rate':<30} : {metrics.get('missed_obstacle_rate_wz', 0.0) * 100.0:>8.2f} %")
    print(border + "\n")

def evaluate(model, dataloader, device, output_dir, save_visualizations, max_vis_samples):
    metrics_tracker = SegmentationMetrics(num_classes=NUM_CLASSES)
    vis_dir = output_dir / "overlays"
    if save_visualizations: vis_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images, masks = batch["image"].to(device), batch["mask"]
            outputs = model(images)
            if isinstance(outputs, dict): outputs = outputs["main"]
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets = masks.numpy()

            for i in range(preds.shape[0]):
                metrics_tracker.update(preds[i], targets[i])
                if save_visualizations and saved_count < max_vis_samples:
                    # Create overlay using an empty image since original image is normalized
                    # In a real scenario, we could unnormalize the image tensor here.
                    dummy_img = np.zeros((*preds[i].shape, 3), dtype=np.uint8)
                    overlay = create_overlay(dummy_img, preds[i])
                    cv2.imwrite(str(vis_dir / f"pred_sample_{saved_count:04d}.png"), overlay)
                    saved_count += 1

    return metrics_tracker.summary()

def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(config, args.checkpoint, device)
    dataloader = build_eval_dataloader(config, split=args.split)
    
    results = evaluate(model, dataloader, device, output_dir, args.save_visualizations, args.max_vis_samples)
    print_results_table(results)

    metrics_file = output_dir / f"evaluation_{args.split}_metrics.yaml"
    with open(metrics_file, "w", encoding="utf-8") as f:
        yaml.dump(results, f, sort_keys=False)
    logger.info(f"Saved evaluation summary to {metrics_file}")

if __name__ == "__main__":
    main()
