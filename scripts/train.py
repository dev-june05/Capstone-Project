#!/usr/bin/env python3
import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import datasets and metrics
from utils.dataset import CityscapesDataset, IDDDataset, CampusDataset, CombinedDataset
from utils.metrics import SegmentationMetrics
from utils.augmentations import get_train_augmentations, get_val_augmentations
from models import get_model

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

CLASSES = ["walkable", "non_walkable", "dynamic_obstacle", "structural_obstacle"]
NUM_CLASSES = len(CLASSES)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("TrainSegmentation")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 4-class semantic segmentation models")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, weight=None):
        super(OhemCrossEntropy2d, self).__init__()
        self.thresh = thresh
        self.min_kept = max(1, min_kept)
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(score, size=(h, w), mode='bilinear', align_corners=False)
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_index
        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_index] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)
        pixel_losses = pixel_losses[mask][pred < threshold]
        if pixel_losses.numel() == 0:
            return score.new_zeros(1)[0]
        return pixel_losses.mean()

def build_datasets(config: Dict[str, Any]) -> Tuple[Any, Any]:
    train_datasets = []
    h = config["model"].get("input_height", 512)
    w = config["model"].get("input_width", 1024)
    train_aug = get_train_augmentations(height=h, width=w)
    
    for d_cfg in config["dataset"]["train"]:
        if d_cfg["name"] == "cityscapes":
            train_datasets.append(CityscapesDataset(d_cfg["root"], d_cfg["split"], transform=train_aug))
        elif d_cfg["name"] == "idd":
            train_datasets.append(IDDDataset(d_cfg["root"], d_cfg["split"], transform=train_aug))
        elif d_cfg["name"] == "campus":
            train_datasets.append(CampusDataset(d_cfg["root"], d_cfg["split"], transform=train_aug))
            
    val_datasets = []
    val_aug = get_val_augmentations(height=h, width=w)
    for d_cfg in config["dataset"]["val"]:
        if d_cfg["name"] == "cityscapes":
            val_datasets.append(CityscapesDataset(d_cfg["root"], d_cfg["split"], transform=val_aug))
        elif d_cfg["name"] == "idd":
            val_datasets.append(IDDDataset(d_cfg["root"], d_cfg["split"], transform=val_aug))
        elif d_cfg["name"] == "campus":
            val_datasets.append(CampusDataset(d_cfg["root"], d_cfg["split"], transform=val_aug))

    train_datasets = [ds for ds in train_datasets if len(ds) > 0]
    val_datasets = [ds for ds in val_datasets if len(ds) > 0]
    if len(train_datasets) == 0:
        raise RuntimeError("No training images found! Please verify that your dataset is extracted to the paths specified in your config (e.g., data/cityscapes).")
    if len(val_datasets) == 0:
        raise RuntimeError("No validation images found! Please verify that your dataset is extracted to the paths specified in your config.")

    train_ds = CombinedDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_ds = CombinedDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
    logger.info(f"Loaded {len(train_ds)} training samples and {len(val_ds)} validation samples across {len(train_datasets)} active dataset(s).")
    return train_ds, val_ds

def build_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, Optional[torch.Tensor]]:
    train_dataset, val_dataset = build_datasets(config)
    batch_size = config["training"]["batch_size"]
    num_workers = config["dataset"].get("num_workers", 4)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    
    class_weights = None
    if config["training"].get("class_weights") == "auto":
        class_weights = torch.ones(config["model"]["num_classes"], dtype=torch.float32) # Can be updated with compute_class_weights
    
    return train_loader, val_loader, class_weights

def build_model_fn(config: Dict[str, Any]) -> nn.Module:
    model_name = config["model"]["name"].lower()
    num_classes = config["model"]["num_classes"]
    model = get_model(name=model_name, num_classes=num_classes)
    return model

def build_criterion(config: Dict[str, Any], class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    thresh = float(config["training"].get("ohem_threshold", 0.7))
    return OhemCrossEntropy2d(thresh=thresh, weight=class_weights)

def build_optimizer_and_scheduler(model: nn.Module, config: Dict[str, Any], total_steps: int) -> Tuple[torch.optim.Optimizer, Any]:
    lr = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"].get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    poly_power = float(config["training"].get("poly_power", 0.9))
    warmup_epochs = int(config["training"].get("warmup_epochs", 5))
    total_epochs = int(config["training"]["epochs"])
    
    def lr_lambda(current_step: int) -> float:
        warmup_steps = int((warmup_epochs / max(1, total_epochs)) * total_steps)
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return (1.0 - progress) ** poly_power

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, epoch, device, config):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for batch in pbar:
        images, masks = batch["image"].to(device, non_blocking=True), batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad()
        
        outputs = model(images)
        if isinstance(outputs, dict):
            main_out = outputs["main"]
            aux_outs = [v for k, v in outputs.items() if k.startswith("aux")]
        else:
            main_out = outputs
            aux_outs = []

        loss = criterion(main_out, masks)
        if aux_outs and config["model"].get("auxiliary_loss", False):
            aux_weights = config["model"].get("aux_weights", [1.0]*len(aux_outs))
            for aux_out, w in zip(aux_outs, aux_weights):
                loss = loss + w * criterion(aux_out, masks)
                
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})
        
    return {"train_loss": running_loss / max(1, len(dataloader)), "learning_rate": optimizer.param_groups[0]["lr"]}

@torch.no_grad()
def validate(model, dataloader, criterion, epoch, device, config):
    model.eval()
    running_loss = 0.0
    metrics_accumulator = SegmentationMetrics(num_classes=config["model"]["num_classes"])
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)
    for batch in pbar:
        images, masks = batch["image"].to(device, non_blocking=True), batch["mask"].to(device, non_blocking=True)
        outputs = model(images)
        if isinstance(outputs, dict):
            outputs = outputs["main"]
            
        loss = criterion(outputs, masks)
        running_loss += loss.item()
        
        metrics_accumulator.update(torch.argmax(outputs, dim=1), masks)
        
    summary = metrics_accumulator.summary()
    
    return {
        "val_loss": running_loss / max(1, len(dataloader)),
        "val_mIoU": summary["mIoU"],
        "val_iou_walkable": summary["per_class_iou"].get("Walkable", 0.0),
        "val_iou_non_walkable": summary["per_class_iou"].get("Non-walkable", 0.0),
        "val_iou_dynamic_obstacle": summary["per_class_iou"].get("Dynamic Obstacle", 0.0),
        "val_iou_structural_obstacle": summary["per_class_iou"].get("Structural Obstacle", 0.0),
        "val_obstacle_recall": summary["obstacle_recall"].get("combined_obstacle_recall", 0.0),
        "val_missed_obstacle_rate": summary["missed_obstacle_rate_wz"],
    }

def save_checkpoint(state, is_best, save_dir, filename="last.pth"):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    target_file = save_path / filename
    torch.save(state, target_file)
    if is_best:
        torch.save(state, save_path / "best.pth")

    # If running in Google Colab with Drive mounted, automatically mirror to Drive every epoch
    gdrive_mount = Path("/content/drive/MyDrive")
    if gdrive_mount.exists():
        try:
            colab_backup = gdrive_mount / "capstone_checkpoints"
            colab_backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target_file, colab_backup / filename)
            if is_best:
                shutil.copy2(save_path / "best.pth", colab_backup / "best.pth")
        except Exception as e:
            logger.warning(f"Failed to auto-backup checkpoint to Google Drive: {e}")

def main():
    args = parse_args()
    config = load_config(args.config)
    
    set_seed(config.get("reproducibility", {}).get("seed", 42), config.get("reproducibility", {}).get("deterministic", True))
    device = torch.device(args.device)
    
    save_dir = config.get("logging", {}).get("save_dir", "checkpoints")
    use_wandb = HAS_WANDB and "wandb_project" in config.get("logging", {})
    if use_wandb:
        wandb.init(project=config["logging"]["wandb_project"], name=f"{config['model']['name']}", config=config)

    model = build_model_fn(config).to(device)
    train_loader, val_loader, class_weights = build_dataloaders(config)
    if class_weights is not None: class_weights = class_weights.to(device)
    criterion = build_criterion(config, class_weights).to(device)
    
    epochs = int(config["training"]["epochs"])
    optimizer, scheduler = build_optimizer_and_scheduler(model, config, epochs * len(train_loader))
    
    start_epoch, best_miou = 1, 0.0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint.get("best_miou", 0.0)

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, device, config)
        val_metrics = validate(model, val_loader, criterion, epoch, device, config)
        
        current_miou = val_metrics["val_mIoU"]
        is_best = current_miou > best_miou
        if is_best: best_miou = current_miou
        
        logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_metrics['train_loss']:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | Val mIoU: {current_miou:.4f}")
        if use_wandb:
            wandb.log({**train_metrics, **val_metrics, "epoch": epoch}, step=epoch)
            
        checkpoint_state = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "mIoU": current_miou, "best_miou": best_miou, "config": config}
        save_checkpoint(checkpoint_state, is_best, save_dir, filename="last.pth")
        if epoch % config.get("logging", {}).get("save_every", 10) == 0:
            save_checkpoint(checkpoint_state, False, save_dir, filename=f"epoch_{epoch}.pth")

    if use_wandb: wandb.finish()

if __name__ == "__main__":
    main()
