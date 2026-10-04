#!/usr/bin/env python3
"""
train.py
--------
Training script for 4-class semantic segmentation (Walkable, Non-walkable,
Dynamic Obstacle, Structural Obstacle) using BiSeNetV2 and Fast-SCNN.

Datasets: Cityscapes + IDD (India Driving Dataset)
Deployment Target: Jetson Orin Nano (TensorRT FP16/INT8)
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

# Optional experiment tracking via Weights & Biases
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Class definition for 4-class segmentation task
CLASSES = [
    "walkable",             # Class 0: Sidewalk, footpath, paved pedestrian path
    "non_walkable",         # Class 1: Road, terrain, vegetation, sky, background
    "dynamic_obstacle",     # Class 2: Pedestrians, riders, vehicles, bicycles, animals
    "structural_obstacle",  # Class 3: Poles, fences, walls, barriers, traffic signs
]
NUM_CLASSES = len(CLASSES)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TrainSegmentation")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for model training."""
    parser = argparse.ArgumentParser(
        description="Train 4-class semantic segmentation models (BiSeNetV2 / Fast-SCNN)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training YAML configuration file (e.g., configs/train_bisenetv2.yaml)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional path to checkpoint .pth to resume training from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run training on ('cuda', 'cuda:0', 'cpu')",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to configuration file.

    Returns:
        Configuration dictionary.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded configuration from %s", config_path)
    return config


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducible experiments.

    Args:
        seed: Random seed value.
        deterministic: If True, configure cuDNN backend for deterministic computation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Deterministic execution enabled (cuDNN deterministic=True, benchmark=False)")
    else:
        torch.backends.cudnn.benchmark = True
    logger.info("Random seed fixed to %d", seed)


def compute_class_weights(
    dataloader_or_dataset: Any,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    """
    Compute class weights automatically to mitigate class imbalance.

    TODO: Implement automated class weight calculation:
      - Iterate over training mask annotations.
      - Accumulate pixel frequencies per class: freq[c] = total_pixels[c] / total_valid_pixels.
      - Apply Median Frequency Balancing:
            weight[c] = median(freq) / freq[c]
        or ENet log inverse frequency:
            weight[c] = 1.0 / ln(c_param + freq[c])
      - Normalize weights so their mean or sum is bounded.

    Args:
        dataloader_or_dataset: Training dataset or dataloader.
        num_classes: Total number of semantic classes (default: 4).

    Returns:
        Tensor of shape [num_classes] with weighting factors.
    """
    logger.info("Calculating auto class weights across training dataset...")
    # TODO: Replace placeholder weights with computed frequencies from the dataset.
    # Placeholder: uniform weights normalized to 1.0
    weights = torch.ones(num_classes, dtype=torch.float32)
    logger.info("Initialized class weights: %s", weights.tolist())
    return weights


def build_datasets(config: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Build training and validation datasets.

    TODO:
      - Import Cityscapes and IDD dataset loaders from `src.data.datasets`.
      - Build Albumentations augmentation pipeline from config['augmentation']
        (RandomCrop 512x1024, HorizontalFlip, ColorJitter, GaussianBlur,
        BrightnessContrast, RandomShadow, CopyPaste).
      - Remap Cityscapes (19 classes) and IDD labels into the 4 target classes:
          0: Walkable
          1: Non-walkable
          2: Dynamic Obstacle
          3: Structural Obstacle
      - If multiple datasets specified (Cityscapes + IDD), concatenate via torch.utils.data.ConcatDataset.

    Args:
        config: Full configuration dictionary.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    # TODO: Replace with actual Dataset instances (e.g., JointCityscapesIDDDataset)
    logger.info("Building datasets: train=%s, val=%s",
                [d["name"] for d in config["dataset"]["train"]],
                [d["name"] for d in config["dataset"]["val"]])
    raise NotImplementedError(
        "Dataset construction is not yet implemented. "
        "Implement Cityscapes and IDD dataset loaders in src/data/ and connect here."
    )


def build_dataloaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, Optional[torch.Tensor]]:
    """
    Construct PyTorch DataLoaders for training and validation splits.

    Args:
        config: Full configuration dictionary.

    Returns:
        Tuple of (train_loader, val_loader, class_weights_tensor).
    """
    batch_size = config["training"]["batch_size"]
    num_workers = config["dataset"].get("num_workers", 4)

    # TODO: Call build_datasets(config)
    # train_dataset, val_dataset = build_datasets(config)

    # Compute class weights if requested
    class_weights = None
    if config["training"].get("class_weights") == "auto":
        # class_weights = compute_class_weights(train_dataset, config["model"]["num_classes"])
        class_weights = torch.ones(config["model"]["num_classes"], dtype=torch.float32)

    # TODO: Initialize DataLoaders:
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     pin_memory=True,
    #     drop_last=False,
    # )
    # return train_loader, val_loader, class_weights

    raise NotImplementedError(
        "DataLoader builder is stubbed. Connect with real datasets once implemented."
    )


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    Instantiate semantic segmentation model based on configuration.

    Supports:
      - 'bisenetv2': Detail Branch + Semantic Branch + Aggregation Layer + 4 Aux Heads
      - 'fastscnn': Learning to Downsample + Global Feature Extractor + Feature Fusion + Classifier

    Args:
        config: Full configuration dictionary.

    Returns:
        Instantiated nn.Module ready for training.
    """
    model_name = config["model"]["name"].lower()
    num_classes = config["model"]["num_classes"]
    auxiliary_loss = config["model"].get("auxiliary_loss", False)

    logger.info("Building model '%s' (num_classes=%d, aux_loss=%s)",
                model_name, num_classes, auxiliary_loss)

    try:
        from models import get_model
        try:
            model = get_model(name=model_name, num_classes=num_classes)
        except TypeError:
            model = get_model(name=model_name, num_classes=num_classes, auxiliary_loss=auxiliary_loss)
        return model
    except ImportError:
        try:
            from src.models import get_model
            try:
                model = get_model(name=model_name, num_classes=num_classes)
            except TypeError:
                model = get_model(name=model_name, num_classes=num_classes, auxiliary_loss=auxiliary_loss)
            return model
        except (ImportError, AttributeError):
            logger.warning(
                "Could not import get_model from models. Stubbing model instantiation."
            )
            # TODO: Implement BiSeNetV2 and Fast-SCNN in models/
            raise NotImplementedError(
                f"Model '{model_name}' builder not found. Implement models/get_model."
            )


def build_criterion(
    config: Dict[str, Any],
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Build combined loss function (OHEM Cross-Entropy + Dice Loss).

    TODO:
      - Implement OhemCrossEntropy2d: keep top hard negative pixels whose loss > threshold
        or retain at least min_kept samples per batch.
      - Implement MulticlassDiceLoss: soft dice loss averaged over classes.
      - Combine: total_loss = loss_ce + dice_weight * loss_dice.
      - If auxiliary heads present (BiSeNetV2), compute weighted auxiliary losses:
            total = main_loss + sum(aux_weight[i] * aux_loss[i])

    Args:
        config: Full configuration dictionary.
        class_weights: Optional per-class weighting tensor.

    Returns:
        Loss criterion module.
    """
    loss_type = config["training"].get("loss", "ohem_ce_dice")
    logger.info("Building criterion: %s (class_weights=%s)", loss_type, class_weights is not None)

    # TODO: Replace with custom OhemDiceLoss module from src.losses
    try:
        from src.losses import get_criterion
        return get_criterion(config=config, class_weights=class_weights)
    except ImportError:
        # Fallback to standard CrossEntropyLoss with weights for baseline skeleton
        logger.warning("Using standard CrossEntropyLoss fallback. Custom OHEM+Dice loss to be connected.")
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)


def build_optimizer_and_scheduler(
    model: nn.Module,
    config: Dict[str, Any],
    total_steps: int,
) -> Tuple[torch.optim.Optimizer, Any]:
    """
    Build AdamW optimizer and Polynomial LR scheduler with warmup.

    Learning rate schedule:
      - Warmup phase (first warmup_epochs): linear scale from 0 to lr
      - Polynomial decay phase: lr * (1 - step / max_steps) ** poly_power

    Args:
        model: Segmentation network.
        config: Full configuration dictionary.
        total_steps: Total training iterations across all epochs.

    Returns:
        Tuple of (optimizer, lr_scheduler).
    """
    lr = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"].get("weight_decay", 0.01))
    opt_name = config["training"].get("optimizer", "adamw").lower()

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    poly_power = float(config["training"].get("poly_power", 0.9))
    warmup_epochs = int(config["training"].get("warmup_epochs", 5))
    total_epochs = int(config["training"]["epochs"])

    # TODO: Build PolynomialLR with Warmup scheduler
    def lr_lambda(current_step: int) -> float:
        warmup_steps = int((warmup_epochs / max(1, total_epochs)) * total_steps)
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return (1.0 - progress) ** poly_power

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    device: torch.device,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """
    Run one epoch of training.

    Args:
        model: Segmentation model.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: PyTorch optimizer.
        scheduler: Iteration-level learning rate scheduler.
        epoch: Current epoch index.
        device: Device to execute computation on.
        config: Full configuration dictionary.

    Returns:
        Dictionary of averaged training metrics.
    """
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch_idx, batch in enumerate(pbar):
        # TODO: Handle batch unpack (images: [B, 3, H, W], masks: [B, H, W])
        images, masks = batch
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass (handles optional auxiliary logits)
        outputs = model(images)
        if isinstance(outputs, dict):
            main_out = outputs.get("main", next(iter(outputs.values())))
            aux_outs = [v for k, v in outputs.items() if k != "main"]
        elif isinstance(outputs, (tuple, list)):
            main_out = outputs[0]
            aux_outs = list(outputs[1:])
        else:
            main_out = outputs
            aux_outs = []

        loss = criterion(main_out, masks)
        if aux_outs:
            aux_weights = config["model"].get("aux_weights", [1.0] * len(aux_outs))
            for aux_out, w in zip(aux_outs, aux_weights):
                loss = loss + w * criterion(aux_out, masks)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.6f}"})

    epoch_loss = running_loss / max(1, total_batches)
    return {"train_loss": epoch_loss, "learning_rate": optimizer.param_groups[0]["lr"]}


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    device: torch.device,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """
    Run evaluation on the validation split.

    TODO:
      - Compute confusion matrix across batches.
      - Calculate:
          - Mean IoU (mIoU)
          - Per-class IoU: [Walkable, Non-walkable, Dynamic Obstacle, Structural Obstacle]
          - Obstacle Recall (Class 2 + Class 3)
          - Missed Obstacle Rate (1.0 - Obstacle Recall)
          - Mean Dice / F1 score

    Args:
        model: Segmentation model.
        dataloader: Validation dataloader.
        criterion: Loss function.
        epoch: Current epoch index.
        device: Device to execute computation on.
        config: Full configuration dictionary.

    Returns:
        Dictionary containing validation metrics.
    """
    model.eval()
    running_loss = 0.0
    total_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)
    for batch in pbar:
        # TODO: Replace with real evaluation inference
        images, masks = batch
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(images)
        if isinstance(outputs, dict):
            outputs = outputs.get("main", next(iter(outputs.values())))
        elif isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        loss = criterion(outputs, masks)
        running_loss += loss.item()

    val_loss = running_loss / max(1, total_batches)

    # TODO: Replace dummy values with actual computed segmentation metrics
    metrics = {
        "val_loss": val_loss,
        "val_mIoU": 0.0,
        "val_iou_walkable": 0.0,
        "val_iou_non_walkable": 0.0,
        "val_iou_dynamic_obstacle": 0.0,
        "val_iou_structural_obstacle": 0.0,
        "val_obstacle_recall": 0.0,
        "val_missed_obstacle_rate": 1.0,
    }
    return metrics


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    save_dir: str,
    filename: str = "last.pth",
) -> None:
    """
    Save training checkpoint and update best model.

    Args:
        state: State dictionary containing model_state, optimizer_state, epoch, metrics.
        is_best: True if current checkpoint achieved best validation mIoU.
        save_dir: Directory where checkpoints are saved.
        filename: Name of the checkpoint file.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    target_file = save_path / filename
    torch.save(state, target_file)
    logger.info("Saved checkpoint to %s", target_file)

    if is_best:
        best_file = save_path / "best.pth"
        torch.save(state, best_file)
        logger.info(">>> New best model saved to %s (mIoU: %.4f)",
                    best_file, state.get("mIoU", 0.0))


def main() -> None:
    """Main execution pipeline for model training."""
    args = parse_args()
    config = load_config(args.config)

    # Set reproducibility
    repro = config.get("reproducibility", {})
    set_seed(
        seed=repro.get("seed", 42),
        deterministic=repro.get("deterministic", True),
    )

    device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # Initialize experiment tracking
    log_cfg = config.get("logging", {})
    save_dir = log_cfg.get("save_dir", "checkpoints")
    save_every = log_cfg.get("save_every", 10)

    use_wandb = HAS_WANDB and ("wandb_project" in log_cfg)
    if use_wandb:
        wandb.init(
            project=log_cfg.get("wandb_project", "capstone-walkability"),
            name=f"{config['model']['name']}-{config['training']['optimizer']}",
            config=config,
        )
        logger.info("Weights & Biases initialized.")
    else:
        logger.info("Wandb tracking skipped (not installed or disabled).")

    # Build model
    try:
        model = build_model(config)
        model.to(device)
    except NotImplementedError as e:
        logger.warning("Model builder stub notice: %s", e)
        logger.info("Training script structure verified. Exiting skeleton run gracefully.")
        return

    # Build DataLoaders & Class Weights
    try:
        train_loader, val_loader, class_weights = build_dataloaders(config)
    except NotImplementedError as e:
        logger.warning("Dataset/DataLoader step: %s", e)
        logger.info("Training script structure verified. Exiting skeleton run gracefully.")
        return

    # Build Loss Criterion
    criterion = build_criterion(config, class_weights=class_weights)
    criterion.to(device)

    # Build Optimizer and Scheduler
    epochs = int(config["training"]["epochs"])
    total_steps = epochs * len(train_loader)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config, total_steps)

    # Optional Resume
    start_epoch = 1
    best_miou = 0.0
    if args.resume and os.path.isfile(args.resume):
        logger.info("Resuming checkpoint from %s", args.resume)
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint.get("best_miou", 0.0)

    # Training Loop
    logger.info("Starting training loop from epoch %d to %d...", start_epoch, epochs)
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            device=device,
            config=config,
        )

        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            epoch=epoch,
            device=device,
            config=config,
        )

        current_miou = val_metrics.get("val_mIoU", 0.0)
        is_best = current_miou > best_miou
        if is_best:
            best_miou = current_miou

        # Log metrics
        combined_metrics = {**train_metrics, **val_metrics, "epoch": epoch}
        logger.info(
            "Epoch %d/%d - Train Loss: %.4f | Val Loss: %.4f | Val mIoU: %.4f",
            epoch, epochs,
            train_metrics["train_loss"],
            val_metrics["val_loss"],
            current_miou,
        )

        if use_wandb:
            wandb.log(combined_metrics, step=epoch)

        # Save checkpoint
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "mIoU": current_miou,
            "best_miou": best_miou,
            "config": config,
        }

        save_checkpoint(checkpoint_state, is_best=is_best, save_dir=save_dir, filename="last.pth")
        if epoch % save_every == 0:
            save_checkpoint(
                checkpoint_state,
                is_best=False,
                save_dir=save_dir,
                filename=f"epoch_{epoch}.pth",
            )

    if use_wandb:
        wandb.finish()
    logger.info("Training complete. Best Validation mIoU: %.4f", best_miou)


if __name__ == "__main__":
    main()
