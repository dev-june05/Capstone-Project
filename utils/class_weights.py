"""Class weight computation for balancing cross-entropy loss.

In semantic segmentation for autonomous driving and pedestrian navigation,
classes like Walkable (road/sidewalk) and Non-walkable (buildings/vegetation)
dominate the pixel count, while Dynamic Obstacles (pedestrians/bicycles) and
Structural Obstacles (poles/traffic signs) account for < 2-5% of total pixels.

This module provides median-frequency balancing and inverse frequency weighting
to compute loss balancing weights for torch.nn.CrossEntropyLoss(weight=...).
"""

import math
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.label_mapping import IGNORE_INDEX


def compute_class_weights(
    dataset: Dataset,
    num_classes: int = 4,
    method: str = "median_freq",
    ignore_index: int = IGNORE_INDEX,
    max_samples: Optional[int] = None,
    batch_size: int = 8,
    num_workers: int = 0,
) -> torch.Tensor:
    """Compute loss balancing class weights by iterating over a dataset.

    Methods:
        1. 'median_freq' (Median Frequency Balancing, Eigen & Fergus, ICCV 2015):
           weight[c] = median(frequencies) / frequency[c]
           where frequency[c] = total_pixels[c] / total_valid_pixels.

        2. 'inverse' (Inverse Frequency Balancing):
           weight[c] = 1.0 / (frequency[c] + epsilon), normalized so mean(weight) == 1.0.

        3. 'enet' (ENet log-frequency balancing, Paszke et al., 2016):
           weight[c] = 1.0 / ln(c_factor + frequency[c]).

    Args:
        dataset (torch.utils.data.Dataset): Dataset yielding {'mask': tensor/ndarray}
            or (image, mask).
        num_classes (int): Number of semantic classes (default: 4).
        method (str): Weight balancing method: 'median_freq', 'inverse', or 'enet'.
        ignore_index (int): Pixel label index to exclude from counts (default: 255).
        max_samples (int, optional): If set, compute frequencies on a random or sequential
            subset of dataset samples to speed up computation on large datasets.
        batch_size (int): Batch size for DataLoader scanning (default: 8).
        num_workers (int): Number of DataLoader worker processes (default: 0).

    Returns:
        torch.Tensor: Float tensor of shape (num_classes,) containing normalized weights.

    Raises:
        ValueError: If unsupported balancing method is specified or dataset is empty.
    """
    total_samples = len(dataset)
    if total_samples == 0:
        raise ValueError("Cannot compute class weights on an empty dataset.")

    method_key = method.lower().strip()
    if method_key not in ("median_freq", "inverse", "enet"):
        raise ValueError(
            f"Unsupported method '{method}'. Supported: 'median_freq', 'inverse', 'enet'."
        )

    # Accumulate pixel counts across dataset
    class_pixel_counts = np.zeros(num_classes, dtype=np.int64)

    # DataLoader for accelerated batched iteration
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    samples_processed = 0
    progress_bar = tqdm(
        total=max_samples if max_samples is not None else total_samples,
        desc=f"Computing class weights ({method_key})",
        unit="samples",
    )

    for batch in loader:
        # Extract mask from batch dictionary or tuple
        if isinstance(batch, dict) and "mask" in batch:
            masks = batch["mask"]
        elif isinstance(batch, (list, tuple)):
            # Assume mask is last item (image, mask)
            masks = batch[-1]
        elif isinstance(batch, torch.Tensor):
            masks = batch
        else:
            raise TypeError(f"Unexpected batch format received: {type(batch)}")

        if isinstance(masks, torch.Tensor):
            masks_np = masks.detach().cpu().numpy()
        else:
            masks_np = np.asarray(masks)

        # Flatten batch and filter ignore_index
        valid_pixels = (masks_np != ignore_index) & (masks_np >= 0) & (masks_np < num_classes)
        valid_labels = masks_np[valid_pixels]

        # Count frequencies
        counts = np.bincount(valid_labels, minlength=num_classes)
        class_pixel_counts += counts

        current_batch_size = masks_np.shape[0] if masks_np.ndim > 2 else 1
        samples_processed += current_batch_size
        progress_bar.update(current_batch_size)

        if max_samples is not None and samples_processed >= max_samples:
            break

    progress_bar.close()

    total_valid_pixels = np.sum(class_pixel_counts)
    if total_valid_pixels == 0:
        raise RuntimeError("No valid labeled pixels found across the evaluated dataset.")

    # Compute frequency per class: f[c] = total_pixels[c] / total_valid_pixels
    frequencies = class_pixel_counts.astype(np.float64) / total_valid_pixels

    # Avoid zero division for classes that might not appear in sample subset
    eps = 1e-6
    safe_frequencies = np.where(frequencies == 0, eps, frequencies)

    if method_key == "median_freq":
        # Median frequency balancing
        # TODO: Fine-tune class weight clipping bounds if obstacle weights become too extreme
        median_val = np.median(safe_frequencies)
        weights = median_val / safe_frequencies

    elif method_key == "inverse":
        # Inverse frequency balancing: weight proportional to 1 / frequency
        weights = 1.0 / (safe_frequencies + eps)
        # Normalize weights so that the average weight is 1.0
        weights = weights / np.mean(weights)

    elif method_key == "enet":
        # ENet log-frequency balancing: weight[c] = 1.0 / ln(c_factor + f[c])
        c_factor = 1.02
        weights = 1.0 / np.log(c_factor + safe_frequencies)

    # Convert to torch tensor with float32
    weights_tensor = torch.from_numpy(weights).float()

    return weights_tensor
