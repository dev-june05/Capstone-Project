"""Label mapping and remapping utilities for semantic segmentation.

This module unifies Cityscapes, IDD (Indian Driving Dataset), and custom campus
datasets into a common 4-class taxonomy:
    0: Walkable (roads, sidewalks, drivable paths)
    1: Non-walkable (buildings, vegetation, walls, fences, sky, terrain)
    2: Dynamic Obstacle (pedestrians, riders, vehicles, animals)
    3: Structural Obstacle (poles, traffic signs, traffic lights, curbs, billboards)
    255: Ignore / Unlabeled (void, ego-vehicle, hood, etc.)
"""

from typing import Dict, Tuple, Union
import numpy as np

# Ignore index used across loss functions and evaluation metrics
IGNORE_INDEX: int = 255

# Target project classes
PROJECT_CLASSES: Dict[int, str] = {
    0: "Walkable",
    1: "Non-walkable",
    2: "Dynamic Obstacle",
    3: "Structural Obstacle",
}

# Visualization color palette (RGB format)
PROJECT_PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (0, 255, 0),      # Walkable -> Green
    1: (128, 128, 128),  # Non-walkable -> Gray
    2: (255, 0, 0),      # Dynamic Obstacle -> Red
    3: (255, 165, 0),    # Structural Obstacle -> Orange
    255: (0, 0, 0),      # Ignore / Void -> Black
}

# ==============================================================================
# Cityscapes to Project Mapping
# Based on original Cityscapes IDs (0-33, -1)
# ==============================================================================
CITYSCAPES_TO_PROJECT: Dict[int, int] = {
    # Walkable (Class 0)
    7: 0,   # road
    8: 0,   # sidewalk

    # Non-walkable (Class 1)
    11: 1,  # building
    12: 1,  # wall
    13: 1,  # fence
    21: 1,  # vegetation
    22: 1,  # terrain
    23: 1,  # sky

    # Dynamic Obstacle (Class 2)
    24: 2,  # person
    25: 2,  # rider
    26: 2,  # car
    27: 2,  # truck
    28: 2,  # bus
    31: 2,  # train
    32: 2,  # motorcycle
    33: 2,  # bicycle

    # Structural Obstacle (Class 3)
    17: 3,  # pole
    19: 3,  # traffic light
    20: 3,  # traffic sign

    # Explicitly ignored classes (mapping to 255)
    -1: IGNORE_INDEX,  # unlabeled
    0: IGNORE_INDEX,   # unlabeled
    1: IGNORE_INDEX,   # ego vehicle
    2: IGNORE_INDEX,   # rectification border
    3: IGNORE_INDEX,   # out of roi
    4: IGNORE_INDEX,   # static
    5: IGNORE_INDEX,   # dynamic
    6: IGNORE_INDEX,   # ground
    9: IGNORE_INDEX,   # parking
    10: IGNORE_INDEX,  # rail track
    14: IGNORE_INDEX,  # guard rail
    15: IGNORE_INDEX,  # bridge
    16: IGNORE_INDEX,  # tunnel
    18: IGNORE_INDEX,  # polegroup
    29: IGNORE_INDEX,  # caravan
    30: IGNORE_INDEX,  # trailer
    255: IGNORE_INDEX, # void
}

# ==============================================================================
# IDD (Indian Driving Dataset) to Project Mapping
# Based on IDD Level-3 / Level-4 segmentation class IDs
# ==============================================================================
IDD_TO_PROJECT: Dict[int, int] = {
    # Walkable (Class 0)
    0: 0,   # road
    1: 0,   # drivable_fallback
    2: 0,   # sidewalk

    # Non-walkable (Class 1)
    4: 1,   # building
    5: 1,   # wall
    6: 1,   # fence
    9: 1,   # vegetation
    10: 1,  # sky
    25: 1,  # fallback_background

    # Dynamic Obstacle (Class 2)
    11: 2,  # person
    12: 2,  # rider
    13: 2,  # car
    14: 2,  # truck
    15: 2,  # bus
    16: 2,  # autorickshaw
    17: 2,  # motorcycle
    18: 2,  # bicycle
    19: 2,  # animal

    # Structural Obstacle (Class 3)
    3: 3,   # curb
    7: 3,   # pole
    8: 3,   # traffic sign
    20: 3,  # traffic light
    21: 3,  # billboard

    # Explicitly ignored classes
    -1: IGNORE_INDEX,  # unlabeled
    22: IGNORE_INDEX,  # bridge
    23: IGNORE_INDEX,  # tunnel
    24: IGNORE_INDEX,  # out of roi
    26: IGNORE_INDEX,  # ego vehicle
    255: IGNORE_INDEX, # void
}


def _build_lut(mapping: Dict[int, int], max_id: int = 256) -> np.ndarray:
    """Pre-build a 256-element lookup table (LUT) for fast array indexing."""
    lut = np.full(max_id, IGNORE_INDEX, dtype=np.uint8)
    for k, v in mapping.items():
        if 0 <= k < max_id:
            lut[k] = v
    return lut


_CITYSCAPES_LUT = _build_lut(CITYSCAPES_TO_PROJECT)
_IDD_LUT = _build_lut(IDD_TO_PROJECT)


def remap_mask(mask: np.ndarray, dataset: str = "cityscapes") -> np.ndarray:
    """Remap dataset-specific class IDs in a mask to the 4 project classes.

    Args:
        mask (np.ndarray): 2D integer numpy array with original label IDs.
        dataset (str): Dataset source name, 'cityscapes' or 'idd'.

    Returns:
        np.ndarray: 2D uint8 numpy array with remapped labels in {0, 1, 2, 3, 255}.

    Raises:
        ValueError: If dataset is not supported.
    """
    dataset_key = dataset.lower().strip()
    if dataset_key == "cityscapes":
        lut = _CITYSCAPES_LUT
    elif dataset_key in ("idd", "idd_level3", "idd_level4"):
        lut = _IDD_LUT
    elif dataset_key in ("campus", "project", "passthrough"):
        # Custom campus dataset may already use 0-3 labels
        # Verify valid values or clamp unknown to 255
        remapped = mask.astype(np.uint8, copy=True)
        valid_mask = np.isin(remapped, [0, 1, 2, 3, IGNORE_INDEX])
        remapped[~valid_mask] = IGNORE_INDEX
        return remapped
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Supported: 'cityscapes', 'idd', 'campus'."
        )

    # Handle negative values or values exceeding 255
    safe_mask = np.where((mask < 0) | (mask > 255), IGNORE_INDEX, mask).astype(np.uint8)

    # Fast O(1) vectorized indexing via pre-built LUT
    remapped_mask = lut[safe_mask]
    return remapped_mask


def colorize_mask(
    mask: np.ndarray,
    palette: Dict[int, Tuple[int, int, int]] = PROJECT_PALETTE,
) -> np.ndarray:
    """Convert a single-channel label mask to an RGB visualization image.

    Args:
        mask (np.ndarray): 2D numpy array of class IDs (H, W).
        palette (dict): Dictionary mapping class ID to RGB tuple.

    Returns:
        np.ndarray: 3D uint8 array of shape (H, W, 3).
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in palette.items():
        match_pixels = (mask == class_id)
        if np.any(match_pixels):
            color_mask[match_pixels] = color

    return color_mask
