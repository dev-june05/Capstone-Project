"""Data augmentation pipelines for 4-class semantic segmentation.

Provides training and validation pipelines using Albumentations, optimized
for autonomous navigation and obstacle detection on Jetson Orin Nano.
Includes a stub for Copy-Paste augmentation tailored to tackle obstacle
class imbalance (Dynamic and Structural Obstacles).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    A = None  # type: ignore
    ToTensorV2 = None  # type: ignore
    ALBUMENTATIONS_AVAILABLE = False


# Standard ImageNet statistics for normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_augmentations(
    height: int = 512,
    width: int = 1024,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> "A.Compose":
    """Build the Albumentations training augmentation pipeline.

    Includes photometric and spatial transformations designed to simulate
    varying illumination, road textures, shadows, and optical blur encountered
    in Indian driving (IDD) and campus environments.

    Pipeline operations:
        1. RandomCrop: Crops to target training resolution (H, W).
        2. HorizontalFlip: Mirroring along the vertical axis.
        3. ColorJitter: Photometric jitter (brightness, contrast, saturation, hue).
        4. GaussianBlur: Optical blur simulation.
        5. RandomBrightnessContrast: Contrast and exposure variation.
        6. RandomShadow: Simulates tree, building, and vehicle shadows.
        7. Normalize: Scales channels using ImageNet mean and std.
        8. ToTensorV2: Converts (H, W, C) numpy arrays to PyTorch (C, H, W) tensors.

    Args:
        height (int): Target crop height (default: 512).
        width (int): Target crop width (default: 1024).
        mean (tuple): Mean values for RGB normalization.
        std (tuple): Std values for RGB normalization.

    Returns:
        albumentations.Compose: Composed augmentation transform.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError(
            "Albumentations is required for get_train_augmentations. "
            "Please install via: pip install albumentations"
        )

    return A.Compose(
        [
            # Spatial augmentations
            A.RandomCrop(height=height, width=width, p=1.0),
            A.HorizontalFlip(p=0.5),

            # Illumination and color augmentations
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=3,
                shadow_dimension=5,
                p=0.3,
            ),

            # Normalization and tensor conversion
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def get_val_augmentations(
    height: int = 512,
    width: int = 1024,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> "A.Compose":
    """Build the Albumentations validation / test augmentation pipeline.

    Applies deterministic resizing to model input dimensions followed by
    ImageNet normalization and tensor conversion.

    Args:
        height (int): Target evaluation height (default: 512).
        width (int): Target evaluation width (default: 1024).
        mean (tuple): Mean values for RGB normalization.
        std (tuple): Std values for RGB normalization.

    Returns:
        albumentations.Compose: Composed validation transform.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError(
            "Albumentations is required for get_val_augmentations. "
            "Please install via: pip install albumentations"
        )

    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


class CopyPasteObstacles:
    """Stub class for Copy-Paste augmentation tailored to semantic obstacles.

    Pastes segmented obstacle instances (Dynamic: Class 2, Structural: Class 3)
    from an object bank or secondary image onto target walkable areas
    (Class 0) to mitigate severe foreground-background class imbalance.
    """

    def __init__(
        self,
        obstacle_classes: Tuple[int, ...] = (2, 3),
        max_paste_objects: int = 4,
        scale_range: Tuple[float, float] = (0.5, 1.5),
        prob: float = 0.5,
    ) -> None:
        """Initialize CopyPasteObstacles parameters.

        Args:
            obstacle_classes (tuple): Target class IDs to copy (2=Dynamic, 3=Structural).
            max_paste_objects (int): Maximum instances to paste per image.
            scale_range (tuple): Random scaling range (min, max) for pasted obstacles.
            prob (float): Probability of applying copy-paste to an image.
        """
        self.obstacle_classes = obstacle_classes
        self.max_paste_objects = max_paste_objects
        self.scale_range = scale_range
        self.prob = prob

    def __call__(
        self,
        image: Any,
        mask: Any,
        source_image: Optional[Any] = None,
        source_mask: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Apply copy-paste augmentation.

        TODO: Implement full copy-paste pipeline:
            1. Extract binary masks of instances belonging to self.obstacle_classes
               from source_mask.
            2. Randomly select up to self.max_paste_objects instances.
            3. Apply random geometric transformations (scaling, flipping) to extracted patches.
            4. Identify valid insertion points (walkable road/sidewalk areas in target mask).
            5. Alpha-blend or overwrite RGB pixels in image and update corresponding mask labels.
            6. Return augmented {'image': image, 'mask': mask}.

        Raises:
            NotImplementedError: Currently a skeleton stub for project implementation.
        """
        raise NotImplementedError(
            "Copy-paste obstacle augmentation is currently a stub. "
            "Implement instance extraction and placement logic as detailed in TODO."
        )


def get_copy_paste_augmentation(
    obstacle_classes: Tuple[int, ...] = (2, 3),
    max_paste_objects: int = 4,
    prob: float = 0.5,
) -> CopyPasteObstacles:
    """Get the Copy-Paste augmentation callable stub for obstacle instances.

    Args:
        obstacle_classes (tuple): Class IDs to extract and paste (default: (2, 3)).
        max_paste_objects (int): Maximum obstacle patches to paste per image.
        prob (float): Probability of applying the transformation.

    Returns:
        CopyPasteObstacles: An instance of CopyPasteObstacles stub.
    """
    # TODO: Connect with offline obstacle crop bank if caching crops across dataset
    return CopyPasteObstacles(
        obstacle_classes=obstacle_classes,
        max_paste_objects=max_paste_objects,
        prob=prob,
    )
