"""Dataset loaders for 4-class semantic segmentation.

This module provides PyTorch Dataset implementations for:
    - CityscapesDataset: Cityscapes European driving dataset.
    - IDDDataset: Indian Driving Dataset (unstructured traffic).
    - CampusDataset: Custom campus mobility dataset.
    - CombinedDataset: Unified multi-dataset concatenation wrapper.

All datasets return a dictionary format:
    {
        'image': torch.FloatTensor of shape (C, H, W),
        'mask': torch.LongTensor of shape (H, W),
        'filename': str identifier of the sample
    }
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

from utils.label_mapping import remap_mask, IGNORE_INDEX


def _ensure_tensor(
    image: Union[np.ndarray, torch.Tensor],
    mask: Union[np.ndarray, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper to ensure image and mask are PyTorch tensors with correct dtype and shape."""
    # Ensure image is torch.FloatTensor of shape (C, H, W)
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[-1] in (1, 3):  # HWC
            image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float()
        if image.max() > 1.0 and image.dtype == torch.float32:
            # If not normalized by albumentations, scale to [0, 1]
            image = image / 255.0
    elif isinstance(image, torch.Tensor):
        if image.ndim == 3 and image.shape[-1] in (1, 3):
            image = image.permute(2, 0, 1)
        image = image.float()

    # Ensure mask is torch.LongTensor of shape (H, W)
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).long()
    elif isinstance(mask, torch.Tensor):
        mask = mask.long()

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)

    return image, mask


class CityscapesDataset(Dataset):
    """Cityscapes dataset loader with automatic 4-class remapping.

    Directory structure expected:
        root_dir/
            leftImg8bit/
                <split>/
                    <city>/
                        *_leftImg8bit.png
            gtFine/
                <split>/
                    <city>/
                        *_gtFine_labelIds.png
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize Cityscapes dataset.

        Args:
            root_dir (str or Path): Path to Cityscapes root directory.
            split (str): One of 'train', 'val', or 'test'.
            transform (callable, optional): Albumentations or torchvision transform.
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split.lower().strip()
        self.transform = transform
        self.images: List[Path] = []
        self.masks: List[Path] = []

        self._find_files()

    def _find_files(self) -> None:
        """Scan root directory and match image files with ground truth masks."""
        # TODO: Adjust sub-paths if Cityscapes directory structure differs
        img_dir = self.root_dir / "leftImg8bit" / self.split
        mask_dir = self.root_dir / "gtFine" / self.split

        if not img_dir.exists():
            # Fallback to direct search if leftImg8bit is not partitioned by split
            img_dir = self.root_dir / self.split / "images"
            mask_dir = self.root_dir / self.split / "masks"

        if not img_dir.exists():
            # If data directory is not yet populated, keep lists empty (allows skeleton instantiation)
            return

        for img_path in sorted(img_dir.rglob("*_leftImg8bit.png")):
            # Cityscapes naming: <city>_<seq>_<frame>_leftImg8bit.png
            # Mask naming:      <city>_<seq>_<frame>_gtFine_labelIds.png
            mask_name = img_path.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
            city = img_path.parent.name
            mask_path = mask_dir / city / mask_name

            if not mask_path.exists():
                # Direct check if city subdirectory is omitted
                mask_path = mask_dir / mask_name

            if mask_path.exists():
                self.images.append(img_path)
                self.masks.append(mask_path)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        if idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.images)}")

        img_path = self.images[idx]
        mask_path = self.masks[idx]

        # Read RGB image
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read ground truth mask (label IDs)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Failed to load mask: {mask_path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Remap original Cityscapes IDs to 4 project classes
        mask = remap_mask(mask, dataset="cityscapes")

        # Apply transformations (Albumentations expected: transform(image=..., mask=...))
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert to PyTorch tensors
        img_tensor, mask_tensor = _ensure_tensor(image, mask)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "filename": img_path.name,
        }


class IDDDataset(Dataset):
    """Indian Driving Dataset (IDD) loader with automatic 4-class remapping.

    Directory structure expected:
        root_dir/
            leftImg8bit/
                <split>/
                    <folder>/
                        *_leftImg8bit.png
            gtFine/
                <split>/
                    <folder>/
                        *_gtFine_labelids.png
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize IDD dataset.

        Args:
            root_dir (str or Path): Path to IDD root directory.
            split (str): One of 'train', 'val', or 'test'.
            transform (callable, optional): Albumentations or torchvision transform.
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split.lower().strip()
        self.transform = transform
        self.images: List[Path] = []
        self.masks: List[Path] = []

        self._find_files()

    def _find_files(self) -> None:
        """Scan root directory and match image files with ground truth masks."""
        # TODO: Adjust for IDD-specific directory structure (e.g. IDD Segmentation level 3/4)
        img_dir = self.root_dir / "leftImg8bit" / self.split
        mask_dir = self.root_dir / "gtFine" / self.split

        if not img_dir.exists():
            img_dir = self.root_dir / self.split / "images"
            mask_dir = self.root_dir / self.split / "masks"

        if not img_dir.exists():
            return

        for img_path in sorted(img_dir.rglob("*.png")):
            # IDD mask pattern typically matches image name with _gtFine_labelids.png or identical stem
            mask_name = img_path.name.replace("_leftImg8bit.png", "_gtFine_labelids.png")
            subfolder = img_path.parent.name
            mask_path = mask_dir / subfolder / mask_name

            if not mask_path.exists():
                mask_path = mask_dir / subfolder / img_path.name
            if not mask_path.exists():
                mask_path = mask_dir / mask_name

            if mask_path.exists():
                self.images.append(img_path)
                self.masks.append(mask_path)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        if idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.images)}")

        img_path = self.images[idx]
        mask_path = self.masks[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load IDD image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Failed to load IDD mask: {mask_path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Remap IDD labels to project classes
        mask = remap_mask(mask, dataset="idd")

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        img_tensor, mask_tensor = _ensure_tensor(image, mask)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "filename": img_path.name,
        }


class CampusDataset(Dataset):
    """Custom campus dataset loader for project-specific mobility deployment.

    Directory structure expected:
        root_dir/
            <split>/
                images/
                    *.jpg or *.png
                masks/
                    *.png
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize Campus dataset.

        Args:
            root_dir (str or Path): Path to campus dataset root directory.
            split (str): One of 'train', 'val', or 'test'.
            transform (callable, optional): Albumentations or torchvision transform.
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split.lower().strip()
        self.transform = transform
        self.images: List[Path] = []
        self.masks: List[Path] = []

        self._find_files()

    def _find_files(self) -> None:
        """Scan custom campus dataset directory for image-mask pairs."""
        # Try root_dir/<split>/images first, then root_dir/images/<split>
        img_dir = self.root_dir / self.split / "images"
        mask_dir = self.root_dir / self.split / "masks"

        if not img_dir.exists():
            img_dir = self.root_dir / "images" / self.split
            mask_dir = self.root_dir / "masks" / self.split

        if not img_dir.exists():
            return

        supported_exts = ("*.png", "*.jpg", "*.jpeg")
        for ext in supported_exts:
            for img_path in sorted(img_dir.glob(ext)):
                # Mask with matching stem (png format)
                mask_path = mask_dir / f"{img_path.stem}.png"
                if mask_path.exists():
                    self.images.append(img_path)
                    self.masks.append(mask_path)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        if idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.images)}")

        img_path = self.images[idx]
        mask_path = self.masks[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load campus image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Failed to load campus mask: {mask_path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Campus annotations are assumed to already follow 0-3 classes (with 255 ignore)
        mask = remap_mask(mask, dataset="campus")

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        img_tensor, mask_tensor = _ensure_tensor(image, mask)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "filename": img_path.name,
        }


class CombinedDataset(Dataset):
    """Combines multiple semantic segmentation datasets into a unified dataset."""

    def __init__(
        self,
        datasets: Sequence[Dataset],
    ) -> None:
        """Initialize CombinedDataset.

        Args:
            datasets (Sequence[Dataset]): Sequence of Dataset instances to combine.
        """
        super().__init__()
        self.datasets = list(datasets)
        self.cumulative_sizes = self._compute_cumulative_sizes()

    def _compute_cumulative_sizes(self) -> List[int]:
        """Compute cumulative lengths for index resolution."""
        cumsum: List[int] = []
        total = 0
        for d in self.datasets:
            total += len(d)
            cumsum.append(total)
        return cumsum

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for CombinedDataset of size {len(self)}")

        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cum_size in enumerate(self.cumulative_sizes):
            if idx < cum_size:
                dataset_idx = i
                break

        # Compute relative index within that dataset
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx][sample_idx]
