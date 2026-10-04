"""Evaluation metrics for 4-class semantic segmentation.

This module provides standard and safety-critical metrics for autonomous navigation:
    - Confusion Matrix accumulation
    - Per-class Intersection over Union (IoU) and mean IoU (mIoU)
    - Per-class Dice coefficient (F1 score)
    - Obstacle Recall (Dynamic Obstacles, Structural Obstacles, Combined)
    - Missed Obstacle Rate in the critical walking zone (bottom field of view)
    - Boundary F1 score (contour precision and boundary alignment)
"""

from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np
import torch

from utils.label_mapping import PROJECT_CLASSES, IGNORE_INDEX


class SegmentationMetrics:
    """Accumulates and computes semantic segmentation evaluation metrics.

    Maintains a confusion matrix across batches and computes per-class IoU,
    mIoU, Dice, obstacle recall, walking-zone missed obstacle rate, and boundary F1.
    """

    def __init__(
        self,
        num_classes: int = 4,
        ignore_index: int = IGNORE_INDEX,
        class_names: Optional[Dict[int, str]] = None,
    ) -> None:
        """Initialize SegmentationMetrics.

        Args:
            num_classes (int): Total number of semantic classes (default: 4).
            ignore_index (int): Label ID to ignore during evaluation (default: 255).
            class_names (dict, optional): Map from class ID to string label.
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or PROJECT_CLASSES

        # Primary confusion matrix: [ground_truth, prediction]
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

        # Walking zone safety metrics counters
        self.walking_zone_obstacle_total: int = 0
        self.walking_zone_obstacle_missed: int = 0

        # Boundary F1 counters
        self.boundary_tp: float = 0.0
        self.boundary_fp: float = 0.0
        self.boundary_fn: float = 0.0

    def reset(self) -> None:
        """Reset all accumulated statistics to zero."""
        self.confusion_matrix.fill(0)
        self.walking_zone_obstacle_total = 0
        self.walking_zone_obstacle_missed = 0
        self.boundary_tp = 0.0
        self.boundary_fp = 0.0
        self.boundary_fn = 0.0

    def _to_numpy_2d(self, tensor_or_array: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert input tensor or ndarray to a 2D/3D integer numpy array."""
        if isinstance(tensor_or_array, torch.Tensor):
            arr = tensor_or_array.detach().cpu().numpy()
        else:
            arr = np.asarray(tensor_or_array)

        # If probability/logits of shape (B, C, H, W) or (C, H, W)
        if arr.ndim == 4:
            arr = np.argmax(arr, axis=1)
        elif arr.ndim == 3 and arr.shape[0] == self.num_classes and arr.shape[0] != arr.shape[1]:
            arr = np.argmax(arr, axis=0)

        return arr.astype(np.int64)

    def update(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
        compute_boundary: bool = False,
        walking_zone_ratio: float = 0.4,
    ) -> None:
        """Update metrics with a batch of predictions and targets.

        Args:
            pred: Predicted labels (B, H, W) or logits/probabilities (B, C, H, W).
            target: Ground truth labels (B, H, W).
            compute_boundary (bool): Whether to calculate boundary metrics for this batch.
            walking_zone_ratio (float): Fraction of image height from bottom representing
                the immediate walking zone (default: 0.4).
        """
        pred_np = self._to_numpy_2d(pred)
        target_np = self._to_numpy_2d(target)

        if pred_np.shape != target_np.shape:
            raise ValueError(
                f"Shape mismatch: pred shape {pred_np.shape} vs target shape {target_np.shape}"
            )

        # Handle batch or single image
        if pred_np.ndim == 2:
            pred_np = pred_np[np.newaxis, ...]
            target_np = target_np[np.newaxis, ...]

        batch_size, h, w = pred_np.shape
        wz_start_y = int(h * (1.0 - walking_zone_ratio))

        for b in range(batch_size):
            p = pred_np[b]
            t = target_np[b]

            # 1. Update Confusion Matrix (ignoring pixels marked with ignore_index)
            valid = (t != self.ignore_index) & (t >= 0) & (t < self.num_classes)
            valid = valid & (p >= 0) & (p < self.num_classes)

            t_valid = t[valid]
            p_valid = p[valid]

            indices = self.num_classes * t_valid + p_valid
            counts = np.bincount(indices, minlength=self.num_classes**2)
            self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

            # 2. Update Walking Zone Missed Obstacle Rate
            # Obstacles: Class 2 (Dynamic) and Class 3 (Structural)
            t_wz = t[wz_start_y:, :]
            p_wz = p[wz_start_y:, :]
            wz_valid = (t_wz != self.ignore_index)

            t_obstacles = wz_valid & ((t_wz == 2) | (t_wz == 3))
            num_gt_obstacles = np.count_nonzero(t_obstacles)
            self.walking_zone_obstacle_total += num_gt_obstacles

            if num_gt_obstacles > 0:
                # Missed: GT is obstacle, but prediction is NOT an obstacle (e.g. predicted Walkable)
                missed = t_obstacles & (p_wz != 2) & (p_wz != 3)
                self.walking_zone_obstacle_missed += np.count_nonzero(missed)

            # 3. Optional Boundary F1 update per image
            if compute_boundary:
                self._update_boundary_batch(p, t, threshold_px=2)

    def _extract_boundaries(self, mask: np.ndarray) -> np.ndarray:
        """Extract 1-pixel boundary mask from semantic label array using morphological erosion."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # Mask boundaries occur where eroded label does not match original label
        eroded = cv2.erode(mask.astype(np.uint8), kernel, borderType=cv2.BORDER_REFLECT)
        boundary = (mask != eroded).astype(np.uint8)
        return boundary

    def _update_boundary_batch(
        self,
        pred_2d: np.ndarray,
        target_2d: np.ndarray,
        threshold_px: int = 2,
    ) -> None:
        """Accumulate boundary precision and recall within pixel distance threshold."""
        valid = (target_2d != self.ignore_index)
        if not np.any(valid):
            return

        pred_bound = self._extract_boundaries(pred_2d) * valid
        gt_bound = self._extract_boundaries(target_2d) * valid

        if not np.any(gt_bound) and not np.any(pred_bound):
            return

        # Distance transform to find pixels within threshold_px
        if np.any(gt_bound):
            gt_dist = cv2.distanceTransform(1 - gt_bound, cv2.DIST_L2, 3)
            # Precision: Predicted boundary pixels within threshold of GT boundary
            precision_match = (pred_bound > 0) & (gt_dist <= threshold_px)
            tp_prec = np.sum(precision_match)
        else:
            tp_prec = 0

        if np.any(pred_bound):
            pred_dist = cv2.distanceTransform(1 - pred_bound, cv2.DIST_L2, 3)
            # Recall: GT boundary pixels within threshold of predicted boundary
            recall_match = (gt_bound > 0) & (pred_dist <= threshold_px)
            tp_rec = np.sum(recall_match)
        else:
            tp_rec = 0

        self.boundary_tp += float(tp_prec + tp_rec) / 2.0
        self.boundary_fp += float(np.sum(pred_bound) - tp_prec)
        self.boundary_fn += float(np.sum(gt_bound) - tp_rec)

    def compute_iou(self) -> Dict[str, float]:
        """Compute per-class Intersection over Union (IoU).

        Returns:
            Dict[str, float]: Mapping from class name to IoU score in [0.0, 1.0].
        """
        ious: Dict[str, float] = {}
        for c in range(self.num_classes):
            tp = self.confusion_matrix[c, c]
            fp = np.sum(self.confusion_matrix[:, c]) - tp
            fn = np.sum(self.confusion_matrix[c, :]) - tp
            denominator = tp + fp + fn
            class_name = self.class_names.get(c, f"Class_{c}")
            ious[class_name] = float(tp / denominator) if denominator > 0 else 0.0
        return ious

    def compute_miou(self) -> float:
        """Compute mean Intersection over Union (mIoU) across all valid classes.

        Returns:
            float: Mean IoU score in [0.0, 1.0].
        """
        ious = self.compute_iou()
        if not ious:
            return 0.0
        return float(np.mean(list(ious.values())))

    def compute_dice(self) -> Dict[str, float]:
        """Compute per-class Dice coefficient (F1 score).

        Formula: 2 * TP / (2 * TP + FP + FN)

        Returns:
            Dict[str, float]: Mapping from class name to Dice score.
        """
        dices: Dict[str, float] = {}
        for c in range(self.num_classes):
            tp = self.confusion_matrix[c, c]
            fp = np.sum(self.confusion_matrix[:, c]) - tp
            fn = np.sum(self.confusion_matrix[c, :]) - tp
            denominator = 2 * tp + fp + fn
            class_name = self.class_names.get(c, f"Class_{c}")
            dices[class_name] = float(2 * tp / denominator) if denominator > 0 else 0.0
        return dices

    def compute_obstacle_recall(self) -> Dict[str, float]:
        """Compute obstacle recall specifically for safety-critical obstacle classes.

        Evaluates:
            - Dynamic Obstacle Recall (Class 2): TP_dyn / (TP_dyn + FN_dyn)
            - Structural Obstacle Recall (Class 3): TP_struct / (TP_struct + FN_struct)
            - Combined Obstacle Recall: Total Obstacle TP / Total Obstacle GT

        Returns:
            Dict[str, float]: Dictionary containing individual and combined obstacle recall.
        """
        # Dynamic Obstacle (Class 2)
        tp_dyn = self.confusion_matrix[2, 2]
        fn_dyn = np.sum(self.confusion_matrix[2, :]) - tp_dyn
        denom_dyn = tp_dyn + fn_dyn
        recall_dyn = float(tp_dyn / denom_dyn) if denom_dyn > 0 else 0.0

        # Structural Obstacle (Class 3)
        tp_struct = self.confusion_matrix[3, 3]
        fn_struct = np.sum(self.confusion_matrix[3, :]) - tp_struct
        denom_struct = tp_struct + fn_struct
        recall_struct = float(tp_struct / denom_struct) if denom_struct > 0 else 0.0

        # Combined Obstacle Recall
        combined_tp = tp_dyn + tp_struct
        combined_gt = denom_dyn + denom_struct
        combined_recall = float(combined_tp / combined_gt) if combined_gt > 0 else 0.0

        return {
            "dynamic_obstacle_recall": recall_dyn,
            "structural_obstacle_recall": recall_struct,
            "combined_obstacle_recall": combined_recall,
        }

    def compute_missed_obstacle_rate(self, walking_zone_ratio: float = 0.4) -> float:
        """Compute the Missed Obstacle Rate in the critical walking zone.

        For mobile navigation, missed obstacles in the bottom field of view (walking zone)
        pose severe collision hazard.

        Formula:
            Missed Obstacle Rate = Missed_Obstacles_WZ / Total_Obstacles_WZ

        Args:
            walking_zone_ratio (float): Ratio used during accumulation (for docstring reference).

        Returns:
            float: Fraction of ground truth obstacle pixels missed in the walking zone.
        """
        if self.walking_zone_obstacle_total == 0:
            return 0.0
        return float(self.walking_zone_obstacle_missed / self.walking_zone_obstacle_total)

    def compute_boundary_f1(self, threshold_px: int = 2) -> float:
        """Compute Boundary F1 score measuring contour alignment.

        Formula:
            Boundary F1 = 2 * Precision * Recall / (Precision + Recall + eps)

        Args:
            threshold_px (int): Pixel radius tolerance (for docstring reference).

        Returns:
            float: Boundary F1 score in [0.0, 1.0].
        """
        # TODO: Implement multi-scale or class-specific boundary evaluation if required
        denom = 2 * self.boundary_tp + self.boundary_fp + self.boundary_fn
        if denom == 0:
            return 0.0
        return float(2 * self.boundary_tp / denom)

    def summary(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """Generate a complete dictionary summary of all evaluation metrics.

        Returns:
            dict: Comprehensive evaluation report with all metrics.
        """
        return {
            "mIoU": self.compute_miou(),
            "per_class_iou": self.compute_iou(),
            "per_class_dice": self.compute_dice(),
            "obstacle_recall": self.compute_obstacle_recall(),
            "missed_obstacle_rate_wz": self.compute_missed_obstacle_rate(),
            "boundary_f1": self.compute_boundary_f1(),
        }
