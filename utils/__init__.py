"""Utility package for 4-class semantic segmentation.

Modules:
    label_mapping: Dataset class mapping (Cityscapes/IDD to 4 project classes).
    dataset: Dataset loaders for Cityscapes, IDD, Campus, and Combined.
    augmentations: Training and validation augmentation pipelines (Albumentations).
    metrics: Semantic segmentation evaluation metrics (mIoU, Boundary F1, Obstacle Recall).
    class_weights: Loss balancing class weight computation.
"""
