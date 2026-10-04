"""Model zoo and architecture factory for 4-class real-time semantic segmentation.

This module provides model definitions and a factory function to instantiate:
    - BiSeNetV2: Bilateral Network with Guided Aggregation (detail + semantic branches)
    - FastSCNN: Fast Semantic Segmentation Network (LDS + GFE + FFM + Classifier)

Both architectures are configured for:
    - Input shape: (B, 3, 512, 1024)
    - Output shape: (B, 4, 512, 1024)
    - Target classes:
        0: Walkable
        1: Non-walkable
        2: Dynamic Obstacle
        3: Structural Obstacle
    - Target hardware: NVIDIA Jetson Orin Nano (TensorRT optimized)
"""

from typing import Any, Dict, Type
import torch.nn as nn

from .bisenetv2 import BiSeNetV2
from .fast_scnn import FastSCNN

# Registry of supported model architectures
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "bisenetv2": BiSeNetV2,
    "bisenet_v2": BiSeNetV2,
    "bisenet": BiSeNetV2,
    "fastscnn": FastSCNN,
    "fast_scnn": FastSCNN,
    "fast-scnn": FastSCNN,
}


def get_model(name: str, num_classes: int = 4, **kwargs: Any) -> nn.Module:
    """Instantiate a segmentation model by name.

    Args:
        name: Name of the model architecture (case-insensitive).
              Supported options: 'bisenetv2', 'bisenet_v2', 'fastscnn', 'fast_scnn'.
        num_classes: Number of semantic output classes (default: 4).
        **kwargs: Additional keyword arguments passed to the model constructor
                  (e.g., in_channels, dropout_rate, etc.).

    Returns:
        An instantiated PyTorch model (nn.Module).

    Raises:
        ValueError: If the requested model name is not recognized.

    Examples:
        >>> model = get_model('bisenetv2', num_classes=4)
        >>> model = get_model('fast_scnn', num_classes=4)
    """
    normalized_name = name.strip().lower()
    if normalized_name not in MODEL_REGISTRY:
        supported = sorted(list(set(MODEL_REGISTRY.keys())))
        raise ValueError(
            f"Unknown model architecture: '{name}'. "
            f"Available models: {supported}"
        )

    model_cls = MODEL_REGISTRY[normalized_name]
    return model_cls(num_classes=num_classes, **kwargs)


__all__ = [
    "BiSeNetV2",
    "FastSCNN",
    "get_model",
    "MODEL_REGISTRY",
]
