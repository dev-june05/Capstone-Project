"""Fast-SCNN (Fast Semantic Segmentation Network) Architecture Skeleton.

Reference:
    Poudel et al., "Fast-SCNN: Fast Semantic Segmentation Network",
    British Machine Vision Conference (BMVC) 2019.
    https://arxiv.org/abs/1902.04502

Architecture Overview:
    Fast-SCNN is an ultra-lightweight real-time semantic segmentation architecture
    optimized for embedded hardware (e.g., NVIDIA Jetson Orin Nano). It uses an
    input-resolution-agnostic design with four core modules:
        1. Learning-to-Downsample (LDS): A fast 3-stage module that rapidly reduces
           the input resolution by 8x (down to 1/8 scale) using depthwise separable
           convolutions while extracting low-level edge and spatial features.
        2. Global Feature Extractor (GFE): Operates solely on the downsampled (1/8 scale)
           features to capture large receptive fields and global context using
           MobileNetV2-style inverted residual bottleneck blocks and a Pyramid
           Pooling Module (PPM) at 1/32 scale.
        3. Feature Fusion Module (FFM): Efficiently merges the high-resolution low-level
           features from LDS (1/8 scale) with the deep context features from GFE (upsampled
           to 1/8 scale) via depthwise separable convolutions and elementwise addition.
        4. Classifier: Refines the fused representations with depthwise separable
           convolutions, applies dropout, projects to target class logits (4 classes),
           and bilinearly upsamples to the original input resolution (512, 1024).

Target Application:
    4-Class Real-Time Urban / Campus Semantic Segmentation:
        - Class 0: Walkable (sidewalks, roads, paved paths)
        - Class 1: Non-walkable (buildings, vegetation, walls, terrain, sky)
        - Class 2: Dynamic Obstacle (pedestrians, cyclists, vehicles, animals)
        - Class 3: Structural Obstacle (poles, fences, bollards, curbs, signs)
    Input Resolution:  (B, 3, 512, 1024) [B, C, H, W]
    Output Resolution: (B, 4, 512, 1024) logits
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Building Blocks: ConvBNReLU & Depthwise Separable Convolution
# ==============================================================================

class ConvBNReLU(nn.Module):
    """Standard Convolution + BatchNorm2d + ReLU block.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of convolution kernel (default: 3).
        stride: Stride of convolution (default: 1).
        padding: Padding added to both sides (default: 1).
        dilation: Spacing between kernel elements (default: 1).
        groups: Number of blocked connections (default: 1).
        bias: If True, adds learnable bias (default: False, as BN follows).
        use_act: If True, applies ReLU activation; otherwise linear (default: True).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        use_act: bool = True,
    ) -> None:
        super().__init__()
        # TODO: Implement ConvBNReLU sequence:
        #   - nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)
        #   - nn.BatchNorm2d(out_channels)
        #   - nn.ReLU(inplace=True) if use_act else nn.Identity()
        raise NotImplementedError("ConvBNReLU.__init__ is a stub.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ConvBNReLU.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Output tensor of shape (B, out_channels, H', W').
        """
        # TODO: Forward pass through conv -> bn -> relu
        raise NotImplementedError("ConvBNReLU.forward is a stub.")


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution block.

    Factorizes standard 2D convolution into:
        1. Depthwise Convolution: Spatial filtering per input channel (groups=in_channels).
        2. Pointwise Convolution: Linear combination of channels across depth (1x1 kernel).

    This reduces computational FLOPs by ~8-9x compared to standard 3x3 convs,
    vital for achieving >= 60 FPS on Jetson Orin Nano.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of depthwise kernel (default: 3).
        stride: Stride of depthwise convolution (default: 1).
        padding: Padding for depthwise convolution (default: 1).
        dilation: Dilation for depthwise convolution (default: 1).
        bias: Whether to include bias (default: False).
        use_act: Whether to apply ReLU after pointwise conv (default: True).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = False,
        use_act: bool = True,
    ) -> None:
        super().__init__()
        # TODO: Implement Depthwise Separable Convolution:
        #   - Depthwise: ConvBNReLU(in_channels, in_channels, kernel_size, stride, padding,
        #                           dilation, groups=in_channels, bias=bias, use_act=True)
        #   - Pointwise: ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
        #                           groups=1, bias=bias, use_act=use_act)
        raise NotImplementedError("DepthwiseSeparableConv.__init__ is a stub.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Depthwise Separable Convolution.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Output tensor of shape (B, out_channels, H/stride, W/stride).
        """
        # TODO: Forward through depthwise -> pointwise
        raise NotImplementedError("DepthwiseSeparableConv.forward is a stub.")


# ==============================================================================
# Inverted Residual Bottleneck Block (MobileNetV2 style)
# ==============================================================================

class InvertedResidual(nn.Module):
    """Inverted Residual Bottleneck Block (MobileNetV2 style).

    Sequence:
        1. Expansion: 1x1 Conv + BN + ReLU expands channels by expansion factor (e.g., 6).
        2. Depthwise Conv: 3x3 Depthwise Conv + BN + ReLU captures spatial features.
        3. Pointwise Projection: 1x1 Conv + BN (linear bottleneck without ReLU) projects back.
        4. Shortcut: Residual addition if stride == 1 and in_channels == out_channels.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for depthwise convolution (1 or 2).
        expansion_factor: Expansion factor multiplier t (default: 6).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion_factor: int = 6,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        # TODO: Implement Inverted Residual Block:
        #   expanded_channels = in_channels * expansion_factor
        #   - Expansion: ConvBNReLU(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0)
        #   - Depthwise: ConvBNReLU(expanded_channels, expanded_channels, kernel_size=3,
        #                           stride=stride, padding=1, groups=expanded_channels)
        #   - Pointwise Linear: ConvBNReLU(expanded_channels, out_channels, kernel_size=1,
        #                                  stride=1, padding=0, use_act=False)
        raise NotImplementedError("InvertedResidual.__init__ is a stub.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Inverted Residual Block.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Output tensor of shape (B, out_channels, H/stride, W/stride).
        """
        # TODO: If self.use_residual: return x + self.conv(x); else: return self.conv(x)
        raise NotImplementedError("InvertedResidual.forward is a stub.")


# ==============================================================================
# Pyramid Pooling Module (PPM)
# ==============================================================================

class PyramidPoolingModule(nn.Module):
    """Pyramid Pooling Module (PPM) for multi-scale context aggregation.

    Pools the feature map at multiple bin scales (e.g., 1x1, 2x2, 3x3, 6x6),
    reduces channels with 1x1 convolutions, upsamples back to the input spatial
    dimensions, and concatenates with the original features.

    Args:
        in_channels: Number of input feature channels (default: 128).
        out_channels: Number of output feature channels after projection (default: 128).
        bins: Bin sizes for multi-scale pooling (default: (1, 2, 3, 6)).
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        bins: Tuple[int, ...] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        self.bins = bins
        # TODO: Implement Pyramid Pooling Module:
        #   reduction_channels = in_channels // len(bins)  # e.g., 128 // 4 = 32
        #   self.stages = nn.ModuleList([
        #       nn.Sequential(
        #           nn.AdaptiveAvgPool2d(bin_size),
        #           ConvBNReLU(in_channels, reduction_channels, kernel_size=1, stride=1, padding=0)
        #       ) for bin_size in bins
        #   ])
        #   self.project = ConvBNReLU(in_channels + len(bins) * reduction_channels, out_channels,
        #                             kernel_size=1, stride=1, padding=0)
        raise NotImplementedError("PyramidPoolingModule.__init__ is a stub.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Pyramid Pooling Module.

        Args:
            x: Feature map tensor of shape (B, in_channels, H, W).

        Returns:
            Context-aggregated feature map of shape (B, out_channels, H, W).
        """
        # TODO: Pool at each bin size, bilinearly interpolate to (H, W), concat with x, project
        raise NotImplementedError("PyramidPoolingModule.forward is a stub.")


# ==============================================================================
# 1. Learning-to-Downsample (LDS) Module
# ==============================================================================

class LearningToDownsample(nn.Module):
    """Learning-to-Downsample (LDS) Module.

    Rapidly downsamples high-resolution input images from (512, 1024) to (64, 128)
    (1/8 scale) using 3 sequential stages:
        - Stage 1: Conv2d (3 -> 32, k=3, s=2, p=1) + BN + ReLU -> (B, 32, 256, 512)
        - Stage 2: Depthwise Separable Conv (32 -> 48, k=3, s=2, p=1) -> (B, 48, 128, 256)
        - Stage 3: Depthwise Separable Conv (48 -> 64, k=3, s=2, p=1) -> (B, 64, 64, 128)

    Expected Input:  (B, 3, 512, 1024)
    Expected Output: (B, 64, 64, 128) -> 1/8 spatial scale
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 64) -> None:
        super().__init__()
        # TODO: Implement LDS 3-stage downsampling sequence:
        #   - self.conv1 = ConvBNReLU(in_channels, 32, kernel_size=3, stride=2, padding=1)
        #   - self.dsconv2 = DepthwiseSeparableConv(32, 48, kernel_size=3, stride=2, padding=1)
        #   - self.dsconv3 = DepthwiseSeparableConv(48, out_channels, kernel_size=3, stride=2, padding=1)
        raise NotImplementedError("LearningToDownsample.__init__ is a stub.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Learning-to-Downsample module.

        Args:
            x: Input image tensor of shape (B, 3, 512, 1024).

        Returns:
            Low-level spatial feature map of shape (B, 64, 64, 128).
        """
        # TODO: out = self.conv1(x) -> self.dsconv2 -> self.dsconv3
        raise NotImplementedError("LearningToDownsample.forward is a stub.")


# ==============================================================================
# 2. Global Feature Extractor (GFE) Module
# ==============================================================================

class GlobalFeatureExtractor(nn.Module):
    """Global Feature Extractor (GFE) Module.

    Extracts high-level semantic representations on low-resolution features (1/8 scale)
    using stacked Inverted Residual Bottleneck blocks and a Pyramid Pooling Module (PPM):
        - Bottlenecks Group 1: 3 blocks (64 -> 64, stride 2, expansion 6) -> 1/16 scale (32, 64)
        - Bottlenecks Group 2: 3 blocks (64 -> 96, stride 2, expansion 6) -> 1/32 scale (16, 32)
        - Bottlenecks Group 3: 3 blocks (96 -> 128, stride 1, expansion 6) -> 1/32 scale (16, 32)
        - PPM: Multi-scale pooling at 1/32 scale -> 128 channels

    Expected Input:  (B, 64, 64, 128) [from LDS]
    Expected Output: (B, 128, 16, 32) [1/32 spatial scale]
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 128) -> None:
        super().__init__()
        # TODO: Implement GFE blocks:
        #   - Group 1: 3x InvertedResidual(in_channels=64, out_channels=64, stride=2 (1st block), exp=6)
        #   - Group 2: 3x InvertedResidual(in_channels=64, out_channels=96, stride=2 (1st block), exp=6)
        #   - Group 3: 3x InvertedResidual(in_channels=96, out_channels=128, stride=1, exp=6)
        #   - PPM: PyramidPoolingModule(in_channels=128, out_channels=out_channels, bins=(1, 2, 3, 6))
        raise NotImplementedError("GlobalFeatureExtractor.__init__ is a stub.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Global Feature Extractor.

        Args:
            x: Low-level feature tensor of shape (B, 64, 64, 128).

        Returns:
            High-level semantic context tensor of shape (B, 128, 16, 32).
        """
        # TODO: Forward through Group 1 -> Group 2 -> Group 3 -> PPM
        raise NotImplementedError("GlobalFeatureExtractor.forward is a stub.")


# ==============================================================================
# 3. Feature Fusion Module (FFM)
# ==============================================================================

class FeatureFusionModule(nn.Module):
    """Feature Fusion Module (FFM).

    Merges low-level spatial features from LDS (64 ch, 1/8 scale) with high-level
    semantic context features from GFE (128 ch, 1/32 scale).

    Data Flow:
        1. High-level GFE branch: Upsampled 4x via bilinear interpolation to match 1/8 scale
           -> Depthwise Convolution (3x3, groups=128) + BN -> Pointwise Conv (1x1, 128 -> 128).
        2. Low-level LDS branch: Pointwise Convolution (1x1, 64 -> 128) + BN.
        3. Fusion: Elementwise addition of both branches -> ReLU activation.

    Expected Input:
        - lds_features: (B, 64, 64, 128)
        - gfe_features: (B, 128, 16, 32)
    Expected Output:
        - fused:        (B, 128, 64, 128)
    """

    def __init__(self, lds_channels: int = 64, gfe_channels: int = 128, out_channels: int = 128) -> None:
        super().__init__()
        # TODO: Implement Feature Fusion branches:
        #   - High-level branch:
        #       - Depthwise Conv (3x3, groups=gfe_channels, stride=1, padding=1) + BN
        #       - Pointwise Conv (gfe_channels -> out_channels, kernel=1) + BN (use_act=False)
        #   - Low-level branch:
        #       - Pointwise Conv (lds_channels -> out_channels, kernel=1) + BN (use_act=False)
        #   - Post-addition activation: nn.ReLU(inplace=True)
        raise NotImplementedError("FeatureFusionModule.__init__ is a stub.")

    def forward(self, lds_feat: torch.Tensor, gfe_feat: torch.Tensor) -> torch.Tensor:
        """Forward pass fusing LDS and GFE representations.

        Args:
            lds_feat: Low-level spatial features of shape (B, 64, 64, 128).
            gfe_feat: High-level context features of shape (B, 128, 16, 32).

        Returns:
            Fused feature map of shape (B, 128, 64, 128).
        """
        # TODO: Upsample gfe_feat by 4x to match lds_feat spatial size:
        #   gfe_up = F.interpolate(gfe_feat, size=lds_feat.shape[2:], mode='bilinear', align_corners=False)
        #   return F.relu(self.lds_conv(lds_feat) + self.gfe_conv(gfe_up), inplace=True)
        raise NotImplementedError("FeatureFusionModule.forward is a stub.")


# ==============================================================================
# 4. Classifier
# ==============================================================================

class Classifier(nn.Module):
    """Classifier / Segmentation Head for Fast-SCNN.

    Refines the fused feature map and produces target class logits:
        - 2x Depthwise Separable Convolutions (128 -> 128, k=3)
        - Spatial Dropout (p=0.1) for regularization
        - 1x1 Convolution projecting to num_classes (4 classes)
        - Bilinear upsampling to original input resolution (512, 1024)

    Args:
        in_channels: Fused input channels (default: 128).
        num_classes: Number of output semantic classes (default: 4).
        dropout_rate: Dropout probability (default: 0.1).
    """

    def __init__(self, in_channels: int = 128, num_classes: int = 4, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.num_classes = num_classes
        # TODO: Implement Classifier layers:
        #   - self.dsconv1 = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=3, padding=1)
        #   - self.dsconv2 = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=3, padding=1)
        #   - self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        #   - self.conv_out = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=True)
        raise NotImplementedError("Classifier.__init__ is a stub.")

    def forward(self, x: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Forward pass through Classifier.

        Args:
            x: Fused feature tensor of shape (B, 128, 64, 128).
            target_size: Target (H, W) to upsample to (default: (512, 1024)).

        Returns:
            Class logits tensor of shape (B, num_classes, 512, 1024).
        """
        # TODO: Forward through dsconv1 -> dsconv2 -> dropout -> conv_out
        #       -> F.interpolate(..., size=target_size, mode='bilinear', align_corners=False)
        raise NotImplementedError("Classifier.forward is a stub.")


# ==============================================================================
# Fast-SCNN Model
# ==============================================================================

class FastSCNN(nn.Module):
    """Fast-SCNN Semantic Segmentation Network for 4-Class Walkability Prediction.

    Full architecture assembling:
        1. LearningToDownsample (LDS): Fast 1/8 downsampling module.
        2. GlobalFeatureExtractor (GFE): Inverted bottlenecks + PPM at 1/32 scale.
        3. FeatureFusionModule (FFM): Fuses LDS spatial features with GFE context.
        4. Classifier: Depthwise separable convolutions + upsampling to original size.

    Input Shape:
        (B, 3, 512, 1024) [B, C, H, W] - RGB images normalized to ImageNet statistics.

    Output Shape:
        Returns a dictionary containing the primary segmentation logits:
        {
            'main': (B, 4, 512, 1024),  # Class logits matching input resolution
        }

    Deployment Note:
        Fast-SCNN has < 1.2M parameters and requires ~1.5 GFLOPs at 512x1024.
        All operations (depthwise separable convs, bilinear upsample with align_corners=False)
        map directly to TensorRT FP16 / INT8 kernels on Jetson Orin Nano, enabling
        ultra-low-latency deployment (> 60 FPS).
    """

    INPUT_SIZE: Tuple[int, int] = (512, 1024)

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 3,
        lds_channels: int = 64,
        gfe_channels: int = 128,
        dropout_rate: float = 0.1,
    ) -> None:
        """Initialize FastSCNN model architecture.

        Args:
            num_classes: Number of semantic output classes (default: 4).
            in_channels: Number of input image channels (default: 3 for RGB).
            lds_channels: Output channels of Learning-to-Downsample module (default: 64).
            gfe_channels: Output channels of Global Feature Extractor (default: 128).
            dropout_rate: Dropout rate for Classifier (default: 0.1).
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # TODO: Initialize all 4 core submodules:
        #   1. Learning-to-Downsample module:
        #      self.learning_to_downsample = LearningToDownsample(in_channels=in_channels, out_channels=lds_channels)
        #   2. Global Feature Extractor:
        #      self.global_feature_extractor = GlobalFeatureExtractor(in_channels=lds_channels, out_channels=gfe_channels)
        #   3. Feature Fusion Module:
        #      self.feature_fusion = FeatureFusionModule(lds_channels=lds_channels, gfe_channels=gfe_channels, out_channels=gfe_channels)
        #   4. Classifier:
        #      self.classifier = Classifier(in_channels=gfe_channels, num_classes=num_classes, dropout_rate=dropout_rate)
        raise NotImplementedError("FastSCNN.__init__ is a stub. Initialize submodules.")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of FastSCNN.

        Processes input image tensor through LDS -> GFE -> FFM -> Classifier
        and produces class logits dictionary.

        Args:
            x: Input image tensor of shape (B, 3, 512, 1024).

        Returns:
            Dict[str, torch.Tensor]:
                {
                    'main': Tensor of shape (B, num_classes, 512, 1024),
                }
        """
        # TODO: Implement forward pass logic:
        #   h, w = x.shape[2:]  # Expected (512, 1024)
        #
        #   # 1. Learning-to-Downsample (1/8 scale)
        #   lds_feat = self.learning_to_downsample(x)  # (B, 64, 64, 128)
        #
        #   # 2. Global Feature Extractor (1/32 scale)
        #   gfe_feat = self.global_feature_extractor(lds_feat)  # (B, 128, 16, 32)
        #
        #   # 3. Feature Fusion Module (1/8 scale)
        #   fused_feat = self.feature_fusion(lds_feat, gfe_feat)  # (B, 128, 64, 128)
        #
        #   # 4. Classifier Head & Upsample to (512, 1024)
        #   output = self.classifier(fused_feat, target_size=(h, w))  # (B, num_classes, 512, 1024)
        #
        #   return {'main': output}
        raise NotImplementedError("FastSCNN.forward is a stub.")
