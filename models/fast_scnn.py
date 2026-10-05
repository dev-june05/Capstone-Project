import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, use_act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_act=True):
        super().__init__()
        self.dw = ConvBNReLU(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, use_act=use_act)
        self.pw = ConvBNReLU(in_channels, out_channels, 1, 1, 0, use_act=use_act)

    def forward(self, x):
        return self.pw(self.dw(x))

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion=6):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        hidden_channels = int(round(in_channels * expansion))
        
        layers = []
        if expansion != 1:
            layers.append(ConvBNReLU(in_channels, hidden_channels, 1, 1, 0))
        layers.extend([
            ConvBNReLU(hidden_channels, hidden_channels, 3, stride, 1, groups=hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, bin_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_size),
                ConvBNReLU(in_channels, out_channels, 1, 1, 0)
            ) for bin_size in bin_sizes
        ])
        self.bottleneck = ConvBNReLU(in_channels + len(bin_sizes) * out_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        h, w = x.shape[2:]
        out = [x]
        for stage in self.stages:
            out.append(F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=False))
        return self.bottleneck(torch.cat(out, dim=1))

class LearningToDownsample(nn.Module):
    def __init__(self, in_channels=3, dw_channels=(32, 48, 64)):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, dw_channels[0], 3, 2, 1)
        self.ds1 = DepthwiseSeparableConv(dw_channels[0], dw_channels[1], 3, 2, 1)
        self.ds2 = DepthwiseSeparableConv(dw_channels[1], dw_channels[2], 3, 2, 1)

    def forward(self, x):
        return self.ds2(self.ds1(self.conv(x)))

class GlobalFeatureExtractor(nn.Module):
    def __init__(self, in_channels=64, block_channels=(64, 96, 128), out_channels=128):
        super().__init__()
        self.bottleneck1 = self._make_layer(InvertedResidual, in_channels, block_channels[0], 3, 2, 6)
        self.bottleneck2 = self._make_layer(InvertedResidual, block_channels[0], block_channels[1], 3, 2, 6)
        self.bottleneck3 = self._make_layer(InvertedResidual, block_channels[1], block_channels[2], 3, 1, 6)
        self.ppm = PyramidPoolingModule(block_channels[2], out_channels)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride, expansion):
        layers = [block(in_channels, out_channels, stride, expansion)]
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, 1, expansion))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.ppm(self.bottleneck3(self.bottleneck2(self.bottleneck1(x))))

class FeatureFusionModule(nn.Module):
    def __init__(self, high_channels=64, low_channels=128, out_channels=128):
        super().__init__()
        self.dw_high = ConvBNReLU(high_channels, out_channels, 3, 1, 1, groups=high_channels, use_act=False)
        self.conv_low = ConvBNReLU(low_channels, out_channels, 1, 1, 0, use_act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_feat, low_feat):
        low_feat = F.interpolate(low_feat, size=high_feat.shape[2:], mode='bilinear', align_corners=False)
        return self.relu(self.dw_high(high_feat) + self.conv_low(low_feat))

class Classifier(nn.Module):
    def __init__(self, in_channels=128, num_classes=4):
        super().__init__()
        self.ds1 = DepthwiseSeparableConv(in_channels, in_channels, 3, 1, 1)
        self.ds2 = DepthwiseSeparableConv(in_channels, in_channels, 3, 1, 1)
        self.drop = nn.Dropout2d(0.1)
        self.conv = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x, target_size=None):
        x = self.conv(self.drop(self.ds2(self.ds1(x))))
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

class FastSCNN(nn.Module):
    INPUT_SIZE = (512, 1024)

    def __init__(self, num_classes=4, in_channels=3):
        super().__init__()
        self.learning_to_downsample = LearningToDownsample(in_channels)
        self.global_feature_extractor = GlobalFeatureExtractor()
        self.feature_fusion = FeatureFusionModule()
        self.classifier = Classifier(num_classes=num_classes)

    def forward(self, x):
        h, w = x.shape[2:]
        higher_res_features = self.learning_to_downsample(x)
        lower_res_features = self.global_feature_extractor(higher_res_features)
        fused_features = self.feature_fusion(higher_res_features, lower_res_features)
        out = self.classifier(fused_features, target_size=(h, w))
        return {'main': out}
