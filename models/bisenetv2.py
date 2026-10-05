import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, use_act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DetailBranch(nn.Module):
    def __init__(self, in_channels=3, detail_channels=(64, 64, 128)):
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBNReLU(in_channels, detail_channels[0], kernel_size=3, stride=2, padding=1),
            ConvBNReLU(detail_channels[0], detail_channels[0], kernel_size=3, stride=1, padding=1)
        )
        self.stage2 = nn.Sequential(
            ConvBNReLU(detail_channels[0], detail_channels[1], kernel_size=3, stride=2, padding=1),
            ConvBNReLU(detail_channels[1], detail_channels[1], kernel_size=3, stride=1, padding=1),
            ConvBNReLU(detail_channels[1], detail_channels[1], kernel_size=3, stride=1, padding=1)
        )
        self.stage3 = nn.Sequential(
            ConvBNReLU(detail_channels[1], detail_channels[2], kernel_size=3, stride=2, padding=1),
            ConvBNReLU(detail_channels[2], detail_channels[2], kernel_size=3, stride=1, padding=1),
            ConvBNReLU(detail_channels[2], detail_channels[2], kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.stage3(self.stage2(self.stage1(x)))

class StemBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.left = nn.Sequential(
            ConvBNReLU(out_channels, out_channels // 2, kernel_size=1, stride=1, padding=0),
            ConvBNReLU(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1)
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse = ConvBNReLU(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return self.fuse(torch.cat([self.left(x), self.right(x)], dim=1))

class GatherExpansionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion_factor=6):
        super().__init__()
        self.stride = stride
        mid_channels = in_channels * expansion_factor
        
        self.main_branch = nn.Sequential(
            ConvBNReLU(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels, use_act=False)
        )
        
        if stride == 2:
            self.main_branch.add_module('dw_conv2', ConvBNReLU(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, use_act=False))
            
        self.main_branch.add_module('pw_conv', ConvBNReLU(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, use_act=False))

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                ConvBNReLU(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, use_act=False),
                ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_act=False)
            )

    def forward(self, x):
        return F.relu(self.main_branch(x) + self.shortcut(x), inplace=True)

class ContextEmbeddingBlock(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv_gap = ConvBNReLU(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_last = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        pooled = self.conv_gap(self.gap(x))
        return self.conv_last(x + pooled)

class SemanticBranch(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.stage1_2 = StemBlock(in_channels, out_channels=16)
        self.stage3 = nn.Sequential(
            GatherExpansionBlock(16, 32, stride=2),
            GatherExpansionBlock(32, 32, stride=1)
        )
        self.stage4 = nn.Sequential(
            GatherExpansionBlock(32, 64, stride=2),
            GatherExpansionBlock(64, 64, stride=1)
        )
        self.stage5 = nn.Sequential(
            GatherExpansionBlock(64, 128, stride=2),
            GatherExpansionBlock(128, 128, stride=1),
            GatherExpansionBlock(128, 128, stride=1),
            GatherExpansionBlock(128, 128, stride=1)
        )
        self.ceblock = ContextEmbeddingBlock(128, 128)

    def forward(self, x):
        stem = self.stage1_2(x)
        aux1 = self.stage3(stem)
        aux2 = self.stage4(aux1)
        aux3 = self.stage5(aux2)
        aux4 = self.ceblock(aux3)
        return aux4, (aux1, aux2, aux3, aux4)

class AggregationLayer(nn.Module):
    def __init__(self, detail_channels=128, semantic_channels=128, out_channels=128):
        super().__init__()
        self.detail_main = nn.Sequential(
            ConvBNReLU(detail_channels, detail_channels, kernel_size=3, stride=1, padding=1, groups=detail_channels, use_act=False),
            ConvBNReLU(detail_channels, detail_channels, kernel_size=1, stride=1, padding=0, use_act=False)
        )
        self.detail_down = nn.Sequential(
            ConvBNReLU(detail_channels, detail_channels, kernel_size=3, stride=2, padding=1, use_act=False),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.semantic_main = nn.Sequential(
            ConvBNReLU(semantic_channels, semantic_channels, kernel_size=3, stride=1, padding=1, use_act=False),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Sigmoid()
        )
        self.semantic_up = nn.Sequential(
            ConvBNReLU(semantic_channels, semantic_channels, kernel_size=3, stride=1, padding=1, groups=semantic_channels, use_act=False),
            ConvBNReLU(semantic_channels, semantic_channels, kernel_size=1, stride=1, padding=0, use_act=False),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Sigmoid()
        )

        self.conv_out = ConvBNReLU(detail_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, detail, semantic):
        branch1 = self.detail_main(detail) * self.semantic_main(semantic)
        branch2 = self.detail_down(detail) * self.semantic_up(semantic)
        branch2 = F.interpolate(branch2, size=detail.shape[2:], mode='bilinear', align_corners=False)
        return self.conv_out(branch1 + branch2)

class SegmentHead(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes=4, scale_factor=None, dropout_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.conv = ConvBNReLU(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.drop = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.out = nn.Conv2d(mid_channels, num_classes, kernel_size=1)

    def forward(self, x, target_size=None):
        x = self.out(self.drop(self.conv(x)))
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

class BiSeNetV2(nn.Module):
    INPUT_SIZE = (512, 1024)

    def __init__(self, num_classes=4, in_channels=3, detail_channels=(64, 64, 128), semantic_channels=128):
        super().__init__()
        self.detail = DetailBranch(in_channels=in_channels, detail_channels=detail_channels)
        self.semantic = SemanticBranch(in_channels=in_channels)
        self.aggregation = AggregationLayer(detail_channels[-1], semantic_channels, 128)
        
        self.head = SegmentHead(in_channels=128, mid_channels=1024, num_classes=num_classes)
        self.aux1 = SegmentHead(in_channels=32, mid_channels=128, num_classes=num_classes)
        self.aux2 = SegmentHead(in_channels=64, mid_channels=128, num_classes=num_classes)
        self.aux3 = SegmentHead(in_channels=128, mid_channels=256, num_classes=num_classes)
        self.aux4 = SegmentHead(in_channels=128, mid_channels=256, num_classes=num_classes)

    def forward(self, x):
        h, w = x.shape[2:]
        feat_detail = self.detail(x)
        feat_semantic, (aux1_feat, aux2_feat, aux3_feat, aux4_feat) = self.semantic(x)
        feat_fused = self.aggregation(feat_detail, feat_semantic)
        
        main_logits = self.head(feat_fused, target_size=(h, w))
        if self.training:
            return {
                'main': main_logits,
                'aux1': self.aux1(aux1_feat, target_size=(h, w)),
                'aux2': self.aux2(aux2_feat, target_size=(h, w)),
                'aux3': self.aux3(aux3_feat, target_size=(h, w)),
                'aux4': self.aux4(aux4_feat, target_size=(h, w)),
            }
        return {'main': main_logits}
