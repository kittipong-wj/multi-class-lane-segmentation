"""
model.py - Custom YOLO-seg Architecture for Single-Class Lane Segmentation
-------------------------------------------------------------------------
Input Size : (Batch_Size, 3, 36, 64) -> Width = 64, Height = 36, RGB
Output Size: (Batch_Size, 1, 36, 64) -> Single-class Binary Mask Logits

Architecture Design:
1. Backbone: Custom Conv-BN-SiLU blocks with strided convolutions for multi-scale feature extraction.
2. Neck    : FPN/PANet-style feature fusion combining low-level high-resolution features and high-level semantic features.
3. SegHead : Proto/Mask segmentation branch upsampling back to (36, 64) resolution for pixel-wise mask prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic Conv2d + BatchNorm2d + SiLU block (YOLO style)."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """YOLO Residual Bottleneck Block."""
    def __init__(self, channels):
        super().__init__()
        hidden_channels = channels // 2
        self.cv1 = ConvBlock(channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.cv2 = ConvBlock(hidden_channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class CustomYOLOSeg(nn.Module):
    """
    Custom YOLO-seg network designed from scratch for 64x36 input images.
    Predicts a 64x36 binary lane segmentation mask.
    """
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # =====================================================================
        # 1. BACKBONE (Feature Extraction)
        # =====================================================================
        # P1: (B, 3, 36, 64) -> (B, 32, 36, 64)
        self.stem = ConvBlock(in_channels, 32, kernel_size=3, stride=1, padding=1)

        # P2: (B, 32, 36, 64) -> (B, 64, 18, 32)
        self.stage1_down = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)
        self.stage1_res = ResidualBlock(64)

        # P3: (B, 64, 18, 32) -> (B, 128, 9, 16)
        self.stage2_down = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.stage2_res = ResidualBlock(128)

        # P4: (B, 128, 9, 16) -> (B, 256, 5, 8)  (Note: H=9 -> 5 with ceil/padding)
        self.stage3_down = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)
        self.stage3_res = ResidualBlock(256)

        # =====================================================================
        # 2. NECK (FPN / PANet Feature Fusion)
        # =====================================================================
        # Lateral 1x1 convs
        self.lat_p4 = ConvBlock(256, 128, kernel_size=1, stride=1, padding=0)
        self.lat_p3 = ConvBlock(128, 64, kernel_size=1, stride=1, padding=0)

        # Smooth Convs after feature fusion (concatenation doubles channels)
        self.smooth_p3 = ConvBlock(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth_p2 = ConvBlock(128, 64, kernel_size=3, stride=1, padding=1)

        # =====================================================================
        # 3. SEGMENTATION HEAD (Proto / Mask Branch)
        # =====================================================================
        # Upsampling Stage 1: (B, 128, 9, 16) -> (B, 64, 18, 32)
        self.up1 = nn.Sequential(
            nn.Upsample(size=(18, 32), mode='bilinear', align_corners=False),
            ConvBlock(128, 64, kernel_size=3, stride=1, padding=1)
        )

        # Upsampling Stage 2: (B, 64, 18, 32) -> (B, 32, 36, 64)
        self.up2 = nn.Sequential(
            nn.Upsample(size=(36, 64), mode='bilinear', align_corners=False),
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1)
        )

        # Final Mask Head Projection: (B, 32, 36, 64) -> (B, 1, 36, 64)
        self.mask_head = nn.Sequential(
            ConvBlock(32, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        """
        Forward pass.
        Input x : Tensor of shape (B, 3, 36, 64)
        Returns : Mask logits of shape (B, 1, 36, 64)
        """
        # Backbone Forward
        c1 = self.stem(x)                      # (B, 32, 36, 64)
        p2 = self.stage1_res(self.stage1_down(c1))  # (B, 64, 18, 32)
        p3 = self.stage2_res(self.stage2_down(p2))  # (B, 128, 9, 16)
        p4 = self.stage3_res(self.stage3_down(p3))  # (B, 256, 5, 8)

        # FPN Top-Down Path
        # P4 -> P3
        p4_up = F.interpolate(self.lat_p4(p4), size=p3.shape[2:], mode='bilinear', align_corners=False)
        p3_fused = self.smooth_p3(torch.cat([p3, p4_up], dim=1))  # (B, 128, 9, 16)

        # P3 -> P2
        p3_up = F.interpolate(self.lat_p3(p3_fused), size=p2.shape[2:], mode='bilinear', align_corners=False)
        p2_fused = self.smooth_p2(torch.cat([p2, p3_up], dim=1))  # (B, 64, 18, 32)

        # Mask Decoder / Segmentation Proto Head
        feat_up1 = self.up1(p3_fused)  # (B, 64, 18, 32)
        feat_combined = feat_up1 + p2_fused  # Skip connection fusion
        feat_up2 = self.up2(feat_combined)  # (B, 32, 36, 64)

        # Final Logits
        logits = self.mask_head(feat_up2)  # (B, 1, 36, 64)
        return logits


if __name__ == "__main__":
    # Test model shape consistency
    dummy_input = torch.randn(2, 3, 36, 64)
    model = CustomYOLOSeg(in_channels=3, num_classes=1)
    output = model(dummy_input)
    print(f"Model Input Shape : {dummy_input.shape}")
    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, 1, 36, 64), f"Unexpected shape {output.shape}"
    print("Sanity test passed successfully!")
