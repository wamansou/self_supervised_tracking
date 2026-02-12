"""
U-Net for Self-Supervised Particle Displacement Estimation

Architecture:
    Encoder  — 4 downsampling levels (Conv-BN-LeakyReLU × 2, MaxPool)
    Bottleneck — deepest feature map
    Decoder  — 4 upsampling levels with skip connections
    Head     — 1×1 conv → 2-channel displacement field (u, v) in pixels

Input:  [B, 2, H, W]  concatenated particle image pair
Output: [B, 2, H, W]  pixel displacement field
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Double conv: Conv3→BN→LeakyReLU → Conv3→BN→LeakyReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FlowEstimatorUNet(nn.Module):
    """
    Lightweight U-Net for dense displacement estimation.

    Args:
        in_channels:   number of input channels (default 2 for an image pair)
        base_features: feature count at level 1 (doubles each level)
    """

    def __init__(self, in_channels=2, base_features=32):
        super().__init__()
        f = base_features

        # ---------- Encoder ----------
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f, f * 2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.enc4 = ConvBlock(f * 4, f * 8)

        self.bottleneck = ConvBlock(f * 8, f * 16)

        # ---------- Decoder ----------
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4 = ConvBlock(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = ConvBlock(f * 2, f)

        # ---------- Output head ----------
        self.head = nn.Conv2d(f, 2, 1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                          # [B, f,   H,   W]
        e2 = self.enc2(self.pool(e1))               # [B, 2f,  H/2, W/2]
        e3 = self.enc3(self.pool(e2))               # [B, 4f,  H/4, W/4]
        e4 = self.enc4(self.pool(e3))               # [B, 8f,  H/8, W/8]

        b = self.bottleneck(self.pool(e4))           # [B, 16f, H/16, W/16]

        # Decoder + skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))

        return self.head(d1)                         # [B, 2, H, W]
