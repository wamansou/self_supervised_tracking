"""
RAFT-Style Flow Estimator for Self-Supervised Particle Tracking

Architecture:
    Feature Encoder   — shared CNN, extracts 1/8-resolution features per frame
    Context Encoder   — frame-1 CNN, produces GRU hidden state + context
    Correlation Volume — all-pairs dot product + 4-level pyramid
    Update Block      — ConvGRU iterative refinement (default 12 iters)
    Convex Upsampling — learned 8× upsampling for sharp flow edges

Input:  [B, 2, H, W]   concatenated particle image pair
Output: list of [B, 2, H, W]  flow predictions (one per refinement step)

Reference:
    Teed & Deng, "RAFT: Recurrent All-Pairs Field Transforms
    for Optical Flow", ECCV 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Residual block with optional stride-2 downsample."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.norm1 = nn.InstanceNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_ch)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.InstanceNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + (self.shortcut(x) if self.shortcut else x))


# ---------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------

class FeatureEncoder(nn.Module):
    """Shared per-frame feature extractor → 1/8 resolution."""

    def __init__(self, in_ch=1, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 7, stride=2, padding=3),      # /2
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(64, 64),
            ResidualBlock(64, 96, stride=2),                    # /4
            ResidualBlock(96, 96),
            ResidualBlock(96, out_dim, stride=2),               # /8
            ResidualBlock(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)  # [B, out_dim, H/8, W/8]


class ContextEncoder(nn.Module):
    """Frame-1 context → GRU initial hidden state + context features."""

    def __init__(self, in_ch=1, hidden_dim=128, context_dim=128):
        super().__init__()
        total = hidden_dim + context_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 7, stride=2, padding=3),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(64, 64),
            ResidualBlock(64, 96, stride=2),
            ResidualBlock(96, 96),
            ResidualBlock(96, total, stride=2),
            ResidualBlock(total, total),
        )
        self.hdim = hidden_dim

    def forward(self, x):
        out = self.net(x)
        return torch.tanh(out[:, :self.hdim]), torch.relu(out[:, self.hdim:])


# ---------------------------------------------------------------
# Correlation volume with multi-scale pyramid
# ---------------------------------------------------------------

class CorrBlock:
    """
    All-pairs correlation between fmap1 [B,D,H,W] and fmap2 [B,D,H,W].

    Stores a 4-level pyramid (avg-pool 2× each level) and provides a
    local-neighbourhood lookup given the current flow estimate.

    Lookup output: [B, levels × (2r+1)², H, W]
    """

    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        B, D, H, W = fmap1.shape
        self.B, self.H, self.W = B, H, W

        # All-pairs correlation → [B·H·W, 1, H, W]
        corr = torch.einsum("bchw,bcij->bhwij", fmap1, fmap2) / (D ** 0.5)
        corr = corr.reshape(B * H * W, 1, H, W)

        self.pyramid = [corr]
        for _ in range(num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.pyramid.append(corr)

    def __call__(self, flow):
        """
        flow: [B, 2, H, W]  current estimate in 1/8-res pixel coords.
              flow[:,0] = u (x),  flow[:,1] = v (y).
        """
        B, _, H, W = flow.shape
        r = self.radius

        # Local offset grid — same integer offsets at every pyramid level;
        # coarser levels automatically cover a wider spatial extent.
        d = torch.arange(-r, r + 1, device=flow.device, dtype=flow.dtype)
        delta_y, delta_x = torch.meshgrid(d, d, indexing="ij")

        # Source-pixel grid
        bx = torch.arange(W, device=flow.device, dtype=flow.dtype)
        by = torch.arange(H, device=flow.device, dtype=flow.dtype)
        gy, gx = torch.meshgrid(by, bx, indexing="ij")

        out = []
        for lvl, corr in enumerate(self.pyramid):
            _, _, Hc, Wc = corr.shape
            s = 2 ** lvl

            # Target position at this pyramid level
            tx = (gx.unsqueeze(0) + flow[:, 0]) / s        # [B,H,W]
            ty = (gy.unsqueeze(0) + flow[:, 1]) / s

            # Sample (2r+1)² neighbourhood
            sx = tx[..., None, None] + delta_x              # [B,H,W,2r+1,2r+1]
            sy = ty[..., None, None] + delta_y

            # Normalise to [-1, 1] for grid_sample
            sx = 2 * sx / max(Wc - 1, 1) - 1
            sy = 2 * sy / max(Hc - 1, 1) - 1

            grid = torch.stack([sx, sy], dim=-1)             # [B,H,W,2r+1,2r+1,2]
            grid = grid.reshape(B * H * W, (2 * r + 1) ** 2, 1, 2)

            sampled = F.grid_sample(
                corr, grid, align_corners=True, padding_mode="zeros",
            )
            sampled = sampled.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            out.append(sampled)

        return torch.cat(out, dim=1)


# ---------------------------------------------------------------
# Iterative update block
# ---------------------------------------------------------------

class MotionEncoder(nn.Module):
    """Fuse correlation lookup + current flow → motion features."""

    def __init__(self, corr_ch, motion_dim=128):
        super().__init__()
        self.corr_net = nn.Sequential(
            nn.Conv2d(corr_ch, 192, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(192, 128, 3, padding=1),
            nn.LeakyReLU(0.1, True),
        )
        self.flow_net = nn.Sequential(
            nn.Conv2d(2, 64, 7, padding=3),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.1, True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(128 + 64, motion_dim, 3, padding=1),
            nn.LeakyReLU(0.1, True),
        )

    def forward(self, corr, flow):
        return self.fuse(torch.cat([self.corr_net(corr),
                                    self.flow_net(flow)], dim=1))


class ConvGRU(nn.Module):
    """Convolutional Gated Recurrent Unit."""

    def __init__(self, hidden_dim, input_dim):
        super().__init__()
        cat = hidden_dim + input_dim
        self.wz = nn.Conv2d(cat, hidden_dim, 3, padding=1)
        self.wr = nn.Conv2d(cat, hidden_dim, 3, padding=1)
        self.wq = nn.Conv2d(cat, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.wz(hx))
        r = torch.sigmoid(self.wr(hx))
        q = torch.tanh(self.wq(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


class UpdateBlock(nn.Module):
    """One refinement step: corr lookup → GRU → Δflow + upsample mask."""

    def __init__(self, hidden_dim, context_dim, corr_ch, motion_dim=128):
        super().__init__()
        self.enc = MotionEncoder(corr_ch, motion_dim)
        self.gru = ConvGRU(hidden_dim, motion_dim + context_dim)
        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 2, 3, padding=1),
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 8 * 8 * 9, 1),
        )

    def forward(self, h, ctx, corr_feat, flow):
        motion = self.enc(corr_feat, flow)
        h = self.gru(h, torch.cat([motion, ctx], dim=1))
        return h, self.flow_head(h), self.mask_head(h)


# ---------------------------------------------------------------
# Main model
# ---------------------------------------------------------------

class FlowEstimatorRAFT(nn.Module):
    """
    RAFT-style iterative flow estimator for particle tracking.

    Args:
        feature_dim:  feature encoder output channels  (default 256)
        hidden_dim:   GRU hidden state channels        (default 128)
        context_dim:  context feature channels         (default 128)
        corr_levels:  correlation pyramid levels       (default 4)
        corr_radius:  local lookup radius              (default 4)
        num_iters:    refinement iterations             (default 12)
    """

    def __init__(self, feature_dim=256, hidden_dim=128, context_dim=128,
                 corr_levels=4, corr_radius=4, num_iters=12):
        super().__init__()
        self.fnet = FeatureEncoder(1, feature_dim)
        self.cnet = ContextEncoder(1, hidden_dim, context_dim)

        corr_ch = corr_levels * (2 * corr_radius + 1) ** 2
        self.update = UpdateBlock(hidden_dim, context_dim, corr_ch)

        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.num_iters = num_iters

    # ---- learned convex upsampling (8×) ----

    @staticmethod
    def _convex_up8(flow, mask):
        """Upsample low-res flow 8× using a learned convex combination
        of 3×3 neighbours (sharper than bilinear)."""
        B, _, H, W = flow.shape
        f = 8

        mask = mask.reshape(B, 1, f * f, 9, H, W)
        mask = torch.softmax(mask, dim=3)

        # Extract 3×3 patches of the flow  [B, 2, 9, H, W]
        patches = F.unfold(
            F.pad(flow, [1, 1, 1, 1], mode="replicate"), 3,
        ).reshape(B, 2, 9, H, W)

        # Weighted sum → [B, 2, 64, H, W]
        up = (mask * patches.unsqueeze(2)).sum(3)
        up = up.reshape(B, 2, f, f, H, W)
        up = up.permute(0, 1, 4, 2, 5, 3).reshape(B, 2, H * f, W * f)
        return up * f                      # scale to full-resolution pixels

    # ---- forward ----

    def forward(self, images, num_iters=None):
        """
        Args:
            images:    [B, 2, H, W]  (two grayscale frames stacked)
            num_iters: override default iteration count
        Returns:
            list of [B, 2, H, W] flow predictions, one per iteration
        """
        iters = num_iters or self.num_iters
        _, _, H, W = images.shape

        # Pad to multiple of 8 if needed
        ph = (8 - H % 8) % 8
        pw = (8 - W % 8) % 8
        if ph or pw:
            images = F.pad(images, [0, pw, 0, ph], mode="replicate")

        img1, img2 = images[:, 0:1], images[:, 1:2]

        # Feature extraction (shared encoder, 1/8 res)
        fmap1 = self.fnet(img1)
        fmap2 = self.fnet(img2)

        # Context + GRU initial hidden state
        hidden, ctx = self.cnet(img1)

        # Build correlation volume & pyramid
        corr_block = CorrBlock(
            fmap1, fmap2, self.corr_levels, self.corr_radius,
        )

        # Initialise flow at 1/8 resolution to zero
        B, _, H8, W8 = fmap1.shape
        flow = torch.zeros(B, 2, H8, W8, device=images.device)

        predictions = []
        for _ in range(iters):
            flow = flow.detach()                     # stop gradient through past iters

            corr_feat = corr_block(flow)
            hidden, dflow, mask = self.update(hidden, ctx, corr_feat, flow)
            flow = flow + dflow

            flow_up = self._convex_up8(flow, mask)   # [B, 2, H_pad, W_pad]
            if ph or pw:
                flow_up = flow_up[:, :, :H, :W]
            predictions.append(flow_up)

        return predictions
