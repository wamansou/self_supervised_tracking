"""
Self-Supervised Loss Functions for Particle Tracking

Three loss components enforce learning without ground-truth labels:

1. Photometric loss   — warp frame 1 by predicted flow → should match frame 2
2. Divergence loss    — ∂u/∂x + ∂v/∂y ≈ 0  (incompressible flow)
3. Smoothness loss    — penalise large spatial gradients of the flow field
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------
# Differentiable image warping
# ---------------------------------------------------------------

def warp_image(image, flow):
    """
    Warp *image* by *flow* using differentiable bilinear sampling.

    Args:
        image: [B, 1, H, W]
        flow:  [B, 2, H, W]  displacement in **pixels** (u=x, v=y)

    Returns:
        warped: [B, 1, H, W]
    """
    B, _, H, W = image.shape

    # Normalised base grid in [-1, 1]
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=image.device),
        torch.linspace(-1, 1, W, device=image.device),
        indexing="ij",
    )
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    # Pixel displacement → normalised displacement
    flow_norm = torch.stack(
        [flow[:, 0] / (W / 2), flow[:, 1] / (H / 2)], dim=-1
    )  # [B, H, W, 2]

    sample_grid = grid - flow_norm
    return F.grid_sample(
        image, sample_grid, mode="bilinear", padding_mode="border", align_corners=True
    )


# ---------------------------------------------------------------
# Individual losses
# ---------------------------------------------------------------

def photometric_loss(img1, img2, flow, method="charbonnier"):
    """
    Penalise photometric difference between warped frame 1 and frame 2.

    Methods: 'l1', 'l2', 'charbonnier' (robust L1).
    """
    warped = warp_image(img1, flow)
    diff = warped - img2

    if method == "l1":
        return torch.abs(diff).mean()
    elif method == "l2":
        return (diff ** 2).mean()
    elif method == "charbonnier":
        eps = 1e-6
        return torch.sqrt(diff ** 2 + eps).mean()
    raise ValueError(f"Unknown method: {method}")


def _gaussian_window(window_size, sigma=1.5):
    """Create a 2D Gaussian kernel for SSIM computation."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)          # [ws, ws]
    return window.unsqueeze(0).unsqueeze(0)             # [1, 1, ws, ws]


def ssim_loss(img1, img2, flow, window_size=11):
    """
    Structural-Similarity (SSIM) based photometric loss.

    More robust than pixel-wise losses for particle images because it
    compares *local structure* (edges, patterns) rather than raw intensity,
    avoiding local-minima where the model matches the wrong particles.

    Returns  1 − mean(SSIM)  so that 0 = perfect match.
    """
    warped = warp_image(img1, flow)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    pad = window_size // 2

    window = _gaussian_window(window_size).to(img1.device, img1.dtype)

    mu1 = F.conv2d(warped, window, padding=pad)
    mu2 = F.conv2d(img2, window, padding=pad)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(warped ** 2, window, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=pad) - mu2_sq
    sigma12 = F.conv2d(warped * img2, window, padding=pad) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1.0 - ssim_map.mean()


def divergence_loss(flow):
    """
    Zero-divergence constraint for incompressible flow.

    div(u) = ∂u/∂x + ∂v/∂y  →  should be ≈ 0.
    Central finite differences, interior pixels only.
    """
    u = flow[:, 0:1]  # [B, 1, H, W]
    v = flow[:, 1:2]

    du_dx = (u[:, :, :, 2:] - u[:, :, :, :-2]) / 2.0   # [B,1,H,W-2]
    dv_dy = (v[:, :, 2:, :] - v[:, :, :-2, :]) / 2.0   # [B,1,H-2,W]

    # Crop to common interior
    du_dx = du_dx[:, :, 1:-1, :]   # [B,1,H-2,W-2]
    dv_dy = dv_dy[:, :, :, 1:-1]   # [B,1,H-2,W-2]

    div = du_dx + dv_dy
    return (div ** 2).mean()


def smoothness_loss(flow):
    """
    First-order spatial smoothness (total-variation style).

    Penalises |∂flow/∂x|² + |∂flow/∂y|².
    """
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    return (dx ** 2).mean() + (dy ** 2).mean()


# ---------------------------------------------------------------
# Combined loss module
# ---------------------------------------------------------------

class SelfSupervisedTrackingLoss(nn.Module):
    """
    L = Σ_i  γ^(N-1-i) · [λ_photo·L_photo + λ_div·L_div + λ_smooth·L_smooth]

    Supports both single predictions and iterative refinement sequences.
    When given a list of predictions (from RAFT), applies exponentially
    increasing weights so later (more refined) predictions matter more.

    Args:
        lambda_photo:  weight for photometric reconstruction
        lambda_div:    weight for zero-divergence (physics) constraint
        lambda_smooth: weight for spatial smoothness regularisation
        ssim_weight:   blend factor between Charbonnier and SSIM
        gamma:         exponential weight decay for iterative loss (default 0.8)
    """

    def __init__(self, lambda_photo=1.0, lambda_div=0.5, lambda_smooth=0.1,
                 ssim_weight=0.5, gamma=0.8):
        super().__init__()
        self.lp = lambda_photo
        self.ld = lambda_div
        self.ls = lambda_smooth
        self.ssim_w = ssim_weight
        self.gamma = gamma

    def forward(self, img1, img2, pred_flow):
        """
        Args:
            img1, img2: [B, 1, H, W]
            pred_flow:  [B, 2, H, W]  OR  list of [B, 2, H, W]
        Returns:
            total_loss (scalar), loss_dict (for logging — final iteration only)
        """
        if not isinstance(pred_flow, (list, tuple)):
            pred_flow = [pred_flow]

        n = len(pred_flow)
        total = 0.0

        for i, flow in enumerate(pred_flow):
            w = self.gamma ** (n - 1 - i)

            lp_charb = photometric_loss(img1, img2, flow)
            lp_ssim = ssim_loss(img1, img2, flow)
            lp = (1.0 - self.ssim_w) * lp_charb + self.ssim_w * lp_ssim

            ldv = divergence_loss(flow)
            ls = smoothness_loss(flow)

            total = total + w * (self.lp * lp + self.ld * ldv + self.ls * ls)

            # Keep final-iteration metrics for logging
            if i == n - 1:
                last_photo, last_div, last_smooth = lp, ldv, ls

        return total, {
            "total": total.item(),
            "photometric": last_photo.item(),
            "divergence": last_div.item(),
            "smoothness": last_smooth.item(),
        }
