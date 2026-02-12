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

    sample_grid = grid + flow_norm
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
    L = λ_photo · L_photo  +  λ_div · L_div  +  λ_smooth · L_smooth

    Args:
        lambda_photo:  weight for photometric reconstruction
        lambda_div:    weight for zero-divergence (physics) constraint
        lambda_smooth: weight for spatial smoothness regularisation
    """

    def __init__(self, lambda_photo=1.0, lambda_div=0.5, lambda_smooth=0.1):
        super().__init__()
        self.lp = lambda_photo
        self.ld = lambda_div
        self.ls = lambda_smooth

    def forward(self, img1, img2, pred_flow):
        """
        Args:
            img1, img2: [B, 1, H, W]
            pred_flow:  [B, 2, H, W]
        Returns:
            total_loss (scalar), loss_dict (for logging)
        """
        lp = photometric_loss(img1, img2, pred_flow)
        ldv = divergence_loss(pred_flow)
        ls = smoothness_loss(pred_flow)

        total = self.lp * lp + self.ld * ldv + self.ls * ls

        return total, {
            "total": total.item(),
            "photometric": lp.item(),
            "divergence": ldv.item(),
            "smoothness": ls.item(),
        }
