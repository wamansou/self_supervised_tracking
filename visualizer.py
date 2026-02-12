"""
Real-Time Training Visualiser (TensorBoard)

Logs to TensorBoard during training:
  - Loss curves  (scalars)
  - Particle image pairs with predicted flow overlays  (images)
  - HSV optical-flow-style colour maps  (images)
  - Quiver plots of velocity field  (images)
  - Divergence field heat maps  (images)

Usage:
    # In another terminal (with SSH tunnel):
    #   ssh -L 6006:localhost:6006 user@gpu-machine
    #   source venv/bin/activate
    #   tensorboard --logdir runs/
    #
    # Then open http://localhost:6006 in your browser.
"""

import math
import io

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.tensorboard import SummaryWriter

from losses import warp_image


class TrainingVisualizer:
    """
    Attaches to the training loop and pushes rich visualisations to
    TensorBoard at configurable intervals.

    Args:
        log_dir:       TensorBoard log directory  (default: "runs/")
        viz_every:     log images every N epochs   (default: 2)
        num_samples:   how many samples to visualise per epoch  (default: 3)
    """

    def __init__(self, log_dir="runs/", viz_every=2, num_samples=3):
        self.writer = SummaryWriter(log_dir)
        self.viz_every = viz_every
        self.num_samples = num_samples

    def close(self):
        self.writer.close()

    # ------------------------------------------------------------------
    # Scalar logging
    # ------------------------------------------------------------------

    def log_scalars(self, epoch, train_losses, val_losses, epe=None, lr=None):
        """Log all scalar metrics."""
        for key in train_losses:
            self.writer.add_scalars(
                f"loss/{key}",
                {"train": train_losses[key], "val": val_losses[key]},
                epoch,
            )
        if epe is not None:
            self.writer.add_scalar("metrics/end_point_error_px", epe, epoch)
        if lr is not None:
            self.writer.add_scalar("optim/learning_rate", lr, epoch)

    # ------------------------------------------------------------------
    # Image logging
    # ------------------------------------------------------------------

    def log_images(self, epoch, images, pred_flow, gt_flow):
        """
        Log a rich set of visualisations for one batch.

        Args:
            images:    [B, 2, H, W]
            pred_flow: [B, 2, H, W]
            gt_flow:   [B, 2, H, W]
        """
        if epoch % self.viz_every != 0:
            return

        n = min(self.num_samples, images.shape[0])

        for i in range(n):
            tag = f"sample_{i}"
            img1 = images[i, 0]        # [H, W]
            img2 = images[i, 1]
            pred = pred_flow[i]         # [2, H, W]
            gt = gt_flow[i]

            # 1) Side-by-side particle frames
            self._log_frame_pair(epoch, tag, img1, img2)

            # 2) HSV flow colour maps (pred vs gt)
            self._log_flow_hsv(epoch, tag, pred, gt)

            # 3) Quiver overlay on frame 1
            self._log_quiver(epoch, tag, img1, pred, gt)

            # 4) Warp quality  (warped frame 1 vs frame 2)
            self._log_warp(epoch, tag, img1, img2, pred)

            # 5) Error & divergence heat maps
            self._log_error_maps(epoch, tag, pred, gt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fig_to_tensor(self, fig):
        """Render a matplotlib figure to a [3, H, W] uint8 tensor."""
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        arr = np.asarray(buf)[:, :, :3]           # drop alpha
        plt.close(fig)
        return torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]

    # ---- 1) Frame pair ----

    def _log_frame_pair(self, epoch, tag, img1, img2):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5), facecolor="black")
        for ax, im, title in [(ax1, img1, "Frame 1"), (ax2, img2, "Frame 2")]:
            ax.imshow(im.cpu().numpy(), cmap="inferno", vmin=0, vmax=1)
            ax.set_title(title, color="white", fontsize=10)
            ax.axis("off")
        fig.tight_layout(pad=1.5)
        self.writer.add_image(f"{tag}/1_frames", self._fig_to_tensor(fig), epoch)

    # ---- 2) HSV flow colour wheel ----

    @staticmethod
    def _flow_to_hsv(flow_tensor):
        """
        Convert a [2, H, W] flow field to an RGB image using the
        optical-flow HSV convention  (hue = direction, value = magnitude).
        """
        u = flow_tensor[0].cpu().numpy()
        v = flow_tensor[1].cpu().numpy()
        mag = np.sqrt(u ** 2 + v ** 2)
        ang = np.arctan2(v, u)  # [-pi, pi]

        hsv = np.zeros((*u.shape, 3), dtype=np.float32)
        hsv[..., 0] = (ang + np.pi) / (2 * np.pi)          # hue  [0, 1]
        hsv[..., 1] = 1.0                                    # full saturation
        hsv[..., 2] = mag / (mag.max() + 1e-8)              # value [0, 1]

        rgb = mcolors.hsv_to_rgb(hsv)
        return rgb  # [H, W, 3] float [0,1]

    def _log_flow_hsv(self, epoch, tag, pred, gt):
        pred_rgb = self._flow_to_hsv(pred)
        gt_rgb = self._flow_to_hsv(gt)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5), facecolor="black")
        ax1.imshow(pred_rgb)
        ax1.set_title("Predicted flow", color="white", fontsize=10)
        ax1.axis("off")
        ax2.imshow(gt_rgb)
        ax2.set_title("Ground truth flow", color="white", fontsize=10)
        ax2.axis("off")
        fig.tight_layout(pad=1.5)
        self.writer.add_image(f"{tag}/2_flow_hsv", self._fig_to_tensor(fig), epoch)

    # ---- 3) Quiver plot ----

    def _log_quiver(self, epoch, tag, img1, pred, gt):
        H, W = img1.shape
        step = max(H, W) // 16        # ~16 arrows per axis

        yy, xx = np.mgrid[0:H:step, 0:W:step]
        pu = pred[0].cpu().numpy()[::step, ::step]
        pv = pred[1].cpu().numpy()[::step, ::step]
        gu = gt[0].cpu().numpy()[::step, ::step]
        gv = gt[1].cpu().numpy()[::step, ::step]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.5), facecolor="black")

        for ax, u, v, title in [
            (ax1, pu, pv, "Predicted"),
            (ax2, gu, gv, "Ground truth"),
        ]:
            ax.imshow(img1.cpu().numpy(), cmap="gray", alpha=0.5)
            ax.quiver(
                xx, yy, u, v,
                np.sqrt(u ** 2 + v ** 2),
                cmap="cool", scale=None, width=0.003,
            )
            ax.set_title(title, color="white", fontsize=10)
            ax.axis("off")

        fig.tight_layout(pad=1.5)
        self.writer.add_image(f"{tag}/3_quiver", self._fig_to_tensor(fig), epoch)

    # ---- 4) Warp quality ----

    def _log_warp(self, epoch, tag, img1, img2, pred):
        warped = warp_image(
            img1.unsqueeze(0).unsqueeze(0),  # [1,1,H,W]
            pred.unsqueeze(0),                # [1,2,H,W]
        ).squeeze()

        diff = (warped - img2).abs().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), facecolor="black")
        titles = ["Warped frame 1", "Frame 2 (target)", "|Warp − Target|"]
        imgs = [warped.cpu().numpy(), img2.cpu().numpy(), diff]
        cmaps = ["inferno", "inferno", "magma"]

        for ax, im, t, cm in zip(axes, imgs, titles, cmaps):
            ax.imshow(im, cmap=cm, vmin=0, vmax=max(1, im.max()))
            ax.set_title(t, color="white", fontsize=10)
            ax.axis("off")

        fig.tight_layout(pad=1.5)
        self.writer.add_image(f"{tag}/4_warp_quality", self._fig_to_tensor(fig), epoch)

    # ---- 5) Error & divergence ----

    def _log_error_maps(self, epoch, tag, pred, gt):
        epe = torch.sqrt(((pred - gt) ** 2).sum(dim=0)).cpu().numpy()

        # Divergence
        u, v = pred[0:1].unsqueeze(0), pred[1:2].unsqueeze(0)
        du_dx = (u[:, :, :, 2:] - u[:, :, :, :-2]) / 2.0
        dv_dy = (v[:, :, 2:, :] - v[:, :, :-2, :]) / 2.0
        du_dx = du_dx[:, :, 1:-1, :]
        dv_dy = dv_dy[:, :, :, 1:-1]
        div = (du_dx + dv_dy).squeeze().cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), facecolor="black")

        im1 = ax1.imshow(epe, cmap="hot")
        ax1.set_title(f"EPE  (mean {epe.mean():.2f} px)", color="white", fontsize=10)
        ax1.axis("off")
        fig.colorbar(im1, ax=ax1, fraction=0.046)

        lim = max(abs(div.min()), abs(div.max()), 0.01)
        im2 = ax2.imshow(div, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax2.set_title("Divergence  (should → 0)", color="white", fontsize=10)
        ax2.axis("off")
        fig.colorbar(im2, ax=ax2, fraction=0.046)

        fig.tight_layout(pad=1.5)
        self.writer.add_image(f"{tag}/5_error_div", self._fig_to_tensor(fig), epoch)
