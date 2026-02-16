"""
Generate publication-quality figures from the trained model.

Produces the same visualisations that appeared in TensorBoard — frame pairs,
HSV flow colour maps, quiver overlays, warp quality, and error/divergence
heat maps — but saves them as standalone PNGs for each flow type.

Usage:
    python generate_figures.py                                # all flow types, default checkpoint
    python generate_figures.py --flows vortex channel         # specific flow types
    python generate_figures.py --checkpoint checkpoints/checkpoint_epoch_80.pth
    python generate_figures.py --num-samples 5 --dpi 200
"""

import os
import argparse

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from data_generator import SyntheticParticleDataset
from model import FlowEstimatorRAFT
from losses import warp_image


# ------------------------------------------------------------------
# Visualisation helpers  (adapted from visualizer.py)
# ------------------------------------------------------------------

def flow_to_hsv(flow_tensor):
    """[2, H, W] flow → [H, W, 3] RGB via the HSV colour-wheel convention."""
    u = flow_tensor[0].cpu().numpy()
    v = flow_tensor[1].cpu().numpy()
    mag = np.sqrt(u ** 2 + v ** 2)
    ang = np.arctan2(v, u)

    hsv = np.zeros((*u.shape, 3), dtype=np.float32)
    hsv[..., 0] = (ang + np.pi) / (2 * np.pi)
    hsv[..., 1] = 1.0
    hsv[..., 2] = mag / (mag.max() + 1e-8)
    return mcolors.hsv_to_rgb(hsv)


def save_frame_pair(img1, img2, path, dpi=150):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5), facecolor="black")
    for ax, im, title in [(ax1, img1, "Frame 1"), (ax2, img2, "Frame 2")]:
        ax.imshow(im.cpu().numpy(), cmap="inferno", vmin=0, vmax=1)
        ax.set_title(title, color="white", fontsize=11)
        ax.axis("off")
    fig.tight_layout(pad=1.5)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_flow_hsv(pred, gt, path, dpi=150):
    pred_rgb = flow_to_hsv(pred)
    gt_rgb = flow_to_hsv(gt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5), facecolor="black")
    ax1.imshow(pred_rgb);  ax1.set_title("Predicted flow", color="white", fontsize=11);  ax1.axis("off")
    ax2.imshow(gt_rgb);    ax2.set_title("Ground truth flow", color="white", fontsize=11); ax2.axis("off")
    fig.tight_layout(pad=1.5)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_quiver(img1, pred, gt, path, dpi=150):
    H, W = img1.shape
    step = max(H, W) // 16

    yy, xx = np.mgrid[0:H:step, 0:W:step]
    pu = pred[0].cpu().numpy()[::step, ::step]
    pv = pred[1].cpu().numpy()[::step, ::step]
    gu = gt[0].cpu().numpy()[::step, ::step]
    gv = gt[1].cpu().numpy()[::step, ::step]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.5), facecolor="black")
    for ax, u, v, title in [(ax1, pu, pv, "Predicted"), (ax2, gu, gv, "Ground truth")]:
        ax.imshow(img1.cpu().numpy(), cmap="gray", alpha=0.5)
        ax.quiver(xx, yy, u, v, np.sqrt(u ** 2 + v ** 2), cmap="cool", scale=None, width=0.003)
        ax.set_title(title, color="white", fontsize=11)
        ax.axis("off")
    fig.tight_layout(pad=1.5)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_warp(img1, img2, pred, path, dpi=150):
    warped = warp_image(
        img1.unsqueeze(0).unsqueeze(0),
        pred.unsqueeze(0),
    ).squeeze()
    diff = (warped - img2).abs().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), facecolor="black")
    titles = ["Warped frame 1", "Frame 2 (target)", "|Warp − Target|"]
    imgs = [warped.cpu().numpy(), img2.cpu().numpy(), diff]
    cmaps = ["inferno", "inferno", "magma"]
    for ax, im, t, cm in zip(axes, imgs, titles, cmaps):
        ax.imshow(im, cmap=cm, vmin=0, vmax=max(1, im.max()))
        ax.set_title(t, color="white", fontsize=11)
        ax.axis("off")
    fig.tight_layout(pad=1.5)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_error_maps(pred, gt, path, dpi=150):
    epe = torch.sqrt(((pred - gt) ** 2).sum(dim=0)).cpu().numpy()

    u, v = pred[0:1].unsqueeze(0), pred[1:2].unsqueeze(0)
    du_dx = (u[:, :, :, 2:] - u[:, :, :, :-2]) / 2.0
    dv_dy = (v[:, :, 2:, :] - v[:, :, :-2, :]) / 2.0
    du_dx = du_dx[:, :, 1:-1, :]
    dv_dy = dv_dy[:, :, :, 1:-1]
    div = (du_dx + dv_dy).squeeze().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), facecolor="black")

    im1 = ax1.imshow(epe, cmap="hot")
    ax1.set_title(f"EPE  (mean {epe.mean():.2f} px)", color="white", fontsize=11)
    ax1.axis("off")
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    lim = max(abs(div.min()), abs(div.max()), 0.01)
    im2 = ax2.imshow(div, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax2.set_title("Divergence  (should → 0)", color="white", fontsize=11)
    ax2.axis("off")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    fig.tight_layout(pad=1.5)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ------------------------------------------------------------------
# Flow-type display names (for figure titles / folder names)
# ------------------------------------------------------------------

FLOW_LABELS = {
    "vortex":       "Lamb–Oseen Vortex",
    "uniform":      "Uniform Translation",
    "shear":        "Linear Shear Flow",
    "channel":      "Poiseuille Channel Flow",
    "multi_vortex": "Multi-Vortex Superposition",
}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load model ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = FlowEstimatorRAFT(
        feature_dim=cfg["feature_dim"],
        hidden_dim=cfg["hidden_dim"],
        context_dim=cfg["context_dim"],
        corr_levels=cfg["corr_levels"],
        corr_radius=cfg["corr_radius"],
        num_iters=cfg["num_iters"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}  (epoch {ckpt['epoch']})")

    # ---- Generate for each flow type ----
    os.makedirs(args.output_dir, exist_ok=True)

    for flow_type in args.flows:
        label = FLOW_LABELS.get(flow_type, flow_type)
        flow_dir = os.path.join(args.output_dir, flow_type)
        os.makedirs(flow_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  Flow type: {label}")
        print(f"  Output   : {flow_dir}/")
        print(f"{'='*60}")

        ds = SyntheticParticleDataset(
            num_samples=args.num_samples,
            image_size=cfg["image_size"],
            num_particles_range=tuple(cfg["num_particles_range"]),
            noise_level=cfg["noise_level"],
            flow_type=flow_type,
            max_displacement=cfg["max_displacement"],
        )

        for idx in range(len(ds)):
            images, gt_flow = ds[idx]                      # [2,H,W], [2,H,W]
            images_dev = images.unsqueeze(0).to(device)    # [1,2,H,W]
            gt_flow_dev = gt_flow.to(device)               # [2,H,W]

            with torch.no_grad():
                pred_flow = model(images_dev)[-1].squeeze(0)  # [2,H,W]

            img1 = images[0]
            img2 = images[1]
            pred = pred_flow.cpu()
            gt   = gt_flow

            prefix = f"sample_{idx:02d}"

            # Compute EPE for this sample
            epe = torch.sqrt(((pred - gt) ** 2).sum(dim=0)).mean().item()
            print(f"  [{prefix}]  EPE = {epe:.3f} px")

            save_frame_pair(img1, img2,
                            os.path.join(flow_dir, f"{prefix}_1_frames.png"),     dpi=args.dpi)
            save_flow_hsv(pred, gt,
                            os.path.join(flow_dir, f"{prefix}_2_flow_hsv.png"),   dpi=args.dpi)
            save_quiver(img1, pred, gt,
                            os.path.join(flow_dir, f"{prefix}_3_quiver.png"),     dpi=args.dpi)
            save_warp(img1, img2, pred,
                            os.path.join(flow_dir, f"{prefix}_4_warp.png"),       dpi=args.dpi)
            save_error_maps(pred, gt,
                            os.path.join(flow_dir, f"{prefix}_5_error_div.png"),  dpi=args.dpi)

    print(f"\nAll figures saved to {args.output_dir}/")


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate report-ready figures from the trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pth",
        help="Path to model checkpoint (default: checkpoints/best_model.pth)",
    )
    p.add_argument(
        "--flows", nargs="+",
        default=["vortex", "uniform", "shear", "channel", "multi_vortex"],
        choices=["vortex", "uniform", "shear", "channel", "multi_vortex"],
        help="Flow types to generate figures for (default: all)",
    )
    p.add_argument(
        "--num-samples", type=int, default=3,
        help="Number of samples per flow type (default: 3)",
    )
    p.add_argument(
        "--output-dir", type=str, default="figures",
        help="Output directory (default: figures/)",
    )
    p.add_argument(
        "--dpi", type=int, default=200,
        help="Figure DPI (default: 200)",
    )
    return p.parse_args()


if __name__ == "__main__":
    generate(parse_args())
