"""
Evaluation & Visualisation for Self-Supervised Particle Tracking

Usage:
    python evaluate.py                                          # defaults
    python evaluate.py --checkpoint checkpoints/best_model.pth  # specific model
    python evaluate.py --num-samples 100 --save-dir results     # more samples
"""

import os
import argparse

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from data_generator import SyntheticParticleDataset
from model import FlowEstimatorUNet
from losses import warp_image, divergence_loss


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def load_model(path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = FlowEstimatorUNet(in_channels=2, base_features=cfg["base_features"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def compute_metrics(pred, gt):
    """
    Returns dict with:
        epe            – mean end-point error (px)
        angular_error  – mean angular error (degrees)
        divergence     – mean squared divergence
        relative_error – EPE / mean |gt|
    """
    epe = torch.sqrt(((pred - gt) ** 2).sum(dim=0)).mean()

    # Angular error (add z=1 trick)
    p = pred.reshape(2, -1).T
    g = gt.reshape(2, -1).T
    p3 = torch.cat([p, torch.ones(p.shape[0], 1, device=pred.device)], 1)
    g3 = torch.cat([g, torch.ones(g.shape[0], 1, device=gt.device)], 1)
    cos_a = (p3 * g3).sum(1) / (p3.norm(dim=1) * g3.norm(dim=1) + 1e-8)
    ang = torch.acos(cos_a.clamp(-1, 1)).mean() * 180 / np.pi

    div = divergence_loss(pred.unsqueeze(0))
    gt_mag = torch.sqrt((gt ** 2).sum(dim=0)).mean()
    rel = epe / (gt_mag + 1e-8)

    return {
        "epe": epe.item(),
        "angular_error": ang.item(),
        "divergence": div.item(),
        "relative_error": rel.item(),
    }


# ------------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------------

def visualize(img1, img2, pred, gt, save_path=None):
    """6-panel figure: frames, warped, pred/gt magnitude, error."""
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    ax[0, 0].imshow(img1.squeeze().cpu().numpy(), cmap="gray")
    ax[0, 0].set_title("Frame 1")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(img2.squeeze().cpu().numpy(), cmap="gray")
    ax[0, 1].set_title("Frame 2")
    ax[0, 1].axis("off")

    warped = warp_image(
        img1.unsqueeze(0).unsqueeze(0), pred.unsqueeze(0)
    )
    ax[0, 2].imshow(warped.squeeze().cpu().numpy(), cmap="gray")
    ax[0, 2].set_title("Frame 1 → warped")
    ax[0, 2].axis("off")

    pred_mag = torch.sqrt((pred ** 2).sum(0)).cpu().numpy()
    im1 = ax[1, 0].imshow(pred_mag, cmap="jet")
    ax[1, 0].set_title("Pred |flow|")
    ax[1, 0].axis("off")
    fig.colorbar(im1, ax=ax[1, 0], fraction=0.046)

    gt_mag = torch.sqrt((gt ** 2).sum(0)).cpu().numpy()
    im2 = ax[1, 1].imshow(gt_mag, cmap="jet")
    ax[1, 1].set_title("GT |flow|")
    ax[1, 1].axis("off")
    fig.colorbar(im2, ax=ax[1, 1], fraction=0.046)

    err = torch.sqrt(((pred - gt) ** 2).sum(0)).cpu().numpy()
    im3 = ax[1, 2].imshow(err, cmap="hot")
    ax[1, 2].set_title(f"EPE  (mean {err.mean():.3f} px)")
    ax[1, 2].axis("off")
    fig.colorbar(im3, ax=ax[1, 2], fraction=0.046)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ------------------------------------------------------------------
# Full evaluation
# ------------------------------------------------------------------

def evaluate(checkpoint, num_samples=50, save_dir="evaluation_results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(checkpoint, device)
    os.makedirs(save_dir, exist_ok=True)

    flow_types = SyntheticParticleDataset.FLOW_TYPES
    all_metrics = {}

    for ft in flow_types:
        print(f"\n--- {ft} ---")
        ds = SyntheticParticleDataset(
            num_samples=num_samples,
            image_size=cfg["image_size"],
            num_particles_range=tuple(cfg["num_particles_range"]),
            noise_level=cfg["noise_level"],
            flow_type=ft,
            max_displacement=cfg["max_displacement"],
        )

        run = []
        for i in range(len(ds)):
            imgs, gt = ds[i]
            imgs = imgs.unsqueeze(0).to(device)
            gt = gt.to(device)

            with torch.no_grad():
                pred = model(imgs).squeeze(0)

            run.append(compute_metrics(pred, gt))

            if i < 3:
                visualize(
                    imgs[0, 0], imgs[0, 1], pred, gt,
                    os.path.join(save_dir, f"{ft}_{i}.png"),
                )

        mean = {k: np.mean([m[k] for m in run]) for k in run[0]}
        std  = {k: np.std([m[k] for m in run])  for k in run[0]}
        all_metrics[ft] = {"mean": mean, "std": std}

        print(f"  EPE:  {mean['epe']:.4f} ± {std['epe']:.4f} px")
        print(f"  Ang:  {mean['angular_error']:.2f} ± {std['angular_error']:.2f}°")
        print(f"  Div:  {mean['divergence']:.6f}")
        print(f"  Rel:  {mean['relative_error']:.4f}")

    # Summary table
    print("\n" + "=" * 72)
    print(f"{'Flow':<16} {'EPE (px)':<16} {'Ang (°)':<16} {'Rel Err':<16}")
    print("-" * 72)
    for ft in flow_types:
        m, s = all_metrics[ft]["mean"], all_metrics[ft]["std"]
        print(
            f"{ft:<16} {m['epe']:.3f}±{s['epe']:.3f}       "
            f"{m['angular_error']:.2f}±{s['angular_error']:.2f}        "
            f"{m['relative_error']:.4f}±{s['relative_error']:.4f}"
        )
    print("=" * 72)

    return all_metrics


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="checkpoints/best_model.pth")
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--save-dir",    default="evaluation_results")
    args = p.parse_args()
    evaluate(args.checkpoint, args.num_samples, args.save_dir)
