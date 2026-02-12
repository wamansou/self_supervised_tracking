"""
Training script for Self-Supervised Particle Tracking

Usage:
    python train.py                          # train with defaults
    python train.py --epochs 100 --bs 16     # custom settings
    python train.py --resume checkpoints/checkpoint_epoch_20.pth
"""

import os
import json
import argparse
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data_generator import SyntheticParticleDataset
from model import FlowEstimatorUNet
from losses import SelfSupervisedTrackingLoss
from visualizer import TrainingVisualizer


# ------------------------------------------------------------------
# Default configuration
# ------------------------------------------------------------------

DEFAULT_CONFIG = {
    # ----- Data -----
    "num_train_samples": 4000,
    "num_val_samples": 400,
    "image_size": 256,
    "num_particles_range": [500, 2000],
    "noise_level": 0.05,
    "max_displacement": 8.0,
    # ----- Model -----
    "base_features": 64,
    # ----- Loss weights -----
    "lambda_photo": 1.0,
    "lambda_div": 0.3,
    "lambda_smooth": 0.15,
    "ssim_weight": 0.5,
    # ----- Optimiser -----
    "batch_size": 64,
    "num_epochs": 80,
    "learning_rate": 3e-4,
    "weight_decay": 1e-5,
    # ----- Misc -----
    "num_workers": 8,
    "save_every": 10,
    "output_dir": "checkpoints",
    "use_amp": True,
    # ----- Visualisation -----
    "viz_every": 2,       # log images to TensorBoard every N epochs
    "viz_samples": 3,     # how many samples to visualise per epoch
    "log_dir": "runs/",   # TensorBoard log directory
}


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Datasets ----
    common = dict(
        image_size=cfg["image_size"],
        num_particles_range=tuple(cfg["num_particles_range"]),
        noise_level=cfg["noise_level"],
        flow_type="random",
        max_displacement=cfg["max_displacement"],
    )
    train_ds = SyntheticParticleDataset(num_samples=cfg["num_train_samples"], **common)
    val_ds   = SyntheticParticleDataset(num_samples=cfg["num_val_samples"],   **common)

    _persistent = cfg["num_workers"] > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=_persistent,
        prefetch_factor=4 if _persistent else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=_persistent,
        prefetch_factor=4 if _persistent else None,
    )

    # ---- cuDNN benchmark (fixed input sizes → pick fastest kernels) ----
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ---- Model ----
    model = FlowEstimatorUNet(
        in_channels=2, base_features=cfg["base_features"]
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ---- torch.compile (PyTorch 2.x graph-mode speedup) ----
    if hasattr(torch, "compile"):
        model = torch.compile(model)
        print("torch.compile enabled")

    # ---- Loss / Optimiser / Scheduler ----
    criterion = SelfSupervisedTrackingLoss(
        cfg["lambda_photo"], cfg["lambda_div"], cfg["lambda_smooth"],
        ssim_weight=cfg["ssim_weight"],
    )
    optimizer = optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["num_epochs"], eta_min=cfg["learning_rate"] * 0.01
    )

    # ---- AMP (mixed precision) ----
    use_amp = cfg.get("use_amp", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ---- Resume support ----
    start_epoch = 1
    if cfg.get("resume"):
        ckpt = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ---- Visualiser (TensorBoard) ----
    viz = TrainingVisualizer(
        log_dir=cfg["log_dir"],
        viz_every=cfg["viz_every"],
        num_samples=cfg["viz_samples"],
    )
    print(f"TensorBoard → {cfg['log_dir']}  (log images every {cfg['viz_every']} epochs)")
    if use_amp:
        print("Mixed-precision training (AMP) enabled")

    # ---- Logging ----
    history = {"train": [], "val": []}
    best_val = float("inf")

    header = (
        f"{'Ep':>4} | {'Tr Loss':>9} | {'Va Loss':>9} | "
        f"{'Photo':>7} | {'Div':>7} | {'Smooth':>7} | {'EPE':>7}"
    )
    print(f"\n{header}\n{'-' * len(header)}")

    for epoch in range(start_epoch, cfg["num_epochs"] + 1):
        # ========== Train ==========
        model.train()
        tr = {"total": 0, "photometric": 0, "divergence": 0, "smoothness": 0}

        for images, _ in train_loader:
            images = images.to(device)
            img1, img2 = images[:, 0:1], images[:, 1:2]

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(images)
                loss, ld = criterion(img1, img2, pred)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            for k in tr:
                tr[k] += ld[k]

        n_tr = len(train_loader)
        for k in tr:
            tr[k] /= n_tr

        # ========== Validate ==========
        model.eval()
        va = {"total": 0, "photometric": 0, "divergence": 0, "smoothness": 0}
        epe_sum = 0.0

        with torch.no_grad():
            for images, gt_flow in val_loader:
                images = images.to(device)
                gt_flow = gt_flow.to(device)
                img1, img2 = images[:, 0:1], images[:, 1:2]

                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(images)
                    loss, ld = criterion(img1, img2, pred)

                for k in va:
                    va[k] += ld[k]

                # End-point error (supervised metric — for monitoring only)
                epe = torch.sqrt(((pred - gt_flow) ** 2).sum(dim=1)).mean()
                epe_sum += epe.item()

        n_va = len(val_loader)
        for k in va:
            va[k] /= n_va
        avg_epe = epe_sum / n_va

        scheduler.step()

        # ========== Logging ==========
        history["train"].append(tr.copy())
        history["val"].append({**va.copy(), "epe": avg_epe})

        print(
            f"{epoch:>4} | {tr['total']:>9.4f} | {va['total']:>9.4f} | "
            f"{va['photometric']:>7.4f} | {va['divergence']:>7.4f} | "
            f"{va['smoothness']:>7.4f} | {avg_epe:>7.4f}"
        )

        # ========== TensorBoard ==========
        current_lr = scheduler.get_last_lr()[0]
        viz.log_scalars(epoch, tr, va, epe=avg_epe, lr=current_lr)

        # Log images on viz epochs — grab one val batch for visuals
        if epoch % cfg["viz_every"] == 0:
            with torch.no_grad():
                viz_imgs, viz_gt = next(iter(val_loader))
                viz_imgs = viz_imgs.to(device)
                viz_gt = viz_gt.to(device)
                viz_pred = model(viz_imgs)
                viz.log_images(epoch, viz_imgs, viz_pred, viz_gt)

        # Best model
        if va["total"] < best_val:
            best_val = va["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val,
                    "config": cfg,
                },
                os.path.join(cfg["output_dir"], "best_model.pth"),
            )

        # Periodic checkpoint
        if epoch % cfg["save_every"] == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": va["total"],
                    "config": cfg,
                },
                os.path.join(cfg["output_dir"], f"checkpoint_epoch_{epoch}.pth"),
            )

    # ---- Save history ----
    with open(os.path.join(cfg["output_dir"], "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    viz.close()
    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Outputs → {cfg['output_dir']}/")
    return model, history


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train self-supervised particle tracker")
    p.add_argument("--epochs",   type=int,   default=None, help="Number of epochs")
    p.add_argument("--bs",       type=int,   default=None, help="Batch size")
    p.add_argument("--lr",       type=float, default=None, help="Learning rate")
    p.add_argument("--img-size", type=int,   default=None, help="Image size (px)")
    p.add_argument("--features", type=int,   default=None, help="Base feature count")
    p.add_argument("--l-photo",  type=float, default=None, help="Photo loss weight")
    p.add_argument("--l-div",    type=float, default=None, help="Div loss weight")
    p.add_argument("--l-smooth", type=float, default=None, help="Smooth loss weight")
    p.add_argument("--ssim-w",   type=float, default=None, help="SSIM blend weight (0=pure Charbonnier, 1=pure SSIM)")
    p.add_argument("--no-amp",   action="store_true",       help="Disable mixed-precision training")
    p.add_argument("--resume",    type=str,   default=None, help="Checkpoint to resume")
    p.add_argument("--out",       type=str,   default=None, help="Output directory")
    p.add_argument("--viz-every", type=int,   default=None, help="Log images every N epochs")
    p.add_argument("--log-dir",   type=str,   default=None, help="TensorBoard log dir")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = DEFAULT_CONFIG.copy()

    overrides = {
        "num_epochs":    args.epochs,
        "batch_size":    args.bs,
        "learning_rate": args.lr,
        "image_size":    args.img_size,
        "base_features": args.features,
        "lambda_photo":  args.l_photo,
        "lambda_div":    args.l_div,
        "lambda_smooth": args.l_smooth,
        "ssim_weight":   args.ssim_w,
        "resume":        args.resume,
        "output_dir":    args.out,
        "viz_every":     args.viz_every,
        "log_dir":       args.log_dir,
    }
    if args.no_amp:
        cfg["use_amp"] = False
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v

    print("Config:")
    for k, v in sorted(cfg.items()):
        print(f"  {k}: {v}")
    print()

    train(cfg)
