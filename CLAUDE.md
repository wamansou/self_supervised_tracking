# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Self-supervised particle tracking system that estimates dense displacement fields (optical flow) from consecutive particle image pairs without ground-truth labels. Uses physics-informed losses for particle image velocimetry (PIV) applications. The current model is a RAFT-style iterative flow estimator (`model.py`); `model_unet.py` is an archived baseline.

## Commands

```bash
# Training
python train.py                                          # defaults: 80 epochs, bs=16, lr=2e-4
python train.py --epochs 100 --bs 8 --lr 1e-4           # custom params
python train.py --resume checkpoints/checkpoint_epoch_40.pth  # resume from checkpoint

# Monitoring
tensorboard --logdir runs/

# Evaluation
python evaluate.py --checkpoint checkpoints/best_model.pth

# Figure generation
python generate_figures.py --flows vortex channel

# Environment setup (remote GPU machines)
bash setup_remote.sh
```

## Architecture

### Data Pipeline (`data_generator.py`)
Synthetic particle image pairs are generated on-the-fly. Five flow types: vortex, uniform, shear, channel, multi_vortex. Particles are rendered as Gaussian blobs using a vectorised separable-Gaussian trick (no per-particle loops). Ground truth flow displaces particles from frame 1 to frame 2.

### Model (`model.py` — `FlowEstimatorRAFT`)
RAFT-style architecture operating at 1/8 resolution:
1. **FeatureEncoder** (shared) — encodes each frame to [B, 256, H/8, W/8]
2. **ContextEncoder** (frame 1 only) — produces GRU hidden state + context features
3. **CorrBlock** — all-pairs correlation volume with 4-level pyramid, local lookup (radius 4)
4. **UpdateBlock** (iterated 12×) — MotionEncoder + ConvGRU + FlowHead predicts Δflow residuals
5. **Convex upsampling** — learned 8× upsampling via softmax-weighted 3×3 patches

The model returns a list of flow predictions (one per iteration) for sequence loss weighting.

### Loss Functions (`losses.py` — `SelfSupervisedTrackingLoss`)
All losses are self-supervised (no ground-truth flow needed):
- **Photometric**: backward-warp frame 1 by predicted flow → compare to frame 2. Blend of Charbonnier (50%) and SSIM (50%).
- **Divergence**: incompressibility constraint, penalises ∂u/∂x + ∂v/∂y via central differences.
- **Smoothness**: total-variation spatial regularization on flow gradients.

Losses are summed over all 12 iterations with exponential decay γ=0.8 (latest iteration weighted highest).

**Critical**: backward warp uses `grid - flow` (not `grid + flow`). This was a past bug.

### Training Loop (`train.py`)
Uses AdamW + cosine annealing, AMP (mixed precision), gradient clipping (max norm 1.0). Validation computes EPE against synthetic ground truth as a supervised monitoring metric. Best model saved to `checkpoints/best_model.pth`; periodic snapshots every 10 epochs.

### Visualization (`visualizer.py`)
TensorBoard logging of scalar metrics and image grids: HSV flow maps, quiver plots, warp quality overlays, error/divergence heatmaps. Logs every 2 epochs.

## Key Design Decisions

- Images are single-channel (grayscale particle images), 256×256 fixed size
- All data is synthetic — no external datasets needed
- EPE (end-point error) is computed for monitoring but never used as a training signal
- Loss weights: λ_photo=1.0, λ_div=0.3, λ_smooth=0.15
