# Self-Supervised Particle Tracking

Dense displacement estimation for particle image pairs **without ground-truth labels**, using a U-Net trained with physics-informed self-supervised losses.

---

## Problem Statement

Given two consecutive frames of a particle-seeded flow (e.g. PIV — Particle Image Velocimetry), estimate the **dense displacement field** $(u, v)$ that maps every pixel in frame 1 to its corresponding location in frame 2.

Traditional cross-correlation methods (e.g. PIV interrogation windows) discretise the field into coarse patches. This project instead predicts a **per-pixel** flow field using a neural network, trained entirely without manual annotations.

---

## Neural Network Architecture

We use a **U-Net** (`FlowEstimatorUNet`) — an encoder–decoder with skip connections, well-suited for dense prediction tasks where both local detail and global context matter.

```
Input:  [B, 2, H, W]   ← two grayscale particle frames stacked channel-wise
Output: [B, 2, H, W]   ← per-pixel displacement (u, v) in pixels
```

### Structure

| Stage       | Layers                                  | Output Shape         |
|-------------|----------------------------------------|----------------------|
| **Encoder 1** | 2 × (Conv3×3 → BN → LeakyReLU)     | `[B, f, H, W]`      |
| **Encoder 2** | MaxPool → 2 × (Conv3×3 → BN → LReLU) | `[B, 2f, H/2, W/2]` |
| **Encoder 3** | MaxPool → 2 × (Conv3×3 → BN → LReLU) | `[B, 4f, H/4, W/4]` |
| **Encoder 4** | MaxPool → 2 × (Conv3×3 → BN → LReLU) | `[B, 8f, H/8, W/8]` |
| **Bottleneck** | MaxPool → 2 × (Conv3×3 → BN → LReLU) | `[B, 16f, H/16, W/16]` |
| **Decoder 4** | ConvTranspose2d ↑ + skip(enc4) → 2×Conv | `[B, 8f, H/8, W/8]` |
| **Decoder 3** | ConvTranspose2d ↑ + skip(enc3) → 2×Conv | `[B, 4f, H/4, W/4]` |
| **Decoder 2** | ConvTranspose2d ↑ + skip(enc2) → 2×Conv | `[B, 2f, H/2, W/2]` |
| **Decoder 1** | ConvTranspose2d ↑ + skip(enc1) → 2×Conv | `[B, f, H, W]`      |
| **Head**      | Conv1×1                                 | `[B, 2, H, W]`      |

With `base_features = 64` (`f = 64`), the deepest feature map has 1024 channels, giving the network sufficient capacity to capture complex flow patterns like vortices.

---

## Loss Function

The model is trained **self-supervised** — no ground-truth flow labels are used during training. Instead, three complementary loss terms provide the learning signal:

$$\mathcal{L} = \lambda_{\text{photo}} \cdot \mathcal{L}_{\text{photo}} + \lambda_{\text{div}} \cdot \mathcal{L}_{\text{div}} + \lambda_{\text{smooth}} \cdot \mathcal{L}_{\text{smooth}}$$

### 1. Photometric Loss ($\mathcal{L}_{\text{photo}}$)

The core self-supervised signal. If the predicted flow is correct, then warping frame 1 by the flow should reconstruct frame 2:

$$\mathcal{L}_{\text{photo}} = (1 - \alpha) \cdot \mathcal{L}_{\text{Charbonnier}} + \alpha \cdot \mathcal{L}_{\text{SSIM}}$$

- **Charbonnier** (robust L1): $\sqrt{(I_{\text{warped}} - I_2)^2 + \epsilon}$ — tolerant to small outliers
- **SSIM** (Structural Similarity): Compares local *structure* (luminance, contrast, correlation) rather than raw pixel values. Critical for particle images where many particles look identical — SSIM forces the model to match the right particle by considering neighbourhood context.

The blend weight $\alpha = 0.5$ balances pixel-level accuracy (Charbonnier) with structural robustness (SSIM).

**Warping** is done via differentiable bilinear sampling (`grid_sample`). The flow is *forward* (frame 1 → frame 2), so we use **backward warping**: at each output pixel $q$, we sample frame 1 at position $q - \text{flow}(q)$.

### 2. Divergence Loss ($\mathcal{L}_{\text{div}}$)

A **physics prior** for incompressible flow:

$$\mathcal{L}_{\text{div}} = \left\| \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} \right\|^2$$

For incompressible fluids, the divergence of the velocity field must be zero ($\nabla \cdot \mathbf{u} = 0$). This constraint prevents the model from predicting non-physical flows (e.g. sources/sinks) even though they might reduce photometric error. Computed with central finite differences on interior pixels.

### 3. Smoothness Loss ($\mathcal{L}_{\text{smooth}}$)

A spatial regulariser (total-variation style):

$$\mathcal{L}_{\text{smooth}} = \left\| \frac{\partial \mathbf{f}}{\partial x} \right\|^2 + \left\| \frac{\partial \mathbf{f}}{\partial y} \right\|^2$$

Penalises noisy or discontinuous flow fields. Without this, the model can "game" the photometric loss by predicting rough, spiky displacements that match individual particles but don't represent coherent physical motion.

### Loss Weights

| Weight | Value | Rationale |
|--------|-------|-----------|
| $\lambda_{\text{photo}}$ | 1.0  | Primary learning signal |
| $\lambda_{\text{div}}$   | 0.3  | Strong enough to enforce physics without dominating |
| $\lambda_{\text{smooth}}$ | 0.15 | Prevents noisy flow while allowing genuine gradients |

---

## Synthetic Data Generation

Training data is generated on-the-fly from parametric flow fields with known ground truth:

| Flow Type | Description |
|-----------|-------------|
| **Vortex** | Lamb–Oseen vortex with random centre, circulation, and core radius |
| **Uniform** | Constant translation at random angle and magnitude |
| **Shear** | Linear velocity profile (horizontal or vertical) |
| **Channel** | Parabolic (Poiseuille) profile simulating pipe flow |
| **Multi-vortex** | Superposition of 2–5 vortices with random parameters |

Particles are rendered as Gaussian blobs (diameter 2–4 px) placed at random positions, with Gaussian sensor noise ($\sigma = 0.05$). The rendering uses a vectorised separable-Gaussian trick (`gy @ gx.T`) instead of per-particle loops for speed.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Image size | 256 × 256 |
| Particles per image | 500–2000 |
| Max displacement | 8 px |
| Batch size | 64 |
| Optimiser | Adam (lr=3e-4, weight_decay=1e-5) |
| Scheduler | Cosine Annealing (80 epochs, η_min = lr × 0.01) |
| Mixed precision | AMP (FP16 forward, FP32 gradients) |
| Gradient clipping | Max norm 1.0 |
| `torch.compile` | Enabled (PyTorch 2.x graph mode) |
| `cudnn.benchmark` | Enabled (fixed input sizes) |

---

## Metrics & Monitoring

| Metric | Description |
|--------|-------------|
| **EPE** (End-Point Error) | $\text{mean}\sqrt{(u_{\text{pred}} - u_{\text{gt}})^2 + (v_{\text{pred}} - v_{\text{gt}})^2}$ — pixel distance between predicted and true displacement. The primary accuracy metric. |
| **Photometric** | Reconstruction quality of warped frame vs. target frame |
| **Divergence** | Flow compressibility — should approach 0 |
| **Smoothness** | Spatial roughness of predicted flow |

TensorBoard visualisations logged every 2 epochs:

1. **Frame pairs** — input particle images
2. **HSV flow maps** — colour-coded flow (hue = direction, brightness = magnitude), predicted vs. ground truth
3. **Quiver plots** — arrow overlays showing flow direction & magnitude
4. **Warp quality** — warped frame 1 vs. frame 2 and their difference
5. **Error & divergence maps** — spatial EPE heatmap + divergence field

---

## Key Bugs Found & Fixed

1. **Warp direction reversed** — `grid_sample` performs backward sampling, so the grid must use `grid - flow` (not `+ flow`). The original sign error caused the model to learn the *negative* of the true flow, explaining why predicted arrows pointed opposite to ground truth and EPE grew as photometric loss decreased.

2. **TensorBoard titles cropped** — `tight_layout(pad=0.5)` was too tight; matplotlib titles were clipped in the rendered buffer. Fixed by increasing pad to 1.5 and adding 0.5″ to figure heights.

3. **GPU underutilisation** — The particle renderer used a Python for-loop over 500–2000 particles per image. Replaced with a vectorised separable-Gaussian matmul, added persistent DataLoader workers with prefetching, `torch.compile`, and `cudnn.benchmark`.

---

## Usage

```bash
# Train with defaults (optimised for RTX 4090)
python train.py

# Custom settings
python train.py --epochs 100 --bs 32 --lr 1e-4 --l-smooth 0.2

# Resume from checkpoint
python train.py --resume checkpoints/checkpoint_epoch_40.pth

# Disable mixed precision
python train.py --no-amp

# Monitor training
tensorboard --logdir runs/
```

---

## Project Structure

```
├── model.py            # U-Net architecture (FlowEstimatorUNet)
├── losses.py           # Self-supervised losses (photometric, SSIM, divergence, smoothness)
├── data_generator.py   # Synthetic particle image pair generator
├── train.py            # Training loop with AMP, logging, checkpointing
├── visualizer.py       # TensorBoard visualisation (flow maps, quiver, warp quality)
├── evaluate.py         # Evaluation utilities
├── checkpoints/        # Saved model weights
└── runs/               # TensorBoard log directory
```
