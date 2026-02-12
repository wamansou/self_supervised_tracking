# Self-Supervised Particle Tracking

Dense displacement estimation for particle image pairs **without ground-truth labels**, using physics-informed self-supervised losses. The project evolved from a baseline U-Net to a RAFT-style iterative flow estimator, achieving **sub-pixel accuracy (0.31 px EPE)**.

---

## Results Summary

| | U-Net Baseline | RAFT (final) | Improvement |
|---|---|---|---|
| **Best EPE** | 0.87 px | **0.31 px** | **2.8×** |
| Divergence | 0.020 | **0.002** | 10× |
| Smoothness | 0.085 | **0.006** | 14× |
| Parameters | 7.8M | 7.1M | — |
| Relative Error | 10.9% | **3.9%** | — |

*(EPE = End-Point Error; max displacement = 8 px; 256×256 images)*

---

## Problem Statement

Given two consecutive frames of a particle-seeded flow (e.g. PIV — Particle Image Velocimetry), estimate the **dense displacement field** $(u, v)$ that maps every pixel in frame 1 to its corresponding location in frame 2.

Traditional cross-correlation methods (e.g. PIV interrogation windows) discretise the field into coarse patches. This project instead predicts a **per-pixel** flow field using a neural network, trained entirely without manual annotations.

---

## Neural Network Architecture

### V2: RAFT-Style Iterative Flow Estimator (current)

The production model (`FlowEstimatorRAFT`) uses **correlation volumes** and **iterative refinement** — the two key ideas from RAFT (Teed & Deng, ECCV 2020) adapted for particle tracking.

```
Input:  [B, 2, H, W]         ← two grayscale particle frames stacked channel-wise
Output: list of [B, 2, H, W] ← per-pixel displacement (u, v), one per refinement iteration
```

#### Pipeline

| Component | Purpose | Output |
|-----------|---------|--------|
| **Feature Encoder** | Shared CNN, extracts 1/8-res features per frame | `[B, 256, H/8, W/8]` per frame |
| **Context Encoder** | Frame-1 CNN, produces GRU initial state + context | `hidden [B,128,H/8,W/8]`, `ctx [B,128,H/8,W/8]` |
| **Correlation Volume** | All-pairs dot product between frame features + 4-level pyramid | `[B·H/8·W/8, 1, H/8, W/8]` × 4 levels |
| **Correlation Lookup** | Sample (2r+1)² local neighbourhood at current flow estimate | `[B, 4×81, H/8, W/8]` |
| **Motion Encoder** | Fuse correlation features + current flow | `[B, 128, H/8, W/8]` |
| **ConvGRU** | Recurrent update of hidden state | `[B, 128, H/8, W/8]` |
| **Flow Head** | Predict Δflow residual | `[B, 2, H/8, W/8]` |
| **Convex Upsampling** | Learned 8× upsampling (sharper than bilinear) | `[B, 2, H, W]` |

The model runs **12 refinement iterations**, progressively correcting the flow estimate. Each iteration detaches the flow gradient (RAFT-style) so only the current update is trained, preventing vanishing gradients through long chains.

#### Why correlation volumes matter

The U-Net had no explicit mechanism to *compare* features between frames — it had to learn matching implicitly from convolutions alone. The correlation volume provides a direct similarity lookup: "at my current flow estimate, how well do features match?" This makes the model's job fundamentally easier and more robust.

### V1: U-Net Baseline (archived in `model_unet.py`)

A 4-level encoder–decoder with skip connections. Single forward pass, no correlation, no iterative refinement. Served as the development baseline.

---

## Loss Function

The model is trained **self-supervised** — no ground-truth flow labels are used during training. Instead, three complementary loss terms provide the learning signal.

### Iterative Sequence Loss

For RAFT's iterative predictions, the loss applies exponentially increasing weights so later (more refined) iterations matter more:

$$\mathcal{L} = \sum_{i=1}^{N} \gamma^{N-i} \left[ \lambda_{\text{photo}} \cdot \mathcal{L}_{\text{photo}}^{(i)} + \lambda_{\text{div}} \cdot \mathcal{L}_{\text{div}}^{(i)} + \lambda_{\text{smooth}} \cdot \mathcal{L}_{\text{smooth}}^{(i)} \right]$$

with $\gamma = 0.8$ so the final iteration has weight 1.0 and the first has weight $0.8^{11} \approx 0.09$.

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
| $\gamma$                 | 0.8  | Iteration decay — prioritises final refinement |

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

| Parameter | U-Net (v1) | RAFT (v2) |
|-----------|-----------|-----------|
| Image size | 256 × 256 | 256 × 256 |
| Particles per image | 500–2000 | 500–2000 |
| Max displacement | 8 px | 8 px |
| Batch size | 64 | 16 |
| Optimiser | Adam | **AdamW** |
| Learning rate | 3e-4 | **2e-4** |
| Scheduler | Cosine Annealing (80 ep) | Cosine Annealing (80 ep) |
| Mixed precision | AMP | AMP |
| Gradient clipping | Max norm 1.0 | Max norm 1.0 |
| Refinement iters | — | **12** |

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

## Development Timeline & Bugs Fixed

### Iteration 1: U-Net baseline
- Initial U-Net with Charbonnier-only photometric loss
- **Problem:** EPE *rose* from 4.3 → 6.9 while photo loss fell — model matched wrong particles

### Iteration 2: SSIM loss + tuned weights
- Added SSIM loss ($\alpha=0.5$ blend with Charbonnier) to force structural matching
- Reduced over-regularization ($\lambda_{\text{smooth}}$: 0.1 → 0.02, $\lambda_{\text{div}}$: 0.5 → 0.1)
- **Problem:** EPE still diverging — predicted flow pointed *opposite* to ground truth

### Iteration 3: Warp direction bug fix
- **Root cause:** `grid_sample` performs backward sampling, requiring `grid - flow`, not `grid + flow`
- Model had been forced to learn $\text{pred} \approx -\text{gt}$ to minimize photometric loss
- After fix: EPE dropped from 4.3 → **0.87 px** with U-Net

### Iteration 4: RAFT architecture
- Replaced U-Net with RAFT-style model: correlation volumes + ConvGRU iterative refinement
- Added sequence loss with $\gamma=0.8$ exponential weighting
- Switched to AdamW optimiser
- **Result:** EPE **0.87 → 0.31 px** (2.8× improvement)

### Other fixes
- **TensorBoard titles cropped** — increased `tight_layout` padding and figure heights
- **GPU at 54% utilisation** — vectorised particle rendering (separable Gaussian matmul), persistent DataLoader workers, `cudnn.benchmark`, increased batch size

---

## Usage

```bash
# Train with defaults (optimised for RTX 4090)
python train.py

# Custom settings
python train.py --epochs 100 --bs 8 --lr 1e-4 --iters 16

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
├── model.py            # RAFT-style flow estimator (FlowEstimatorRAFT)
├── model_unet.py       # U-Net baseline (FlowEstimatorUNet, archived)
├── losses.py           # Self-supervised losses (photometric, SSIM, divergence, smoothness)
├── data_generator.py   # Synthetic particle image pair generator
├── train.py            # Training loop with AMP, logging, checkpointing
├── visualizer.py       # TensorBoard visualisation (flow maps, quiver, warp quality)
├── evaluate.py         # Evaluation utilities
├── checkpoints/        # Saved model weights
└── runs/               # TensorBoard log directory
```
