# Flow Matching for Cosmological Halo Distribution Generation

A generative model for cosmological dark matter halo distributions using **E(3)-equivariant graph neural networks (EGNN)** and **flow matching**, conditioned on cosmological parameters with **classifier-free guidance**.

---

## Overview

This project learns to generate realistic 3D dark matter halo distributions inside a periodic cosmological simulation box. Given a set of cosmological parameters (Ω_m, σ_8, etc.), the model samples halo positions that are statistically consistent with N-body simulations. The architecture combines:

- **Flow Matching** — a simulation-free generative framework that learns a vector field transporting a prior distribution (uniform or Gaussian) to the data distribution.
- **EGNN** — an E(3)-equivariant graph neural network that predicts the vector field while respecting the 3D symmetries of the problem (rotations, translations, reflections).
- **Periodic Boundary Conditions (PBC)** — all distance computations and coordinate wrapping obey the minimum image convention of a periodic cosmological box.
- **Classifier-Free Guidance (CFG)** — cosmological parameters condition the generation, with 15% random drop during training to enable unconditional + conditional sampling at inference.

---

## Architecture

### Flow Matching (`fm.py`)

The `FlowMatching` module defines the generative process:

1. **Prior sampling**: halo positions are initialized from a uniform or wrapped Gaussian distribution over the periodic box.
2. **Probability path**: for each training example, a noisy interpolant `x_t` is constructed between the prior `x_0` and data `x_1` at a random time `t ∈ [0, 1]`.
3. **Target vector field**: the conditional vector field `u_t(x | x_0, x_1)` is computed analytically.
4. **MSE loss**: the EGNN is trained to match the target vector field.

Two flow matching variants are supported:
- **ICFM** (Independent Coupling Flow Matching) — straight-line interpolant between prior and data.
- **FM** — stochastic interpolant with noise schedule.

### EGNN (`egnn.py`)

The velocity network (`vnet`) is an EGNN stacking multiple **EGCL** (Equivariant Graph Convolution Layer) blocks:

- Node features are halo masses (log₁₀-scaled to [0, 1]).
- A **KNN graph** is constructed at each step using PBC-aware k-nearest neighbours (k=32).
- **Sinusoidal time embedding** encodes the flow time `t` and is broadcast to all nodes.
- **Cosmological parameters** (θ) are embedded via a sinusoidal embedding and concatenated with the time embedding to form a global conditioning vector.
- Each EGCL layer updates node features and accumulates an equivariant velocity contribution.

### EGCL (`egcl.py`)

Each equivariant convolution layer performs:
1. **Edge model**: computes edge features from node feature differences, squared distances, and the global conditioning (time + θ).
2. **Attention**: optional softmax attention weighting over incoming edges.
3. **Coordinate model**: equivariant velocity aggregation using `coord_diff × MLP(edge_feat)`.
4. **Node model**: aggregates edge features, applies residual connection and layer norm.

All distance calculations use the **minimum image convention** to correctly handle the periodic box.

---

## Model Variants

| Variant | Description |
|---|---|
| **Base Model** | EGNN conditioned on time + θ via global attribute concatenation |
| **T5000-FiLM** | Cosmological parameters injected via **FiLM** (Feature-wise Linear Modulation) inside each EGCL layer (γ, β affine transform on node features) |
| **T5000-MV** | Additional halo **velocity** features as node input |
| **T5000-OVF** | **Optimal Transport** prior coupling + velocity features |
| **T5000-OVFM** | OT coupling + velocity features + FiLM conditioning |
| **T256** | Same architecture trained on the T256 simulation dataset |
| **T256-SUBBOX** | T256 variant trained on sub-box patches |

---

## Data

Each simulation snapshot contains N halos with features `[x, y, z, vx, vy, vz, mass]`. Positions are normalized to `[0, 1]` by dividing by the box size (1000 Mpc/h). Each simulation is associated with a vector of cosmological parameters θ (Ω_m, Ω_b, h, n_s, σ_8).

**Data augmentation** at training time:
- Random rotations from the 48-element hyperoctahedral group (sign flips × axis permutations).
- Random uniform translations (with periodic wrapping).

**Optional prior coupling** (OVF/OVFM variants):
- Optimal Transport coupling using the Hungarian algorithm to minimize transport cost between the uniform prior and the data positions under the PBC distance metric. Pre-computed and cached as `train_x0_ot.npy`.

---

## Training

Training uses **PyTorch DDP** for multi-GPU distributed training. Hyperparameter search uses **Ray Tune**.

Key hyperparameters (default):

| Parameter | Value |
|---|---|
| k (KNN neighbours) | 32 |
| Hidden features | 128 |
| Latent features | 16 |
| EGCL layers | 4 |
| Time embedding dim | 64 |
| θ embedding dim | 32 per parameter |
| Batch size | 16 |
| Peak LR | 5e-4 |
| LR schedule | Cosine with 100-epoch warmup |
| Gradient clipping | 0.5 |

```bash
torchrun --nproc_per_node=<N_GPUS> src/train_ddp.py --config Configs/train_configs.yaml
```

---

## Inference

At inference, a prior sample is integrated forward in time using Euler steps over the learned vector field with **classifier-free guidance**:

```
v_guided = v_uncond + W * (v_cond - v_uncond)
```

where `W` is the guidance weight. The trajectory is integrated for `T` steps (configurable), with the KNN graph rebuilt at each step and PBC wrapping applied after each update.

```bash
python src/infer.py --config Configs/infer_configs.yaml
```

---

## Requirements

- Python 3.10+
- PyTorch
- PyTorch Geometric (`torch_geometric`)
- `torch_scatter`
- `scipy` (for OT coupling pre-computation)
- `ray[tune]` (for hyperparameter search)
- `wandb` (for experiment logging)

---

## Repository Structure

```
├── T5000/
│   ├── T5000 Base Model/     # Baseline EGNN + flow matching
│   ├── T5000-FiLM/           # FiLM-conditioned variant
│   ├── T5000-MV/             # Multi-velocity variant
│   ├── T5000-OVF/            # OT + velocity features
│   └── T5000-OVFM/           # OT + velocity + FiLM
├── T256/
│   ├── T256/                 # T256 simulation dataset
│   └── T256-SUBBOX/          # Sub-box variant
└── Boids/                    # Boids toy problem & practicals
```

Each variant contains:
- `src/` — model source code (`egnn.py`, `egcl.py`, `fm.py`, `dataset.py`, `train_ddp.py`, `infer.py`)
- `Configs/` — YAML configuration files for training and inference
- `Notebooks/` — analysis and evaluation notebooks
