# FlowEGNN: Flow Matching for Cosmological Halo Generation

A generative model for 3D dark matter halo distributions using **flow matching** and **E(3)-equivariant graph neural networks (EGNN)**, conditioned on cosmological parameters.

---

## What is this?

This project is part of a thesis on simulation-based inference and emulation of large-scale structure. The goal is to train a generative model that can produce realistic dark matter halo catalogues for any given set of cosmological parameters (Ω_m, Ω_b, h, n_s, σ_8), without running expensive N-body simulations.

Training data comes from the [Quijote simulation suite](https://quijote-simulations.readthedocs.io/en/latest/). The model is evaluated by comparing generated halo distributions against the ground truth using the two-point correlation function (2PCF).

**The core idea**: flow matching learns a continuous transformation from a simple prior (uniform random positions in the box) to the target halo distribution, guided by an EGNN that is aware of the 3D spatial structure of the halos and respects the symmetries of the problem (rotations, reflections, translations). Conditioning on cosmological parameters is done via **classifier-free guidance**.

---

## Model variants

The repository contains two main models, targeting different representations of the halo catalogue:

### T256 — Sub-box model

Trained on subsampled **hypercube sub-regions** of the simulation box, each containing roughly 256 halos. This decomposition makes the problem more tractable and allows the model to focus on local structure.

### T5000 — Full catalogue model

Trained on the complete **top-5000 heaviest halo catalogues** directly from Quijote, representing each simulation as a single point cloud of the 5000 most massive halos in the full box. This captures global structure but is a harder generation problem.

---

## Training & Inference

```bash
# Multi-GPU training (SLURM)
sbatch training_script.sh

# Single-GPU training
python src/train_ddp.py --config Configs/train_configs.yaml

# Hyperparameter search
bash tune_script.sh

# Generate samples
python src/infer.py --config Configs/infer_configs.yaml
```

All settings (data paths, model size, training schedule, inference parameters) are controlled through YAML config files in `Configs/`.

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Repository Structure

```
FlowEGNN/
├── T256/
│   ├── src/
│   │   ├── egnn.py             # EGNN velocity network
│   │   ├── egcl.py             # Equivariant graph convolution layer
│   │   ├── fm.py               # Flow matching module
│   │   ├── dataset.py          # Data loading & augmentation
│   │   ├── train_ddp.py        # Distributed training
│   │   ├── train_tune.py       # Ray Tune hyperparameter search
│   │   ├── infer.py            # Inference / generation
│   │   ├── validation.py       # 2PCF evaluation
│   │   ├── mlp_baseline.py     # Non-equivariant MLP baseline
│   │   ├── prior.py            # Prior distributions
│   │   ├── pbc_config.py       # Periodic boundary utilities
│   │   ├── mlp.py              # Generic MLP
│   │   └── utils/              # Config, embeddings, graph, scaling, logging
│   ├── Configs/
│   │   ├── train_configs.yaml
│   │   ├── tuned_configs.yaml
│   │   ├── overfit_configs.yaml
│   │   └── infer_configs.yaml
│   ├── Notebooks/              # Analysis & evaluation notebooks
│   ├── training_script.sh      # SLURM multi-GPU job
│   ├── training_script_single.sh
│   ├── tune_script.sh
│   └── infer_script.sh
├── T5000/
│   ├── src/                    # Same structure as T256/src/
│   ├── Configs/
│   │   ├── train_configs.yaml
│   │   ├── tuned_configs.yaml
│   │   ├── overfit_configs.yaml
│   │   └── infer_configs.yaml
│   ├── Notebooks/              # 2PCF, 3PCF, power spectrum, bispectrum analysis
│   ├── training_script.sh
│   ├── training_script_single.sh
│   ├── tune_script.sh
│   └── infer_script.sh
└── Data/                       # Simulation data (not tracked in git)
```
