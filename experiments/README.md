# Reproducing the paper experiments

This folder contains the original training code (`reference/`), a launcher
that trains with the installed `tvae` package, and scripts that compare runs
with the paper's numbers. For protocol details and caveats, see
[docs/experiment_details.md](../docs/experiment_details.md).

## Setup

From the repository root:

```bash
python -m pip install ".[experiments]"
```

**Data**
- **MNIST / Fashion-MNIST** are downloaded into `--data-root` automatically.
- **OASIS MRI** needs [OASIS](https://sites.wustl.edu/oasisbrains/) access. Put
  the prepared `coronal.npz`, `axial.npz` or `sagittal.npz` files in
  `--data-root`. Each file holds an `images` array of shape `(253, 128, 128)`:
  float32 values in [0, 1]. The prepared data is not distributed with this
  repository.

## Train

```bash
python experiments/run_reference.py --dataset mnist --data-root ./data \
  --output ./outputs/mnist-g012 --gamma 0.12 --epochs 100 --device cuda:0
```

- `--dataset`: `mnist`, `fashion_mnist`, `oasis_coronal`, `oasis_axial` or `oasis_sagittal`
- `--gamma` (Cauchy noise scale): the paper grid is `0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14`
- `--output` must be a new directory. The launcher refuses to overwrite an existing run.

Quick check that training runs (not a benchmark):

```bash
python experiments/run_reference.py --dataset mnist --data-root ./data \
  --output ./outputs/smoke --gamma 0.12 --epochs 2 \
  --samples-per-class 64 --device cpu --workers 0
```

## Compare with the paper

```bash
python experiments/summarize_reference.py --run ./outputs/mnist-g012 \
  --output ./outputs/mnist-comparison.json
```

Pass `--run` more than once to summarize several runs. For MRI, also pass a
γ = 0 run of the same view with `--clean` (it is the reference for PSNR).
Results for all three views are averaged only when all three are present.

## Evaluate a checkpoint

```bash
python experiments/replay_checkpoint.py --dataset mnist --data-root ./data \
  --checkpoint ./outputs/mnist-g012/best_refit_vae.pth --gamma 0.12 \
  --output ./outputs/evaluation.json --device cpu
```

## Results from our reruns

[docs/fresh_results.md](../docs/fresh_results.md) compares eight fresh
100-epoch runs with the paper. Their per-epoch traces and audit records are in
[audit_results/](audit_results/).
