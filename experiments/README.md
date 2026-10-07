# Paper reference experiments

The `reference/` tree contains the original v4.1 training code for MNIST,
Fashion-MNIST, and the coronal/axial/sagittal MRI views. `source_manifest.json`
records each original and staged source hash. The model source is unchanged;
the only source adaptation is a Matplotlib colormap compatibility fix.
The portable launcher chooses data, output, device, and epoch settings without
overwriting the original experiment directories. It uses the installed wheel's
`TVaDE28`/`TVaDE128`, so training and checkpoint evaluation exercise the package.

The portable launcher estimates GMM covariances in float64 by default. A fresh
sagittal-MRI run exposed a float32 Cholesky failure with current scikit-learn;
upcasting embeddings preserves their values and the covariance model while
improving numerical accuracy. Neural-network training remains float32. The
run manifest records this choice; `--gmm-dtype original` selects the archived
input dtype explicitly. The archived source files themselves are unchanged.

For large digit/clothing runs, `--train-silhouette-samples 2000` can reduce the
quadratic cost of the **training-only** Silhouette diagnostic. Test metrics,
train ACC/ARI/NMI/DB/cluster counts, GMM fitting, checkpoint selection, and all
gradient updates still use the complete respective split. Sampling uses an
isolated fixed RNG and is recorded in the run manifest and metric column;
the default 0 retains the original full training-set diagnostic.

Read [the reproduction report](../docs/reproduction.md) first. These are the
archived implementation settings, not a claim that the manuscript's different
Adam/pretraining description has been reproduced.

## Install and provide data

From the repository root:

```bash
python -m pip install '.[experiments]'
```

MNIST and Fashion-MNIST are downloaded by torchvision if not already present
in `--data-root`. The archived experiment uses the 60,000 training images and
an 80/20 random split (seed 42); it does not use the official 10,000 test set.
The `12800` samples-per-class setting is a cap, not actual balanced oversampling.

For MRI, obtain the appropriate OASIS access and supply the already prepared
`coronal.npz`, `axial.npz`, or `sagittal.npz` under `--data-root`. Each archive
contains `images`, shape `(253,128,128)`, float32 in `[0,1]`. Optional identifiers
are not required for the metrics. The prepared data is not distributed here.
The exact historical preprocessing/split is needed for quantitative comparison;
this repository does not claim that arbitrary MRI slices reproduce Table 1.

## Run training

```bash
python experiments/run_reference.py --dataset mnist --data-root ./data \
  --output ./outputs/mnist-g012 --gamma 0.12 --epochs 100 --device cuda:0
```

Change `--dataset` to `fashion_mnist`, `oasis_coronal`, `oasis_axial`, or
`oasis_sagittal`. Use `--device cpu` for CPU. Every output directory must be new;
the launcher refuses to overwrite a run. Repeat with gamma values
`0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14` for the complete condition grid.

The archived training objective is summed batch reconstruction plus 5 times
mean prior penalty. AdamW uses lr=0.001, weight_decay=0.0001, batch_size=32.
At epoch 20 the prior is overwritten from a train-embedding GMM; df continues
learning. Labels are used for evaluation, not gradient training or GMM fitting.

The launcher saves the legacy reconstruction-selected `best_vae.pth` and adds
`best_refit_vae.pth`, each with an epoch/selection metadata JSON. The latter is
needed to retain the epoch used by the historical MNIST/Fashion table. This
selection uses held-out labels and should be disclosed in benchmark comparisons.

After a full 100-epoch run, compare it with the same-condition archived target:

```bash
python experiments/summarize_reference.py --run ./outputs/mnist-g012 \
  --output ./outputs/mnist-comparison.json
```

Repeat `--run` for several completed runs. For MRI, repeat `--clean` with a
separately trained gamma=0 run for each view, using the same seed and settings.
The script selects epochs using the historical rule, computes per-cluster PSNR
with Hungarian matching, and aggregates MRI only when all three views are
present. It refuses incomplete runs. JSON reports retain raw values, differences,
configuration and source hashes; they do not substitute archived scores for
fresh measurements. The PSNR reference is each method's own clean prototype,
not a shared subject-level ground-truth image.

A small training functionality check, **not a paper benchmark**:

```bash
python experiments/run_reference.py --dataset mnist --data-root ./data \
  --output ./outputs/smoke --gamma 0.12 --epochs 2 \
  --samples-per-class 64 --device cpu --workers 0
```

## Evaluate an available checkpoint

```bash
python experiments/replay_checkpoint.py --dataset mnist --data-root ./data \
  --checkpoint /path/to/best_vae.pth --gamma 0.12 \
  --output ./outputs/evaluation.json --device cpu
```

This recomputes encoder representations, the model-native Student-t assignment
(`stage`) and a separately fitted GMM assignment (`refit`). The report identifies
the checkpoint by SHA256. The seed fixes this evaluation's split and noise;
it does not recreate the original run's unstored RNG state. Old checkpoints
have no epoch metadata; counter-based epoch estimates are labeled as estimates.

## Audit the archived paper artifacts

```bash
python experiments/audit_artifacts.py --source-root /path/to/original-archive \
  --paper /path/to/accepted/neurips_2026.tex --output ./outputs/artifact-audit
```

This requires the original `exp*` run directories and `paper_results/tables`,
as identified in `source_manifest.json` and the report. It is not runnable from
this source-only checkout without those separate artifacts. Logs are reselected
using the documented epoch rule, then stage/refit metrics and historical PSNR
are checked. Archived aggregation is explicitly distinct from new training.
