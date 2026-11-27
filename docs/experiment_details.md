# Experiment details and protocol notes

Technical notes moved from the top-level and experiment READMEs. Read
[reproduction.md](reproduction.md) and [fresh_results.md](fresh_results.md)
for the audit and fresh-run comparisons. Basic commands are in
[experiments/README.md](../experiments/README.md).

## Reproduce the recovered paper workflow

The logs supporting the reported metrics use **AdamW, lr=0.001 for all
parameters, batch size 32, 100 epochs, and a train-GMM prior initialization at
epoch 20**. There is no separate reconstruction-only pretraining phase in this
workflow. The final NeurIPS version states the same optimizer, learning rate,
weight decay, batch size and epoch count; it does not describe the epoch-20 GMM
initialization, the split, or the checkpoint-selection rule. Use the portable launcher from a source checkout:

```bash
python -m pip install '.[experiments]'
python experiments/run_reference.py --dataset mnist --data-root ./data \
  --output ./outputs/mnist-g012 --gamma 0.12 --epochs 100 --device cuda:0 \
  --gmm-dtype float64 --train-silhouette-samples 2000
```

Change the dataset to `fashion_mnist`, `oasis_coronal`, `oasis_axial`, or
`oasis_sagittal`; MRI needs the prepared data described in
[experiments](../experiments/README.md). The single-component `TVAE` in
[usage.md](usage.md) is a separate model, not this clustering benchmark.

See the [source evidence](recovered_configuration.md) for the exact split,
loss normalization, checkpoint-selection rule, and distinction between native
and GMM-refit metrics. A diagnostic implementing an earlier draft's training
description is described next; do not mix those two protocols.

## Diagnostic of an earlier draft's two-stage schedule

The installed `tvae-paper-train` command uses the reference ResNet models, Adam
with separate autoencoder/mixture learning rates (0.001/0.002), batch size 32,
50 reconstruction-only pretraining epochs, and then 500 minibatch updates.
This is the literal interpretation of the earlier draft's “500 iterations.”
`--main-epochs 450` or `--main-epochs 500` records the alternative durations
mentioned elsewhere in that draft. The final NeurIPS version no longer
describes this schedule. Tests of these interpretations did not
consistently match the paper, including native axial-MRI collapse. See
[the fresh-run comparison](fresh_results.md); use the recovered workflow
above for the reported reference experiment.

```bash
python -m pip install '.[experiments]'
tvae-paper-train --dataset mnist --data-root ./data --output ./outputs/mnist-paper \
  --device cuda:0 --gamma 0.12
```

Change `--dataset` to `fashion_mnist` or an MRI view (`oasis_coronal`,
`oasis_axial`, `oasis_sagittal`). MRI needs the prepared NPZ files described in
[experiments](../experiments/README.md). The command reports both train and test
metrics, separating native assignments from a GMM fitted only on train embeddings.
Digits/clothing use the official 60,000/10,000 split by default; MRI uses a
seeded 80/20 subject split. `--split seeded_80_20` explicitly selects the archived
digit/clothing split for a separate comparison. Evaluation noise is held fixed.

Both reconstruction and prior loss use a per-observation mean with KL weight 1,
matching the stated ELBO. The archived trainer instead mixes a batch sum with
5 times a batch mean; its effective per-observation KL weight is 5/batch_size.
The run manifest explicitly records these differences and other choices not
specified in the main text. Fresh runs retain local checkpoints and fitted-GMM
parameters for audit/resumption; **data and weights are not uploaded**.

See [the fresh-run protocol](fresh_reproduction.md) for comparison criteria
and the complete recorded configuration.

## Reference launcher notes

The `experiments/reference/` tree contains the original v4.1 training code for MNIST,
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

These are the archived implementation settings, which the final NeurIPS
version also describes.

## Data notes

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

## Training objective and checkpoint selection

The archived training objective is summed batch reconstruction plus 5 times
mean prior penalty. AdamW uses lr=0.001, weight_decay=0.0001, batch_size=32.
At epoch 20 the prior is overwritten from a train-embedding GMM; df continues
learning. Labels are used for evaluation, not gradient training or GMM fitting.

The launcher saves the legacy reconstruction-selected `best_vae.pth` and adds
`best_refit_vae.pth`, each with an epoch/selection metadata JSON. The latter is
needed to retain the epoch used by the historical MNIST/Fashion table. This
selection uses held-out labels and should be disclosed in benchmark comparisons.

## Comparing with the paper

`experiments/summarize_reference.py` accepts `--run` repeatedly for several
completed runs. For MRI, repeat `--clean` with a
separately trained gamma=0 run for each view, using the same seed and settings.
The script selects epochs using the historical rule, computes per-cluster PSNR
with Hungarian matching, and aggregates MRI only when all three views are
present. It refuses incomplete runs. JSON reports retain raw values, differences,
configuration and source hashes; they do not substitute archived scores for
fresh measurements. The PSNR reference is each method's own clean prototype,
not a shared subject-level ground-truth image.

## Checkpoint replay

`experiments/replay_checkpoint.py` recomputes encoder representations, the model-native Student-t assignment
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
