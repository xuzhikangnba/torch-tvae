# Reproduction audit of the accepted Pro-t-VAE manuscript

Audit target: the author's `paper/20261006/neurips_2026.tex`, SHA256
`2a77a16ff6252f272e6fb9e3d07f7c9a447c60ae5dce6ee0cfe06aa8aab9115d`.

**Status: package functionality and eight fresh 100-epoch runs completed.**
The [fresh results](fresh_results.md) report same-condition numerical
comparisons and their limits. Exact end-to-end reproduction of the whole paper
is not verified. Machine-readable evidence is in `experiments/audit_results/`.

Update: [the recovered configuration](recovered_configuration.md) now ties
40 historical stdout logs to all 4,000 corresponding epoch losses in the
paper's source CSVs. It directly confirms AdamW/0.001/32/100 epochs and the
epoch-20 GMM overwrite. Fresh experiments now validate the recorded scope below.

## Fresh MRI training completed

Six complete 100-epoch runs cover all three MRI views at gamma=0 and 0.12.
They use the installed 0.3.1 model classes and the recovered configuration,
with float64 GMM covariance estimation for numerical stability. At gamma=0.12:

| Three-view mean | Fresh run | Paper | Difference |
|---|---:|---:|---:|
| Refit Silhouette | 0.313567 | 0.350933 | -0.037367 |
| Refit Davies–Bouldin | 1.205067 | 1.099333 | +0.105733 |
| Self-clean atlas PSNR (dB) | 20.071248 | 20.013365 | +0.057884 |

Silhouette and DB fall within the exploratory bands specified before reading
fresh results (0.05 absolute and 15% relative respectively). PSNR had no
predeclared acceptance threshold; the actual difference is shown. All three
fresh noisy views also have finite native Silhouette scores, indicating that
their native assignments retain two occupied clusters. Individual views vary
more than these averages; see every value, selected epoch, source hash, and
per-cluster PSNR in `experiments/audit_results/fresh_mri_gamma012.json`.

These results validate this condition and recipe; they do not establish the
seven-noise average, every baseline, or confidence intervals across seeds.
MNIST/Fashion-MNIST runs with the same historical schedule have also completed;
their metrics, including the larger positive Fashion-MNIST differences, are
reported in [fresh_results.md](fresh_results.md).

## Completed checks

- Installed non-editable 0.3.1 wheel, including the original 28x28 and 128x128
  multi-component models. All 21 tests pass, including independent Monte Carlo
  joint-KL checks, gradients, Gaussian limit, training and checkpoint round trips.
  A separate CPU check verifies bitwise-identical model tensors for four main
  updates run continuously or resumed as three plus one.
- Rechecked all 826 archived run logs against selected-epoch summaries.
- Recovered native assignment ACC of 0.798285714 on MNIST and 0.555 on
  Fashion-MNIST over seven noisy conditions. The earlier audit incorrectly
  compared these manuscript values with **refit** accuracy; they are **stage**
  accuracy at epochs selected by maximum post-warmup refit accuracy. The
  reported stage ARI/NMI averages are 0.695429/0.753429 and 0.409286/0.570286.
- Verified all 91 numerical entries in Table 1: 28 Silhouette, 28 DB, 21 PSNR,
  and 14 collapse-count/rate entries. Recomputed 126 per-cluster atlas PSNRs
  directly from the PNG images used by that table.
- Ran fresh full-split inference/GMM refitting for the five available ours
  checkpoints at gamma=0.12 (MNIST, Fashion-MNIST, and three MRI views).
- Completed training functionality runs for all five dataset/view entry points:
  two epochs on 640 MNIST images, one epoch on 640 Fashion-MNIST images, and
  one epoch on each 253-image MRI view. The initial MNIST check
  found a removed Matplotlib `cm.get_cmap` call; the rendering-only compatibility
  fix was applied and all training/checkpoint/plot pipelines then completed.
  These short runs are not benchmark reproduction or convergence tests.

## Unresolved differences

| Topic | Accepted manuscript | Available experiment source/artifacts |
|---|---|---|
| Optimizer | Adam | AdamW, weight decay 0.0001 |
| Learning rates | 0.002 mixture; 0.001 autoencoder | 0.001 for all parameters |
| Schedule | 50 pretraining epochs, then 500 iterations | 100 epochs, no separate reconstruction pretraining |
| Retained checkpoint | Epoch used by the displayed result | Existing `best_vae.pth` often selects another epoch |
| Evaluation corruption | Cauchy noise | Resampled on every dataset access; historic evaluation RNG state absent |

The basic `TVAE` intentionally has a different backbone and loss normalization;
the reference models preserve the archived implementation exactly. Replacing the
optimizer/schedule merely to match the manuscript wording would create a new
experiment; it would not establish the provenance of old results.

At gamma=0.12, decoding the existing checkpoint's prior means matches these
archived image epochs (mean pixel differences below 0.006/255):

| Dataset/view | Checkpoint image epoch | Table-selected epoch |
|---|---:|---:|
| MNIST | 92 | 57 |
| Fashion-MNIST | 57 | 26 |
| MRI coronal | 7 | 60 |
| MRI axial | 2 | 85 |
| MRI sagittal | 9 | 94 |

All 40 checkpoint epoch estimates in the ours eight-condition grid disagree
with the table-selected epochs; the five gamma=0.12 examples above are also
independently confirmed against decoded prototype images. Fresh inference
therefore evaluates different retained model states and noise realizations.
It cannot certify the table's selected-epoch metrics. In particular the three
early MRI checkpoints have collapsed native assignments; refitting a GMM is a
different evaluation and must not be presented as native model separation.

No complete noise-grid reproduction, runtime benchmark, or five-seed confidence
interval replication was performed in this audit. Fresh seed-42 runs cover the
gamma=0.12 comparison and clean MRI reference conditions described above.
Exact checkpoint replay needs the selected states and original data/noise/RNG
protocol. New from-scratch benchmark runs must be labeled as new experiments.

## PSNR convention

Historical PSNR compares each method with its **own clean-run atlas**, using
one-to-one Hungarian matching and peak value 1. The original montage-splitting
code produces 128x127 cells from 128x128 prototype images, trimming opposite
edge columns of the two cells. We retain that crop only to reproduce the
existing table. Full-image PSNR would require consistent re-evaluation.

Table 1 averages the two cluster PSNR values, then averages valid anatomical
views; it drops a view if its refit Silhouette is undefined. It does not compute
PSNR on a concatenated grid. This measures consistency under corruption, not
fidelity to a shared patient-specific ground-truth target or prototype diversity.

## Before claiming full paper reproducibility

Recover the selected-epoch weights and the final training/data protocol, or
perform and document fresh experiments with a reconciled protocol. Retain
epoch, optimizer settings, data split indices, random states, and the fitted
evaluation GMM. Update the public report with the actual comparisons; do not
replace missing measurements with archived numbers.
