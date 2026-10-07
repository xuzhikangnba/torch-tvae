# Configuration recovered from the reported experiment runs

The author authorized recovering the historical configuration when the current
manuscript description did not reproduce the metrics. This report separates
directly logged evidence from settings inferred from the available source.

## Logs tied to the actual paper results

All **40 ours runs** (eight Cauchy conditions, MNIST, Fashion-MNIST, three MRI
views) have a matching stdout file in the experiment's `run_logs/` directory.
The dated output identifier matches the paper's underlying metrics CSV, and
**all 4,000 printed epoch training losses match those CSV rows**. Older files
under `robustness_eval/logs/` can describe earlier runs and were not substituted.
The machine-readable evidence, source hashes, line numbers, and header excerpts
are in `experiments/audit_results/historical_configuration.json`.

The matched stdout directly establishes:

| Setting | Historical value |
|---|---|
| Optimizer | AdamW |
| Printed learning rate | 0.001 |
| Batch size | 32 |
| Training duration | 100 epochs |
| Initial degrees of freedom | 5 |
| Printed analytic prior weight | 5 |
| Prior overwrite | Train-embedding GMM at epoch 20 |

For example, MNIST gamma=0.12's stdout line 373 reports epoch 57 native ACC
0.832 and refit ACC 0.830, exactly the selected paper row. Fashion-MNIST's
line 179 reports epoch 26 native ACC 0.525 and refit ACC 0.574. Coronal MRI's
line 311 identifies epoch 60, the reconstruction-selected paper row. The
selection is not necessarily the epoch retained by an old `best_vae.pth` file.

## Settings supported by the available source

The source associated with those directories specifies one optimizer parameter
group (all parameters at 0.001), weight decay 0.0001, seed 42, gradient clipping
at norm 5, no separate reconstruction pretraining, and initial random mixture
parameters followed by the epoch-20 GMM overwrite. The absence of pretraining is
supported by the entry-point code; it is not independently logged as a field.

The exact objective is **pixel-and-batch-summed BCE + 5 times batch-mean analytic
prior loss**. It is not per-sample BCE plus five times per-sample KL. At batch
size 32 its relative KL coefficient is 5/32. At a smaller last batch it is 5/B.
An alternative normalization must be recorded as a different experiment.

Latent dimensions are 8 (digits/clothing) and 32 (MRI). K=10/2. Train/test are
an 80/20 seeded split of the available 60,000 training images or 253 MRI subjects;
the standard digit/clothing 10,000-image test partition was not used by this
source. Cauchy corruption is clipped to [0,1] and resampled on each access.
GMMs use train embeddings, full covariance, 10 restarts and random_state=0.

Fashion-MNIST runs heavy evaluation at epoch 1, every tenth epoch, epoch 20,
the final epoch, and when post-warmup test reconstruction improves. Other
reference entry points evaluate every epoch. This affects available selection
epochs and RNG consumption; the portable reference launcher retains it.

The exact historical source revision was not snapshotted within every run.
Accordingly, these source-supported settings should not be described as
independently logged facts. Fresh reruns are needed to validate the recovered
recipe against reported metrics.

The old batch-launch script mentions a `py37_pt18` environment in a comment,
but the matched run headers do not record dependency versions. This is a clue,
not proof of the historical environment. The current machine still has Python
3.7 / PyTorch 1.8 and 1.12 environments; one-epoch diagnostic attempts encountered
a DataLoader worker crash and an NVRTC GPU-architecture error respectively.
They did not yield a valid numerical comparison. Fresh package validation uses
the versions recorded in `experiments/audit_results/fresh_environment.json`;
they are not presented as the historical dependency versions.

## Manuscript reconciliation

The current main-text optimizer, separate learning rates, and 50+500 schedule
do not describe the matched historical runs. The checklist also alternates
between 450 and 500 main epochs. Any released reproduction claim must identify
the configuration actually tested and correct those descriptions accordingly.
This report does not silently edit the accepted manuscript.

Evaluation also needs a precise definition: after excluding epochs 1–25,
MNIST/Fashion select maximum held-out **refit** accuracy and report native
ACC/ARI/NMI at that epoch; MRI selects minimum held-out reconstruction and
reports **refit** Silhouette/DB. Labels are not used in the training loss, but
the digit/clothing checkpoint selection does use held-out labels. Native and
refit metrics must remain separate in the released outputs.

The basic label-free single-component TVAE is a separate usable model and is
not claimed to reproduce this multi-component benchmark.
