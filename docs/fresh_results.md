# Fresh training results for the public release

Eight runs completed all 100 epochs using the installed 0.3.1 model classes:
MNIST and Fashion-MNIST at Cauchy gamma=0.12, and all three MRI views at
gamma=0 and 0.12. No historical weights initialized these models. No weights,
patient data, or subject images are included in the release.

The recovered historical recipe uses AdamW, lr=0.001 for all parameters,
weight decay 0.0001, batch size 32, no separate pretraining, and a training-GMM
prior overwrite at epoch 20. The exact batch loss and splits are described in
[recovered_configuration.md](recovered_configuration.md). Seed is 42.

## Same-condition comparisons

All rows below compare **gamma=0.12**, not the seven-noise average.

| Dataset | Metric | Fresh | Paper | Fresh minus paper |
|---|---|---:|---:|---:|
| MNIST | Native ACC | 0.817 | 0.832 | -0.015 |
| MNIST | Native ARI | 0.714 | 0.715 | -0.001 |
| MNIST | Native NMI | 0.751 | 0.751 | 0.000 |
| Fashion-MNIST | Native ACC | 0.599 | 0.525 | +0.074 |
| Fashion-MNIST | Native ARI | 0.449 | 0.392 | +0.057 |
| Fashion-MNIST | Native NMI | 0.604 | 0.571 | +0.033 |
| MRI, three-view mean | Refit Silhouette | 0.313567 | 0.350933 | -0.037367 |
| MRI, three-view mean | Refit DB | 1.205067 | 1.099333 | +0.105733 |
| MRI, three-view mean | Self-clean atlas PSNR (dB) | 20.071248 | 20.013365 | +0.057884 |

MNIST and aggregate MRI Silhouette/DB satisfy the exploratory closeness bands
declared before reading these results. Fashion-MNIST performs better on all
three metrics, but ACC and ARI are **outside** those absolute-difference bands;
they are not labeled a numerical match. No acceptance threshold was declared
for PSNR. These results show functioning fresh training and comparable or
better performance in the measured condition; they are not an exact replay of
the old experiment or verification of every paper claim.

MNIST selects epoch 40 and Fashion-MNIST epoch 65 by maximum held-out refit ACC
after epoch 25, then reports native ACC/ARI/NMI at that epoch. Labels are not
used by training or GMM fitting, but are used by this selection rule.
MRI selects minimum held-out reconstruction loss after epoch 25. No best native
score is substituted for the score at the selected checkpoint.

MRI noisy/clean selected epochs are coronal 100/81, axial 96/86, and sagittal
74/81. Individual noisy-view refit Silhouette scores are 0.2509, 0.3806,
and 0.3092; their differences from the paper can be larger than the averaged
difference. All three noisy views retain two occupied native clusters.
PSNR uses independently trained clean atlases, one-to-one Hungarian matching,
and the historical 128x127 montage crop. It measures consistency under noise,
not fidelity to a shared patient-specific ground-truth target.

The [full JSON report](../experiments/audit_results/fresh_gamma012.json) includes
every comparison, selected epoch, per-cluster PSNR, and hash. All eight complete
numeric traces are in [fresh_traces](../experiments/audit_results/fresh_traces).
MRI traces retain the original dummy-label accuracy/ARI/NMI columns solely for
byte-identical audit; those columns have no scientific interpretation and are
excluded from the summarized MRI metrics. The training-run environment is
recorded in [fresh_environment.json](../experiments/audit_results/fresh_environment.json).

## Commands and recorded numerical options

From a source checkout with the package installed:

```bash
python -m pip install '.[experiments]'
python experiments/run_reference.py --dataset mnist --data-root ./data \
  --output ./outputs/mnist-g012 --device cuda:0 --gamma 0.12 --epochs 100 \
  --gmm-dtype float64 --train-silhouette-samples 2000
python experiments/run_reference.py --dataset fashion_mnist --data-root ./data \
  --output ./outputs/fashion-g012 --device cuda:0 --gamma 0.12 --epochs 100 \
  --gmm-dtype original --train-silhouette-samples 0
```

For each of `oasis_coronal`, `oasis_axial`, and `oasis_sagittal`, run both gamma=0
and gamma=0.12 using `--gmm-dtype float64`, an appropriate prepared-data root,
and a different output directory for each run. All runs use two data-loader
workers and four PyTorch threads. The launcher maps the selected physical GPU
to its local cuda:0.

Float64 GMM covariance estimation avoids the float32 Cholesky failure observed
in a sagittal-MRI diagnostic. Network training remains float32. The Fashion
run started before that precision option was introduced and used the archived
float32 GMM input; the command above records that distinction explicitly.
Its optional final uncertainty visualization emitted a covariance warning;
training, the selected clustering metrics, and all 100 CSV rows completed.

The MNIST training-only Silhouette diagnostic uses a fixed 2,000-point sample;
test metrics, GMM fits, train ACC/ARI/NMI/DB, all gradient updates, and selection
still use their full respective splits. Fashion and MRI use the full diagnostic.
These numerical/diagnostic choices are recorded, not silently equated with the
historical environment, whose exact versions are not independently established.

## Other tested interpretations and remaining scope

The main text's Adam/50-epoch-pretraining/500-update interpretation was tested
separately. Native ACC was 0.8737 on MNIST and 0.5720 on Fashion-MNIST using
their official test partitions, but axial MRI collapsed to one native cluster.
For MRI, extending main training to 450 or 500 epochs did not resolve that
collapse. The literal main-text variant is therefore not the recommended
reproduction recipe. The deterministic-pretraining variance-head overflow was
fixed by initializing that untrained head to unit variance in 0.3.1.
The completed alternative endpoints and differences are retained in
[alternative_protocol_comparison.csv](../experiments/audit_results/alternative_protocol_comparison.csv).

The full seven-noise average, multi-seed confidence intervals, all baselines,
clinical analyses, and runtime claims have not been newly reproduced here.
The earlier archive audit verifies reported aggregations separately. The
final NeurIPS version reconciles the optimizer, learning rate and schedule with
the recovered implementation; the split and checkpoint-selection rule are still
not described there. No manuscript metric has been replaced with these
new measurements.
