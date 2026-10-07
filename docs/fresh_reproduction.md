# Fresh reproduction protocol

The author requests newly trained local models, with results approximately
matching the manuscript; historical checkpoint identity is not a prerequisite.
No checkpoints or MRI data are to be published.

## First configuration

- Installed wheel, original ResNet model classes, float32.
- Adam, autoencoder lr 0.001, mixture lr 0.002, batch size 32, no weight decay.
- 50 epochs reconstruction-only pretraining, then **500 minibatch updates**.
- K=10 and latent dimension 8 for digits/clothing; K=2 and dimension 32 for MRI.
- Degrees of freedom initialized to 5 and learned through softplus.
- Pixel-summed BCE and analytic prior penalty both averaged over observations;
  prior coefficient 1, as in the displayed ELBO.
- Fresh clipped additive Cauchy training noise, reconstructed against the noisy
  observation. Labels never enter the loss or mixture initialization.
- Seed 42. Digit/clothing datasets use the official 60,000/10,000 split; MRI
  uses the archived-style seeded 80/20 split of 253 prepared subjects.
- GMM initialization after pretraining uses only train embeddings (10 restarts,
  full covariance whose diagonal initializes the prior), with seed 0.
- At the deterministic-AE to VAE transition, initialize the untrained encoder
  variance head to unit variance (zero weights and bias). Version 0.3.0 left
  this head random; sagittal MRI exposed exponential overflow. Version 0.3.1
  fixes this initialization without changing the pretrained means or decoder.
  `--posterior-init preserve` retains the old behavior for an explicit comparison.
- Fixed independent evaluation noise for train/test; both native Student-t and
  separately refitted GMM assignments are reported and never interchanged.
- Final training endpoint is primary. Intermediate results remain diagnostics;
  no best test accuracy is substituted for the final endpoint.

The main text does not specify latent dimension, pretraining latent sampling,
observation likelihood, gradient clipping, GMM initialization, split details, or
fixed versus resampled evaluation noise. These are explicit implementation
choices recorded in every `run_config.json`, not claims about historical runs.
The default pretraining decoder consumes the encoder mean. A sampled-latent
variant is available as `--pretrain-latent sample`.

## Comparisons

First run gamma=0.12 on MNIST, Fashion-MNIST, and all three MRI views. Compare
against the same condition in `experiments/paper_targets.json`. A single noisy
condition must not be compared with the paper's seven-condition average.
Only a complete 0.02:0.02:0.14 sweep can validate that average. MRI Table 1
aggregates coronal/axial/sagittal; an individual view is not its table value.

For exploratory diagnosis, practical closeness bands are declared before
reading fresh results: absolute ACC difference <=0.03, ARI/NMI <=0.05,
Silhouette <=0.05, and relative DB difference <=15%. These are transparent
engineering thresholds, not statistical confidence intervals. All raw values
and differences must accompany any pass/fail summary.

If the literal schedule misses the targets, compare the explicitly recorded
50+450 and 50+500 epoch alternatives and, where needed, the archived configuration
and data split. The author authorized these comparisons. Configurations chosen
using these reported test metrics are exploratory replication diagnostics, not
an independent untouched-test evaluation. Freeze a final protocol and validate
it on another seed before a broad reproducibility claim.

Any configuration that succeeds must be described accurately in the released
code and reconciled with the manuscript. Do not label a different schedule as
“500 iterations” or hide changed loss weights/optimizer settings. No result is
currently certified by this protocol document alone.

## Local outputs

Each run creates configuration and data manifests, append-only training logs,
fixed-noise endpoint metrics, local model/optimizer/RNG state, and native/refit
prototype arrays. The public repository excludes these model/data artifacts.
The `--resume` option continues the stored minibatch order and RNG state.
`--from-pretrained .../pretrain.pt` starts a separately named configuration from
the same completed pretraining, checking pretraining settings for agreement.
Both commands require explicit output paths and refuse silent overwrites.
