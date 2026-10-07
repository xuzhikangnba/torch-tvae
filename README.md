# torch-tvae

**Pro-t-VAE: a label-free Student-t VAE, reusable Gaussian-Gamma prior, and original multi-cluster experiment models for PyTorch.**

Use the default MLP to train a t-VAE, supply your own encoder and decoder, or
replace the Gaussian KL in an existing VAE with an analytic augmented-model
regularizer. No labels, classification head, mixture assignments, or GMM fitting
are required. The core package depends only on PyTorch.

This library develops the Gaussian-Gamma construction in **Pro-t-VAE: Closed-Form
Student's t Mixture VAEs for Robust Image Prototype Discovery**
(accepted at NeurIPS 2026), by Zhikang Xu, Jiarui Xing, and Jian Wang. The arXiv
link will be added when the preprint is available.

Version 0.3.1 includes two interfaces: `TVAE` is the basic single-component model;
`tvae.reference.TVaDE28` and `TVaDE128` preserve the original multi-component
models and checkpoint parameter names for 28x28 images and 128x128 MRI.

**Validation:** 21 software tests pass, and eight fresh 100-epoch runs cover
MNIST/Fashion-MNIST at Cauchy gamma=0.12 and three MRI views at gamma=0/0.12.
MNIST and aggregate MRI metrics are close to the same-condition paper values;
Fashion-MNIST scores are higher, with ACC/ARI outside the declared numerical
closeness bands. See [the actual fresh results](docs/fresh_results.md).
This is **not a full-paper reproduction claim**: the seven-noise average and
multi-seed comparisons were not newly rerun. An independent archive audit
verifies 91 Table 1 entries. The [recovered configuration](docs/recovered_configuration.md)
matches 40 historical logs to the paper's source CSVs and identifies differences
from the manuscript's training wording. See [the audit](docs/reproduction.md)
and [mathematical objective](docs/mathematics.md) for scope and assumptions.

## Install

Python 3.10+ and PyTorch 2.1+ are required. Install the appropriate CPU/CUDA build
of PyTorch for your machine before installing this package if needed.

From the public repository:

```bash
python -m pip install "git+https://github.com/xuzhikangnba/torch-tvae.git"
```

Or from a downloaded source checkout:

```bash
python -m pip install .
```

Or install a built wheel:

```bash
python -m pip install dist/torch_tvae-0.3.1-py3-none-any.whl
```

**PyPI publication is pending.** The command `pip install torch-tvae` should
only be advertised after that release is published. Git/source/wheel installs
do not require PyPI publication.

## Original multi-cluster models

```python
from tvae.reference import TVaDE28, TVaDE128

digits = TVaDE28(latent_dim=8, num_clusters=10, nu_init=5.0)
brain = TVaDE128(latent_dim=32, num_clusters=2, nu_init=5.0)
```

These models require no classification labels in their loss. They retain the
archived detached cluster responsibilities and batch-loss convention. Original
training scripts, a portable launcher, and fresh checkpoint evaluation commands
are in [experiments](experiments/README.md). The experiment scripts are included
in the Git checkout/source distribution; the model classes are included in the
installable wheel. Data and model weights are not bundled.

## Reproduce the recovered paper workflow

The logs supporting the reported metrics use **AdamW, lr=0.001 for all
parameters, batch size 32, 100 epochs, and a train-GMM prior initialization at
epoch 20**. There is no separate reconstruction-only pretraining phase in this
workflow. Use the portable launcher from a source checkout:

```bash
python -m pip install '.[experiments]'
python experiments/run_reference.py --dataset mnist --data-root ./data \
  --output ./outputs/mnist-g012 --gamma 0.12 --epochs 100 --device cuda:0 \
  --gmm-dtype float64 --train-silhouette-samples 2000
```

Change the dataset to `fashion_mnist`, `oasis_coronal`, `oasis_axial`, or
`oasis_sagittal`; MRI needs the prepared data described in
[experiments](experiments/README.md). The basic `TVAE` example below is a
separate single-component model, not this clustering benchmark.

The recovered recipe differs from the accepted manuscript's optimizer and
pretraining wording. See the [source evidence](docs/recovered_configuration.md)
for the exact split, loss normalization, checkpoint-selection rule, and
distinction between native and GMM-refit metrics. A diagnostic implementing the
literal manuscript wording is described next; do not mix those two protocols.

## Diagnostic of the manuscript's stated two-stage schedule

The installed `tvae-paper-train` command uses the reference ResNet models, Adam
with separate autoencoder/mixture learning rates (0.001/0.002), batch size 32,
50 reconstruction-only pretraining epochs, and then 500 minibatch updates.
This is the literal interpretation of the main text's “500 iterations.”
`--main-epochs 450` or `--main-epochs 500` records the alternative durations
mentioned elsewhere in the manuscript. Tests of these interpretations did not
consistently match the paper, including native axial-MRI collapse. See
[the fresh-run comparison](docs/fresh_results.md); use the recovered workflow
above for the reported reference experiment.

```bash
python -m pip install '.[experiments]'
tvae-paper-train --dataset mnist --data-root ./data --output ./outputs/mnist-paper \
  --device cuda:0 --gamma 0.12
```

Change `--dataset` to `fashion_mnist` or an MRI view (`oasis_coronal`,
`oasis_axial`, `oasis_sagittal`). MRI needs the prepared NPZ files described in
[experiments](experiments/README.md). The command reports both train and test
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

See [the fresh-run protocol](docs/fresh_reproduction.md) for comparison criteria
and the complete recorded configuration. Basic-model examples below are separate
from these manuscript experiments.

## Train the basic model without labels

```python
import torch
from tvae import TVAE

model = TVAE(input_dim=784, latent_dim=16, df=5.0, learn_df=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Replace this with a batch of grayscale images in [0, 1].
x = torch.rand(32, 1, 28, 28)
optimizer.zero_grad(set_to_none=True)
result = model.loss(x)
result.loss.backward()
optimizer.step()

model.eval()
reconstructed = model.reconstruct(x)       # (32, 1, 28, 28)
generated = model.sample(16).view(-1, 1, 28, 28)
print(result.reconstruction.item(), result.regularization.item())
```

The default decoder returns Bernoulli **logits**; the loss uses a numerically
stable binary cross entropy. `reconstruct()` returns probabilities and decodes
the encoder mean. `sample()` returns decoder observation means at sampled
Student-t latents; it does not sample pixel observation noise.

For unconstrained continuous data, choose `likelihood="gaussian"` and a positive
`observation_scale`. Both objectives sum over event coordinates and average over
the batch. `beta=1` is the augmented negative ELBO for the selected observation
model; changing beta gives a beta-VAE objective. Fractional image values with
Bernoulli BCE use the conventional soft-target surrogate. The analytic
regularizer does not make the nonlinear reconstruction expectation analytic.

## Use the prior in an existing VAE

```python
from tvae import StudentTPrior

prior = StudentTPrior(latent_dim=16, df=5.0, learn_df=True)

# mean, logvar: (batch, latent_dim), from your Gaussian encoder.
# reconstruction_nll: (batch,), already summed over pixels/features.
regularizer = prior.kl_bound(mean, logvar)  # (batch,)
loss = (reconstruction_nll + regularizer).mean()
```

Replace the old Gaussian KL; do not add both penalties. Include
`prior.parameters()` in your optimizer when learning df, location, or scale.
Move the prior to the same device as your encoder. The regularizer supports
float32 and float64; cast mixed-precision encoder moments to float32 first.

`prior.terms(mean, logvar)` exposes the expected Gaussian KL, Gamma KL, total
bound, and inferred precision. `prior.scale_posterior(...)` returns a
shape/rate `torch.distributions.Gamma`. `prior.log_prob(z)` evaluates the actual
multivariate Student-t density, and `prior.rsample((n,))` samples from it.

## Custom networks

```python
model = TVAE(
    latent_dim=16,
    encoder=my_encoder,  # returns mean, logvar with shape (batch, 16)
    decoder=my_decoder,  # returns logits/means with the same shape as x
    likelihood="gaussian",
    observation_scale=0.5,
)
```

The custom Bernoulli decoder must return logits, without a final sigmoid.
The Gaussian decoder returns means. A complete CPU example with no external
data is available in `examples/custom_backbone.py`:

```bash
python examples/custom_backbone.py
```

## MNIST example

Install the optional image example dependencies from a source checkout:

```bash
python -m pip install '.[examples]'
tvae-mnist --epochs 10 --device cpu
# Quick functional check; this is not a quality benchmark:
tvae-mnist --epochs 1 --limit 128 --batch-size 32
```

Use `--device cuda:0` for GPU training. Labels are discarded. This command saves
a checkpoint, reconstructions, and decoded prior samples in `outputs/mnist`.
The single-component model's performance should be evaluated independently;
the paper's clustering improvements are not benchmarks for this new model.

## Save and restore

```python
config = dict(input_dim=784, latent_dim=16, df=5.0, learn_df=True)
torch.save({"config": config, "state_dict": model.state_dict()}, "model.pt")
checkpoint = torch.load("model.pt", map_location="cpu", weights_only=True)
restored = TVAE(**checkpoint["config"])
restored.load_state_dict(checkpoint["state_dict"])
restored.eval()
```

Record the actual constructor settings used for your model. Custom backbones
must be reconstructed explicitly before loading their state dictionaries.

## Tests and releases

```bash
python -m pip install '.[dev,experiments]'
python -m unittest discover -s tests -v
python -m build
python -m twine check dist/*
```

Tests cover an independent Monte Carlo estimate of the joint KL, finite
difference gradients, the Gaussian limit, shared-scale sampling, loss
normalization, custom networks, and checkpoint round trips.

See [publication instructions](docs/publishing.md) for GitHub and PyPI release
steps. Code is licensed under [MIT](LICENSE). Datasets and pretrained weights
are not included in the package.
