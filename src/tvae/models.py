"""A label-free t-VAE with an optional MLP backbone."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .priors import StudentTPrior


@dataclass
class ForwardOutput:
    decoder_output: Tensor
    mean: Tensor
    logvar: Tensor
    latent: Tensor


@dataclass
class LossOutput:
    loss: Tensor
    reconstruction: Tensor
    regularization: Tensor
    per_sample_loss: Tensor
    expected_precision: Tensor


def _mlp(dimensions):
    layers = []
    for i, (a, b) in enumerate(zip(dimensions, dimensions[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dimensions) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class _Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims):
        super().__init__()
        self.network = _mlp([input_dim, *hidden_dims, 2 * latent_dim])

    def forward(self, x):
        mean, logvar = self.network(x.flatten(1)).chunk(2, dim=-1)
        return mean, logvar


class TVAE(nn.Module):
    """Student-t latent VAE without labels, mixture components, or GMM fitting.

    Specify input_dim for the default MLP, or provide both encoder and decoder.
    A custom encoder returns (mean, logvar) of shape (batch, latent_dim).
    A custom decoder returns the input shape, with Bernoulli logits or Gaussian
    means according to likelihood. The Gaussian observation scale is fixed.
    """

    def __init__(self, input_dim: int | None = None, latent_dim: int = 16, *,
                 hidden_dims=(128, 64), df: float = 5.0, learn_df: bool = True,
                 encoder: nn.Module | None = None, decoder: nn.Module | None = None,
                 prior: StudentTPrior | None = None,
                 likelihood: str = "bernoulli", observation_scale: float = 1.0):
        super().__init__()
        if likelihood not in ("bernoulli", "gaussian"):
            raise ValueError("likelihood must be 'bernoulli' or 'gaussian'")
        if not math.isfinite(observation_scale) or observation_scale <= 0:
            raise ValueError("observation_scale must be finite and positive")
        self.prior = prior if prior is not None else StudentTPrior(latent_dim, df, learn_df=learn_df)
        if self.prior.latent_dim != latent_dim:
            raise ValueError("prior.latent_dim must match latent_dim")
        self._flat_decoder = encoder is None and decoder is None
        if self._flat_decoder:
            if not isinstance(input_dim, int) or isinstance(input_dim, bool) or input_dim < 1:
                raise ValueError("a positive input_dim is required for the default MLP")
            if any(not isinstance(d, int) or isinstance(d, bool) or d < 1 for d in hidden_dims):
                raise ValueError("hidden_dims must contain positive integers")
            encoder = _Encoder(input_dim, latent_dim, hidden_dims)
            decoder = _mlp([latent_dim, *reversed(hidden_dims), input_dim])
        elif encoder is None or decoder is None:
            raise ValueError("provide both a custom encoder and a custom decoder")
        self.encoder, self.decoder = encoder, decoder
        self.likelihood = likelihood
        self.observation_scale = float(observation_scale)

    def encode(self, x: Tensor):
        """Return diagonal-Gaussian posterior mean and log variance."""
        return self.encoder(x)

    def forward(self, x: Tensor, *, sample: bool = True) -> ForwardOutput:
        mean, logvar = self.encode(x)
        self.prior._validate(mean, logvar)
        if mean.ndim != 2 or mean.shape[0] != x.shape[0]:
            raise ValueError("encoder outputs must have shape (batch, latent_dim)")
        z = mean + (0.5 * logvar).exp() * torch.randn_like(mean) if sample else mean
        decoded = self.decoder(z)
        if self._flat_decoder:
            decoded = decoded.reshape_as(x)
        if decoded.shape != x.shape:
            raise ValueError("custom decoder output must match the input shape")
        return ForwardOutput(decoded, mean, logvar, z)

    def loss(self, x: Tensor, *, beta: float = 1.0) -> LossOutput:
        """One-sample reconstruction estimate plus an analytic prior penalty.

        Both losses are summed over event dimensions and averaged over the
        batch. beta=1 is the augmented negative ELBO for the chosen likelihood;
        other values define a beta-VAE objective. BCE on fractional [0,1]
        images is the usual soft-target surrogate rather than a Bernoulli
        observation likelihood for those fractional values.
        """
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        if x.ndim < 2 or x.shape[0] == 0:
            raise ValueError("x must have a nonempty batch and at least one event dimension")
        out = self(x)
        if self.likelihood == "bernoulli":
            if not torch.isfinite(x).all() or not ((x >= 0) & (x <= 1)).all():
                raise ValueError("Bernoulli targets must be finite and in [0, 1]")
            nll = F.binary_cross_entropy_with_logits(out.decoder_output, x, reduction="none")
        else:
            nll = (0.5 * ((x - out.decoder_output) / self.observation_scale).square()
                   + math.log(self.observation_scale) + 0.5 * math.log(2 * math.pi))
        reconstruction = nll.flatten(1).sum(-1)
        terms = self.prior.terms(out.mean, out.logvar)
        per_sample = reconstruction + beta * terms.total
        return LossOutput(per_sample.mean(), reconstruction.mean(), terms.total.mean(),
                          per_sample, terms.expected_precision)

    def _observation_mean(self, decoded):
        return decoded.sigmoid() if self.likelihood == "bernoulli" else decoded

    @torch.no_grad()
    def reconstruct(self, x: Tensor) -> Tensor:
        """Decode the posterior mean; call eval() first for custom dropout/BN."""
        return self._observation_mean(self(x, sample=False).decoder_output)

    @torch.no_grad()
    def sample(self, num_samples: int) -> Tensor:
        """Decode sampled prior latents to observation means (MLP output is flat)."""
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples < 1:
            raise ValueError("num_samples must be a positive integer")
        return self._observation_mean(self.decoder(self.prior.rsample((num_samples,))))
