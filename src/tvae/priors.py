"""An elliptical Student-t prior with one shared Gamma precision per vector.

Gamma parameters use shape/rate. The diagonal scale squared is not the
marginal covariance: Cov[z] = df / (df - 2) * diag(scale**2), for df > 2.
The analytic joint variational KL upper-bounds the marginal Gaussian-to-t KL.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions import Gamma


@dataclass
class PriorTerms:
    """Unreduced terms with the encoder's leading batch dimensions."""

    total: Tensor
    expected_gaussian_kl: Tensor
    gamma_kl: Tensor
    expected_precision: Tensor


class StudentTPrior(nn.Module):
    """Diagonal-scale multivariate Student-t prior, independent of a backbone.

    The encoder posterior is a diagonal Gaussian. A scalar auxiliary precision
    is shared by every latent coordinate, giving an elliptical multivariate t,
    not a product of univariate Student-t distributions.

    Args:
        latent_dim: Number of latent coordinates.
        df: Initial degrees of freedom, strictly greater than min_df.
        learn_df: Learn df with a positive softplus parameterization.
        loc: Scalar or length-latent_dim location; fixed unless learn_loc=True.
        scale: Positive scalar or length-latent_dim scale, not variance.
        learn_loc: Learn the prior location.
        learn_scale: Learn the prior diagonal scale.
        min_df: Positive numerical floor for the degrees of freedom.

    Use float32 or float64 tensors on the same device as this module.
    """

    def __init__(self, latent_dim: int, df: float = 5.0, *,
                 learn_df: bool = True, loc=0.0, scale=1.0,
                 learn_loc: bool = False, learn_scale: bool = False,
                 min_df: float = 1e-3):
        super().__init__()
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError("latent_dim must be a positive integer")
        if not math.isfinite(min_df) or min_df <= 0:
            raise ValueError("min_df must be finite and positive")
        if not math.isfinite(df) or df <= min_df:
            raise ValueError("df must be finite and greater than min_df")
        self.latent_dim = latent_dim
        self.min_df = float(min_df)
        dtype = torch.get_default_dtype()

        def vector(value, name):
            value = torch.as_tensor(value, dtype=dtype).detach().clone()
            if value.ndim == 0:
                value = value.expand(latent_dim).clone()
            if value.shape != (latent_dim,) or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite and scalar or shape ({latent_dim},)")
            return value

        def parameter_or_buffer(name, value, learn):
            if learn:
                self.register_parameter(name, nn.Parameter(value))
            else:
                self.register_buffer(name, value)

        location = vector(loc, "loc")
        scales = vector(scale, "scale")
        if not (scales > 0).all():
            raise ValueError("scale must be strictly positive")
        parameter_or_buffer("loc", location, learn_loc)
        parameter_or_buffer("log_scale", scales.log(), learn_scale)
        # Stable inverse softplus, including large df used for the Gaussian limit.
        value = df - min_df
        raw = value + math.log(-math.expm1(-value))
        parameter_or_buffer("raw_df", torch.tensor(raw, dtype=dtype), learn_df)

    @property
    def df(self) -> Tensor:
        return F.softplus(self.raw_df) + self.min_df

    @property
    def scale(self) -> Tensor:
        return self.log_scale.exp()

    def _validate(self, mean: Tensor, logvar: Tensor):
        if mean.shape != logvar.shape or mean.ndim < 1 or mean.shape[-1] != self.latent_dim:
            raise ValueError("mean and logvar must have matching (..., latent_dim) shapes")
        if mean.dtype != logvar.dtype or mean.dtype not in (torch.float32, torch.float64):
            raise TypeError("mean and logvar must share float32 or float64 dtype")
        if mean.device != logvar.device or mean.device != self.loc.device:
            raise ValueError("encoder tensors and prior must be on the same device")

    def _distance(self, mean: Tensor, logvar: Tensor) -> Tensor:
        return ((logvar.exp() + (mean - self.loc).square())
                * (-2 * self.log_scale).exp()).sum(dim=-1)

    def scale_posterior(self, mean: Tensor, logvar: Tensor) -> Gamma:
        """Optimal factor q(u|x), using encoder moments rather than a sampled z."""
        self._validate(mean, logvar)
        rate = (self.df + self._distance(mean, logvar)) / 2
        shape = ((self.df + self.latent_dim) / 2).expand_as(rate)
        return Gamma(shape, rate)

    def terms(self, mean: Tensor, logvar: Tensor) -> PriorTerms:
        """Return the two analytic KL terms; no latent sample is needed."""
        self._validate(mean, logvar)
        distance = self._distance(mean, logvar)
        a0 = b0 = self.df / 2
        a = a0 + self.latent_dim / 2
        b = (self.df + distance) / 2
        expected_u = a / b
        expected_log_u = torch.digamma(a) - b.log()
        gaussian = 0.5 * (
            expected_u * distance - self.latent_dim * expected_log_u
            + 2 * self.log_scale.sum() - logvar.sum(dim=-1) - self.latent_dim
        )
        gamma = ((a - a0) * torch.digamma(a) + torch.lgamma(a0) - torch.lgamma(a)
                 + a0 * (b.log() - b0.log()) + a * (b0 / b - 1))
        return PriorTerms(gaussian + gamma, gaussian, gamma, expected_u)

    def kl_bound(self, mean: Tensor, logvar: Tensor, reduction: str = "none") -> Tensor:
        """Joint KL upper bound on KL(q(z|x) || p_t(z)), not that marginal KL.

        Sum latent coordinates, then optionally reduce batch dimensions.
        This replaces the usual Gaussian VAE KL; do not add both penalties.
        """
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        value = self.terms(mean, logvar).total
        return value if reduction == "none" else getattr(value, reduction)()

    def forward(self, mean: Tensor, logvar: Tensor) -> Tensor:
        """Alias for the unreduced joint KL bound."""
        return self.kl_bound(mean, logvar)

    def log_prob(self, value: Tensor) -> Tensor:
        """Normalized marginal multivariate Student-t log density."""
        if value.ndim < 1 or value.shape[-1] != self.latent_dim:
            raise ValueError("value must have shape (..., latent_dim)")
        d, nu = self.latent_dim, self.df
        distance = (((value - self.loc) / self.scale).square()).sum(-1)
        return (torch.lgamma((nu + d) / 2) - torch.lgamma(nu / 2)
                - d / 2 * (nu.log() + math.log(math.pi)) - self.log_scale.sum()
                - (nu + d) / 2 * torch.log1p(distance / nu))

    def rsample(self, sample_shape=torch.Size()) -> Tensor:
        """Reparameterized latent samples using one Gamma draw per vector."""
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        shape = torch.Size(sample_shape)
        u = Gamma(self.df / 2, self.df / 2).rsample(shape)
        noise = torch.randn(shape + (self.latent_dim,), dtype=self.loc.dtype,
                            device=self.loc.device)
        return self.loc + self.scale * noise / u.sqrt().unsqueeze(-1)
