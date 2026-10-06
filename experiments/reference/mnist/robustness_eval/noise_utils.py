#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared noise utilities for robustness evaluation.

Provides NoisyDataset wrapper that adds dynamic noise to images on-the-fly,
as well as a set of composable noise primitives suitable for MRI / structured
artifact simulation on OASIS-2D.

Noise spec format (used by build_noise_fn / NoisyDataset(noise_spec=...)):

    spec = [
        {"type": "cauchy",    "level": 0.03,  "p": 1.0},
        {"type": "gaussian",  "sigma": 0.05,  "p": 1.0},
        {"type": "rician",    "sigma": 0.05,  "p": 1.0},
        {"type": "ring",      "n_rings": 3, "amp": 0.05, "freq": 40, "p": 0.5},
        {"type": "biasfield", "order": 3, "amp": 0.15,  "p": 0.5},
    ]

Each primitive returns a tensor of the same shape as x, clamped to [0, 1].

Backward compatibility
----------------------
Legacy construction `NoisyDataset(ds, "cauchy", 0.05)` still works; it is
translated into a single-primitive spec [{"type": "cauchy", "level": 0.05}].

Four preset OASIS noise tracks are exposed via PRESET_TRACKS:
    cauchy_only       Cauchy only (baseline; single-element spec)
    cauchy_gaussian   70% Cauchy + Gaussian
    cauchy_rician     Rician + Cauchy (MRI-realistic)
    cauchy_ring       Cauchy + light ring artifact
Each preset takes an intensity tag in {"mild","medium","severe"} and returns
a spec list.
"""

import math
import torch
from torch.utils.data import Dataset


# ============================================================
# Noise primitives (single-image tensor in [0, 1])
# ============================================================

def _cauchy(x, level):
    """Additive Cauchy noise (location 0, scale=level)."""
    return torch.distributions.Cauchy(0.0, float(level)).sample(x.shape)


def _gaussian(x, sigma):
    """Additive zero-mean Gaussian noise."""
    return float(sigma) * torch.randn_like(x)


def _rician(x, sigma):
    """
    Rician noise model (common in MRI magnitude images):
        x_noisy = sqrt((x + sigma * n1)^2 + (sigma * n2)^2)
    Returned value is the NOISE ONLY (x_noisy - x) so the caller can add/clip
    uniformly with other primitives.
    """
    n1 = torch.randn_like(x)
    n2 = torch.randn_like(x)
    sigma = float(sigma)
    noisy = torch.sqrt((x + sigma * n1) ** 2 + (sigma * n2) ** 2)
    return noisy - x


def _ring_artifact(x, n_rings=3, amp=0.05, freq=40.0, phase=None):
    """
    Low-frequency concentric ring artifact in the spatial domain.
    Approximates k-space radial undersampling artefacts.

    Uses a sum of sinusoids in polar radius with random phase per ring.
    """
    if x.dim() == 3:
        _, H, W = x.shape
    elif x.dim() == 2:
        H, W = x.shape
    else:
        raise ValueError(f"_ring_artifact: unexpected ndim {x.dim()}")
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy = torch.arange(H, dtype=x.dtype, device=x.device).view(-1, 1) - cy
    xx = torch.arange(W, dtype=x.dtype, device=x.device).view(1, -1) - cx
    r = torch.sqrt(xx * xx + yy * yy) / max(H, W)
    noise = torch.zeros_like(r)
    for i in range(int(n_rings)):
        phase_i = torch.rand(1, device=x.device).item() * 2 * math.pi if phase is None else phase
        freq_i = float(freq) * (0.8 + 0.4 * torch.rand(1).item())
        noise = noise + torch.sin(2 * math.pi * freq_i * r + phase_i)
    noise = (amp / max(1, n_rings)) * noise
    if x.dim() == 3:
        noise = noise.unsqueeze(0).expand_as(x)
    return noise


def _bias_field(x, order=3, amp=0.15):
    """
    Smooth low-frequency multiplicative bias field (N4-style).
    Implemented as a low-order 2D polynomial in normalised coords.

    Returns the NOISE = (B - 1) * x so it can be added like any other primitive.
    """
    if x.dim() == 3:
        _, H, W = x.shape
    elif x.dim() == 2:
        H, W = x.shape
    else:
        raise ValueError(f"_bias_field: unexpected ndim {x.dim()}")
    y = torch.linspace(-1.0, 1.0, H, dtype=x.dtype, device=x.device).view(-1, 1)
    xc = torch.linspace(-1.0, 1.0, W, dtype=x.dtype, device=x.device).view(1, -1)
    field = torch.zeros((H, W), dtype=x.dtype, device=x.device)
    # random polynomial coefficients up to given order
    for i in range(int(order) + 1):
        for j in range(int(order) + 1 - i):
            c = (2 * torch.rand(1, device=x.device).item() - 1.0)
            field = field + c * (y ** i) * (xc ** j)
    # normalise to zero mean, unit max-abs, then scale
    field = field - field.mean()
    mx = field.abs().max().clamp_min(1e-8)
    field = field / mx                         # [-1, 1] approx
    B = 1.0 + float(amp) * field               # multiplicative
    if x.dim() == 3:
        B = B.unsqueeze(0)
    return (B * x - x)


# ============================================================
# Spec → callable
# ============================================================

_PRIMITIVES = {
    "cauchy":    lambda x, p: _cauchy(x, p.get("level", 0.03)),
    "gaussian":  lambda x, p: _gaussian(x, p.get("sigma", 0.03)),
    "rician":    lambda x, p: _rician(x, p.get("sigma", 0.03)),
    "ring":      lambda x, p: _ring_artifact(x,
                                             n_rings=p.get("n_rings", 3),
                                             amp=p.get("amp", 0.05),
                                             freq=p.get("freq", 40.0)),
    "biasfield": lambda x, p: _bias_field(x,
                                          order=p.get("order", 3),
                                          amp=p.get("amp", 0.15)),
}


def apply_spec(x, spec, clamp=True):
    """
    Apply a list of noise primitives (additive contributions, each with prob p).

    Args:
        x:     (C, H, W) or (H, W) tensor in [0, 1].
        spec:  list of dicts (see module docstring).
        clamp: clamp the final tensor to [0, 1].

    Returns:
        Tensor of same shape as x.
    """
    out = x.clone()
    for p in spec:
        prob = float(p.get("p", 1.0))
        if prob < 1.0 and torch.rand(1).item() > prob:
            continue
        t = p.get("type")
        fn = _PRIMITIVES.get(t)
        if fn is None:
            raise ValueError(f"Unknown noise primitive: {t!r}")
        out = out + fn(x, p)
    if clamp:
        out = torch.clamp(out, 0.0, 1.0)
    return out


# ============================================================
# Preset tracks for exp16-24 robustness evaluation
# ============================================================

# Strength grid: mild / medium / severe  (medium matches the old "g0.05"-ish)
_STRENGTH = {
    "mild":    {"cauchy_gamma": 0.02, "gauss_sigma": 0.02, "rician_sigma": 0.03,
                "ring_amp": 0.03, "bias_amp": 0.08},
    "medium":  {"cauchy_gamma": 0.05, "gauss_sigma": 0.04, "rician_sigma": 0.05,
                "ring_amp": 0.05, "bias_amp": 0.15},
    "severe":  {"cauchy_gamma": 0.10, "gauss_sigma": 0.08, "rician_sigma": 0.08,
                "ring_amp": 0.08, "bias_amp": 0.22},
}


def _spec_cauchy_only(s):
    return [{"type": "cauchy", "level": s["cauchy_gamma"], "p": 1.0}]


def _spec_cauchy_gaussian(s):
    # 70/30 weighted mixture; implemented as two independent additive draws
    return [
        {"type": "cauchy",   "level": 0.7 * s["cauchy_gamma"], "p": 1.0},
        {"type": "gaussian", "sigma": 0.3 * s["gauss_sigma"], "p": 1.0},
    ]


def _spec_cauchy_rician(s):
    return [
        {"type": "rician", "sigma": s["rician_sigma"], "p": 1.0},
        {"type": "cauchy", "level": 0.5 * s["cauchy_gamma"], "p": 1.0},
    ]


def _spec_cauchy_ring(s):
    return [
        {"type": "cauchy", "level": s["cauchy_gamma"], "p": 1.0},
        {"type": "ring",   "n_rings": 3, "amp": s["ring_amp"], "freq": 40.0, "p": 1.0},
    ]


PRESET_TRACKS = {
    "cauchy_only":     _spec_cauchy_only,
    "cauchy_gaussian": _spec_cauchy_gaussian,
    "cauchy_rician":   _spec_cauchy_rician,
    "cauchy_ring":     _spec_cauchy_ring,
}


def build_noise_spec(track: str, strength: str):
    """
    Look up a preset track + strength → noise spec list.

    Args:
        track:    one of PRESET_TRACKS keys.
        strength: 'mild' | 'medium' | 'severe'.

    Returns:
        List of noise primitive dicts suitable for apply_spec().
    """
    if track not in PRESET_TRACKS:
        raise ValueError(f"Unknown track {track!r}; pick one of {list(PRESET_TRACKS)}")
    if strength not in _STRENGTH:
        raise ValueError(f"Unknown strength {strength!r}; pick one of {list(_STRENGTH)}")
    return PRESET_TRACKS[track](_STRENGTH[strength])


# ============================================================
# Dataset wrapper
# ============================================================

class NoisyDataset(Dataset):
    """
    Wraps an existing dataset and adds noise dynamically (re-sampled each access).

    Two construction modes:

      Legacy:
          NoisyDataset(ds, "cauchy", 0.05)   # (noise_type, noise_level)

      Spec-based:
          NoisyDataset(ds, noise_spec=[...], tag="cauchy_rician_medium")

    When `noise_spec` is given it takes precedence and `noise_type/noise_level`
    are ignored.
    """

    def __init__(self, dataset, noise_type="cauchy", noise_level=0.05,
                 noise_spec=None, tag=None):
        self.dataset = dataset
        self.noise_type = noise_type.lower() if isinstance(noise_type, str) else None
        self.noise_level = float(noise_level)
        if noise_spec is None:
            # translate legacy -> spec (with level=0 being a no-op)
            if self.noise_level <= 0:
                self.spec = []
            elif self.noise_type == "cauchy":
                self.spec = [{"type": "cauchy", "level": self.noise_level, "p": 1.0}]
            elif self.noise_type == "gaussian":
                self.spec = [{"type": "gaussian", "sigma": self.noise_level, "p": 1.0}]
            else:
                raise ValueError(f"Unknown legacy noise type: {self.noise_type!r}")
        else:
            self.spec = list(noise_spec)
        self.tag = tag or (f"{self.noise_type}_{self.noise_level}" if noise_spec is None else "custom")

        # Forward sample_ids from the inner dataset (e.g., OASIS2DDataset) so
        # that downstream cluster_samples CSVs can show real subject IDs even
        # though the dataset is wrapped with noise here. Falls back gracefully
        # if the inner dataset does not expose sample_ids.
        if hasattr(dataset, "sample_ids"):
            self.sample_ids = dataset.sample_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if not self.spec:
            return x, y
        x = apply_spec(x, self.spec, clamp=True)
        return x, y
