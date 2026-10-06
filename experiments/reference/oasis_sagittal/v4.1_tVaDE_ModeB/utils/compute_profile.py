#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute-efficiency profiling utilities.

Provides:
  - EpochProfiler: context manager capturing epoch wall-clock, mean step time,
                   peak GPU memory.
  - count_params(model): total trainable parameter count.
  - compute_flops(model, input_shape, device): forward FLOPs per sample (best-effort,
                   uses fvcore if available, falls back to thop, else returns None).
"""

import time
import torch


class EpochProfiler:
    """
    Lightweight per-epoch profiler.

    Usage:
        prof = EpochProfiler(device=device)   # device arg is kept for API
                                              # compat but not passed to CUDA
                                              # APIs (torch 1.8 quirks).
        prof.start_epoch()
        for batch in loader:
            prof.step_begin()
            # forward + backward + step
            prof.step_end()
        stats = prof.end_epoch()   # dict: epoch_time_sec, step_time_ms_mean,
                                   #       step_time_ms_std, peak_gpu_mem_mb, n_steps
    """

    def __init__(self, device=None):
        self.device = device
        self._epoch_t0 = None
        self._step_t0 = None
        self._step_times = []
        self._cuda = torch.cuda.is_available() and (
            device is None or (hasattr(device, "type") and device.type == "cuda")
        )
        # Record whether we've warned once about a broken CUDA-stats API,
        # so we don't spam the log.
        self._mem_api_broken = False

    # All CUDA API calls below pass device=None and rely on CUDA_VISIBLE_DEVICES
    # to pin the active device. This avoids the "Invalid device argument"
    # failures seen in older PyTorch when the device arg is a torch.device obj.
    def _sync(self):
        if not self._cuda:
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    def _reset_peak(self):
        if not self._cuda:
            return
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            if not self._mem_api_broken:
                print(f"[EpochProfiler] reset_peak_memory_stats() failed: "
                      f"{type(e).__name__}: {e} — will still report max via "
                      f"max_memory_allocated")
                self._mem_api_broken = True

    def _max_mem_mb(self):
        if not self._cuda:
            return None
        try:
            return float(torch.cuda.max_memory_allocated()) / 1e6
        except Exception as e:
            if not self._mem_api_broken:
                print(f"[EpochProfiler] max_memory_allocated() failed: "
                      f"{type(e).__name__}: {e}")
                self._mem_api_broken = True
            return None

    def start_epoch(self):
        self._step_times = []
        self._reset_peak()
        self._sync()
        self._epoch_t0 = time.perf_counter()

    def step_begin(self):
        self._sync()
        self._step_t0 = time.perf_counter()

    def step_end(self):
        self._sync()
        self._step_times.append(time.perf_counter() - self._step_t0)

    def end_epoch(self):
        self._sync()
        epoch_time = time.perf_counter() - self._epoch_t0
        n = len(self._step_times)
        if n == 0:
            mean_ms = 0.0
            std_ms = 0.0
        else:
            st = torch.tensor(self._step_times) * 1000.0
            mean_ms = float(st.mean())
            std_ms = float(st.std(unbiased=False)) if n > 1 else 0.0
        peak_mb = self._max_mem_mb()
        return {
            "epoch_time_sec": round(epoch_time, 3),
            "step_time_ms_mean": round(mean_ms, 3),
            "step_time_ms_std": round(std_ms, 3),
            "peak_gpu_mem_mb": None if peak_mb is None else round(peak_mb, 2),
            "n_steps": n,
        }


def count_params(model):
    """Return total trainable parameter count."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


class _FlopsProbe(torch.nn.Module):
    """Wraps encode->decode as a single forward so fvcore/thop can trace it."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        mu, _ = self.model.encode(x)
        return self.model.decode(mu)


def compute_flops(model, input_shape, device=None, verbose=True):
    """
    Best-effort forward FLOPs per sample (encoder + decoder path).

    FLOPs are a function of the ARCHITECTURE, not the device, so we always
    run the probe on CPU via a deep-copied model. This avoids fvcore/CUDA
    version incompatibilities that would otherwise silently return None.

    Args:
        model:        nn.Module, must expose .encode(x) and .decode(z)
        input_shape:  tuple without batch dim, e.g. (1, 128, 128)
        device:       (unused, accepted for API compat)
        verbose:      print a diagnostic message on each failed attempt
    """
    import copy
    cpu = torch.device("cpu")
    try:
        probe_model = copy.deepcopy(model).to(cpu).eval()
    except Exception as e:
        if verbose:
            print(f"[compute_flops] deepcopy failed ({type(e).__name__}: {e}); "
                  f"moving the original model to CPU as fallback")
        was_training = model.training
        model.eval().to(cpu)
        probe_model = model
    probe = _FlopsProbe(probe_model).to(cpu).eval()
    x = torch.randn(1, *input_shape, device=cpu)

    flops = None
    last_err = None
    # 1) fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(probe, x)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        flops = int(fca.total()) * 2         # MACs -> FLOPs
    except Exception as e:
        last_err = ("fvcore", type(e).__name__, str(e))
        if verbose:
            print(f"[compute_flops] fvcore failed: {type(e).__name__}: {e}")
    # 2) thop fallback
    if flops is None:
        try:
            from thop import profile as _thop_profile
            macs, _ = _thop_profile(probe, inputs=(x,), verbose=False)
            flops = int(macs) * 2
        except Exception as e:
            last_err = ("thop", type(e).__name__, str(e))
            if verbose:
                print(f"[compute_flops] thop failed: {type(e).__name__}: {e}")

    if flops is None and verbose and last_err is not None:
        print(f"[compute_flops] all backends failed; last error: {last_err}")

    # If we had to move the original model to CPU, move it back.
    orig_device = next(model.parameters()).device
    target_device = device if device is not None else orig_device
    if probe_model is model and target_device != cpu:
        model.to(target_device)
    return flops


def compute_module_flops(module, args, verbose=True):
    """
    Best-effort forward FLOPs per call for an arbitrary nn.Module.

    Use this for components NOT covered by `compute_flops` (which only traces
    encode->decode), such as v1.3's latent DDPM noise predictor.

    Args:
        module: nn.Module to trace.
        args:   tuple of positional inputs that match `module.forward(...)`.
                Each tensor must be batch_size=1 so the result is per-sample.
        verbose: print diagnostics on failure.

    Returns int (FLOPs, MACs * 2) or None if all backends failed.
    """
    import copy
    cpu = torch.device("cpu")
    try:
        probe = copy.deepcopy(module).to(cpu).eval()
    except Exception as e:
        if verbose:
            print(f"[compute_module_flops] deepcopy failed ({type(e).__name__}: "
                  f"{e}); using original module on CPU as fallback")
        module.eval().to(cpu)
        probe = module
    cpu_args = tuple(
        a.to(cpu) if isinstance(a, torch.Tensor) else a for a in args
    )

    flops = None
    last_err = None
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(probe, cpu_args)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        flops = int(fca.total()) * 2
    except Exception as e:
        last_err = ("fvcore", type(e).__name__, str(e))
        if verbose:
            print(f"[compute_module_flops] fvcore failed: {type(e).__name__}: {e}")
    if flops is None:
        try:
            from thop import profile as _thop_profile
            macs, _ = _thop_profile(probe, inputs=cpu_args, verbose=False)
            flops = int(macs) * 2
        except Exception as e:
            last_err = ("thop", type(e).__name__, str(e))
            if verbose:
                print(f"[compute_module_flops] thop failed: {type(e).__name__}: {e}")
    if flops is None and verbose and last_err is not None:
        print(f"[compute_module_flops] all backends failed; last error: {last_err}")
    return flops


def write_meta_json(path, **kwargs):
    """Write a small meta.json summarising model size / FLOPs / config."""
    import json
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(kwargs, f, indent=2)
