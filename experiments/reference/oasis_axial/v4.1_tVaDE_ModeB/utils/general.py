#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General utility functions.
"""

import os
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def write_csv(rows, out_path: str) -> None:
    """
    Write list of dictionaries to CSV file, atomically and flushed.

    Design choices:
        * Union of ALL row keys is used as the header. Earlier rows missing a
          later-added key get an empty value (not a crash).
        * File is written to a '.tmp' sibling then renamed over the target, so
          readers never see a half-written file.
        * flush() + os.fsync() before close ensure the contents hit disk even
          if the Python process exits immediately after.

    Args:
        rows: List of dictionaries.
        out_path: Target CSV path.
    """
    import csv as _csv
    if not rows:
        return

    # Union of keys, preserving first-seen order (stable column layout).
    seen = {}
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen[k] = len(seen)
    keys = sorted(seen, key=seen.get)

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp_path, out_path)
