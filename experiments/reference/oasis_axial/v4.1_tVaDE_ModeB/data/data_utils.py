#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OASIS-2D brain MRI data loading utility (coronal view).

Replaces the MNIST pipeline with OASIS-2D. Keeps the original function
names (`create_balanced_mnist_subset`, `create_data_loaders`) so the
downstream train.py / train_noisy.py scripts need no modification.

Source data:
  NIPS2026/data/2026-04-OASIS-2D/{coronal,axial,sagittal}.npz
  Each .npz contains 'images' (N, 128, 128) float32 in [0, 1].

Processing Pipeline (data_summary.txt) -- already applied in the .npz:
  1. Original 128x128 MRI slice
  2. Bbox crop: 94x83 (Y:[21,114], X:[21,103])
  3. Pad to square: 94x94
  4. Resize back to 128x128

Excluded subject (OAS30296_MR_d0069) has already been removed from all
.npz files. Final dataset: 253 subjects per view.

No ground-truth class labels are available for this unsupervised task;
dummy label 0 is used for all samples. (K=2 clustering is evaluated
purely via ELBO / reconstruction / cluster coherence; ARI/NMI/Acc will
not be meaningful unless the user supplies external labels later.)

To match the existing VAE encoder/decoder (28x28), we bilinear-downsample
from 128x128 to 28x28 on the fly inside the Dataset.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# View selected for this experiment folder.
OASIS_VIEW = "coronal"

# Absolute path to the OASIS-2D data directory (inside this exp folder).
# data_utils.py lives at: <exp_root>/<version>/data/data_utils.py
# Go up two levels to <exp_root>, then into data/2026-04-OASIS-2D/.
OASIS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "2026-04-OASIS-2D")
)


class OASIS2DDataset(Dataset):
    """OASIS-2D single-view dataset.

    Loads the preprocessed .npz (N x 128 x 128 float32 in [0, 1]) and
    resizes to `target_size` x `target_size` (default 28) on construction
    so that __getitem__ is cheap and deterministic.
    """

    def __init__(self, view: str = OASIS_VIEW, root: str = OASIS_ROOT,
                 target_size: int = 128):
        assert view in ("coronal", "axial", "sagittal"), f"Invalid view: {view}"
        self.view = view
        self.root = root
        self.target_size = target_size

        npz_path = os.path.join(root, f"{view}.npz")
        with np.load(npz_path, allow_pickle=True) as d:
            imgs = d["images"].astype(np.float32)  # (N, 128, 128) in [0, 1]
            # Subject IDs:
            # 1) Prefer a sidecar plain-text file (one ID per line) — robust
            #    across numpy versions. Pre-extracted to the OASIS data root.
            # 2) Fallback: try to unpickle from npz directly. Some envs
            #    (older numpy) cannot deserialise them.
            sids = None
            sidecar = os.path.join(root, f"{view}_subject_ids.txt")
            if os.path.exists(sidecar):
                with open(sidecar) as fh:
                    sids = [ln.strip() for ln in fh if ln.strip()]
            if sids is None and "subject_ids" in d.files:
                try:
                    sids = [str(s) for s in np.asarray(d["subject_ids"])]
                except Exception:
                    sids = None

        # Ensure positive strides and contiguous memory
        imgs = np.ascontiguousarray(imgs)

        # Bilinear resize to target_size
        imgs_t = torch.from_numpy(imgs).unsqueeze(1)  # (N, 1, 128, 128)
        imgs_t = F.interpolate(
            imgs_t, size=(target_size, target_size),
            mode="bilinear", align_corners=False,
        )
        imgs_t = imgs_t.clamp(0.0, 1.0).contiguous()
        self.images = imgs_t                                  # (N, 1, T, T)
        self.labels = torch.zeros(self.images.size(0), dtype=torch.long)
        # Per-sample identifiers, aligned with self.images / self.labels.
        # Useful for cross-referencing cluster_samples PNGs back to subjects.
        if sids is None or len(sids) != int(self.images.size(0)):
            sids = [f"{view}_{i:04d}" for i in range(int(self.images.size(0)))]
        self.sample_ids = sids

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        return self.images[idx], int(self.labels[idx])


def create_balanced_mnist_subset(num_samples_per_class: int = 12800):
    """
    Drop-in replacement for the MNIST subset builder.

    The argument is ignored (OASIS-2D is a small fixed 253-sample set);
    keeping the function name avoids touching train.py and train_noisy.py.
    """
    return OASIS2DDataset(view=OASIS_VIEW, root=OASIS_ROOT, target_size=128)


def create_data_loaders(subset_data, batch_size: int = 128, train_split: float = 0.8):
    """Create train/test DataLoaders (train shuffled; eval loaders not shuffled)."""
    n = len(subset_data)
    train_size = int(train_split * n)
    test_size = n - train_size
    train_data, test_data = random_split(subset_data, [train_size, test_size])

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=torch.cuda.is_available(),
    )
    train_loader_eval = DataLoader(
        train_data, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=torch.cuda.is_available(),
    )
    test_loader_eval = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader, train_loader_eval, test_loader_eval
