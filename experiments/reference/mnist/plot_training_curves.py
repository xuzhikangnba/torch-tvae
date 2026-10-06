#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1: Per-algorithm training curves visualization.

Reads a metrics CSV and produces a 3-subplot figure:
  (1) Loss convergence: train/test total + component losses
  (2) Stage accuracy/ari/nmi (train & test)
  (3) Refit accuracy/ari/nmi (train & test)

Usage:
  python plot_training_curves.py <csv_path> [--algo_name <name>] [--out_dir <dir>]

  Or import and call:
    from plot_training_curves import plot_training_curves
    plot_training_curves(csv_path, algo_name="v4.0_ModeA_AdamW_lr0.001_bs32", out_dir="results")
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curves(csv_path, algo_name=None, out_dir=None):
    """
    Plot 3-subplot training curves from a metrics CSV.

    Parameters
    ----------
    csv_path : str
        Path to the metrics CSV file.
    algo_name : str, optional
        Algorithm name for title and filename. If None, derived from csv filename.
    out_dir : str, optional
        Output directory. If None, same directory as csv_path.
    """
    df = pd.read_csv(csv_path)

    if algo_name is None:
        # Derive from CSV filename: metrics_YYYYMMDD_AdamW_lr0.001_bs32.csv
        basename = os.path.splitext(os.path.basename(csv_path))[0]
        # Remove 'metrics_' prefix
        if basename.startswith("metrics_"):
            algo_name = basename[len("metrics_"):]
        else:
            algo_name = basename

        # Prepend directory name (e.g., v4.0_tVaDE_ModeA)
        parent_dir = os.path.basename(os.path.dirname(csv_path))
        if parent_dir.startswith("v"):
            algo_name = f"{parent_dir}_{algo_name}"

    if out_dir is None:
        out_dir = os.path.dirname(csv_path)

    epochs = df["epoch"].values

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(algo_name, fontsize=14, fontweight="bold")

    # ---- Subplot 1: Loss convergence ----
    ax = axes[0]
    ax.set_title("Loss Convergence")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (per sample)")

    # Total loss
    if "train_loss" in df.columns:
        ax.plot(epochs, df["train_loss"], "b-", linewidth=2, label="Train Total")
    if "test_loss" in df.columns:
        ax.plot(epochs, df["test_loss"], "b--", linewidth=2, label="Test Total")

    # Component losses (Mode A: recon, kl, gmm; Mode B: recon, elbo)
    component_styles = {
        "train_loss_recon": ("r-", "Train Recon"),
        "test_loss_recon": ("r--", "Test Recon"),
        "train_loss_kl": ("g-", "Train KL"),
        "test_loss_kl": ("g--", "Test KL"),
        "train_loss_gmm": ("m-", "Train GMM/t-mix"),
        "test_loss_gmm": ("m--", "Test GMM/t-mix"),
        "train_loss_elbo": ("c-", "Train ELBO"),
        "test_loss_elbo": ("c--", "Test ELBO"),
    }
    for col, (style, label) in component_styles.items():
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().any():
                ax.plot(epochs, vals, style, linewidth=1, alpha=0.8, label=label)

    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Mark phase transition if present
    if "phase" in df.columns:
        phase_vals = df["phase"].values
        for i in range(1, len(phase_vals)):
            if phase_vals[i] != phase_vals[i - 1]:
                ax.axvline(x=epochs[i], color="k", linestyle=":", linewidth=1.5, alpha=0.7)
                ax.text(epochs[i], ax.get_ylim()[1] * 0.95, "Phase\nTransition",
                        ha="center", va="top", fontsize=7, color="k")
                break

    # ---- Subplot 2: Stage metrics ----
    ax = axes[1]
    ax.set_title("Stage Metrics (Prior-based)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")

    stage_cols = {
        "train_stage_accuracy": ("b-", "Train ACC"),
        "test_stage_accuracy": ("b--", "Test ACC"),
        "train_stage_ari": ("r-", "Train ARI"),
        "test_stage_ari": ("r--", "Test ARI"),
        "train_stage_nmi": ("g-", "Train NMI"),
        "test_stage_nmi": ("g--", "Test NMI"),
    }
    for col, (style, label) in stage_cols.items():
        if col in df.columns:
            ax.plot(epochs, df[col], style, linewidth=1.5, label=label)

    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    if "phase" in df.columns:
        phase_vals = df["phase"].values
        for i in range(1, len(phase_vals)):
            if phase_vals[i] != phase_vals[i - 1]:
                ax.axvline(x=epochs[i], color="k", linestyle=":", linewidth=1.5, alpha=0.7)
                break

    # ---- Subplot 3: Refit metrics ----
    ax = axes[2]
    ax.set_title("Refit Metrics (sklearn GMM)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")

    refit_cols = {
        "train_refit_accuracy": ("b-", "Train ACC"),
        "test_refit_accuracy": ("b--", "Test ACC"),
        "train_refit_ari": ("r-", "Train ARI"),
        "test_refit_ari": ("r--", "Test ARI"),
        "train_refit_nmi": ("g-", "Train NMI"),
        "test_refit_nmi": ("g--", "Test NMI"),
    }
    for col, (style, label) in refit_cols.items():
        if col in df.columns:
            ax.plot(epochs, df[col], style, linewidth=1.5, label=label)

    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    if "phase" in df.columns:
        phase_vals = df["phase"].values
        for i in range(1, len(phase_vals)):
            if phase_vals[i] != phase_vals[i - 1]:
                ax.axvline(x=epochs[i], color="k", linestyle=":", linewidth=1.5, alpha=0.7)
                break

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{algo_name}_training_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot per-algorithm training curves")
    parser.add_argument("csv_path", help="Path to metrics CSV")
    parser.add_argument("--algo_name", default=None, help="Algorithm name for title/filename")
    parser.add_argument("--out_dir", default=None, help="Output directory")
    args = parser.parse_args()

    plot_training_curves(args.csv_path, algo_name=args.algo_name, out_dir=args.out_dir)
