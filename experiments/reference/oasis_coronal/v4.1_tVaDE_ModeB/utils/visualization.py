#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization and atlas building utilities.
"""

import os
import warnings
import numpy as np
import torch
import torchvision.utils as vutils
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore", category=FutureWarning)


@torch.no_grad()
def save_grid(imgs: torch.Tensor, out_path: str, nrow: int) -> None:
    """Save a grid of images."""
    vutils.save_image(
        imgs.detach().cpu(),
        out_path,
        nrow=nrow,
        normalize=True,
        value_range=(0, 1)
    )


@torch.no_grad()
def save_refit_gmm_means_atlas(vae, gmm: GaussianMixture, out_dir: str, epoch: int, nrow: int):
    """
    Atlas = decode(refit GMM means).
    Saves:
      gmm_refit_means_atlas_epoch_XXX.png (grid)
      gmm_refit_mean_atlas_cluster_XX_epoch_XXX.png (per cluster)
      
    Args:
        vae: VAE model
        gmm: Fitted GMM
        out_dir: Output directory
        epoch: Current epoch
        nrow: Number of images per row
        
    Returns:
        Decoded images tensor
    """
    os.makedirs(out_dir, exist_ok=True)

    device = next(vae.parameters()).device
    means = torch.tensor(gmm.means_, dtype=torch.float32, device=device)  # [K,D]

    vae.eval()
    imgs = vae.decode(means).detach().cpu()  # [K,1,28,28]

    # per-cluster
    K = imgs.shape[0]
    for k in range(K):
        vutils.save_image(
            imgs[k],
            os.path.join(out_dir, f"gmm_refit_mean_atlas_cluster_{k:02d}_epoch_{epoch:03d}.png"),
            normalize=True,
            value_range=(0, 1)
        )

    # grid
    vutils.save_image(
        imgs,
        os.path.join(out_dir, f"gmm_refit_means_atlas_epoch_{epoch:03d}.png"),
        nrow=nrow,
        normalize=True,
        value_range=(0, 1)
    )
    return imgs


def save_cluster_sample_grids(X_cpu, Y_true, prob, out_dir, epoch,
                              topn_per_cluster=64, conf_thresh=0.0, nrow=8,
                              sample_ids=None, label_every=25):
    """
    Save per-cluster topN sample grids and return purity stats list.

    Args:
        X_cpu: [N,1,H,W] CPU tensor
        Y_true: [N] numpy labels
        prob: [N,K] numpy probabilities
        out_dir: Output directory
        epoch: Current epoch
        topn_per_cluster: Number of top samples per cluster
        conf_thresh: Confidence threshold
        nrow: Number of images per row
        sample_ids: Optional list[str] of length N giving each row of X_cpu
                    a per-sample identifier (e.g. OASIS subject_id). When
                    provided:
                      • a sidecar CSV `cluster_<k>_samples_epoch_<e>.csv`
                        is always written, listing the sample_ids in
                        grid-row-major order (matches the .png layout)
                      • additionally, every `label_every` epochs (and on
                        epoch 1) a matplotlib variant
                        `cluster_<k>_samples_labeled_epoch_<e>.png`
                        is rendered with the sample_id printed under each
                        image. This is the visualisation used to look
                        up subjects' medical metadata.

    Returns:
        List of purity statistics dictionaries.
    """
    os.makedirs(out_dir, exist_ok=True)
    N, K = prob.shape
    hard = prob.argmax(axis=1)
    conf = prob.max(axis=1)

    write_labels = sample_ids is not None
    do_labeled_image = write_labels and (epoch == 1 or epoch % label_every == 0)
    if do_labeled_image:
        # Lazy matplotlib import; not needed if labels off
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    if write_labels:
        import csv as _csv

    stats = []
    for k in range(K):
        idx = np.where(hard == k)[0]
        if idx.size == 0:
            row = {"cluster": k, "count": 0, "top_digit": -1, "purity": 0.0}
            for d in range(10):
                row[f"digit_{d}_count"] = 0
            stats.append(row)
            continue

        if conf_thresh > 0.0:
            idx = idx[conf[idx] >= conf_thresh]
            if idx.size == 0:
                row = {"cluster": k, "count": 0, "top_digit": -1, "purity": 0.0}
                for d in range(10):
                    row[f"digit_{d}_count"] = 0
                stats.append(row)
                continue

        scores = prob[idx, k]
        order = np.argsort(-scores)
        idx = idx[order][:topn_per_cluster]

        grid = X_cpu[idx]
        save_path = os.path.join(out_dir, f"cluster_{k:02d}_samples_epoch_{epoch:03d}.png")
        vutils.save_image(grid, save_path, nrow=nrow, normalize=True, value_range=(0, 1))

        # Always write the sidecar CSV when sample_ids are available, so
        # users can cross-reference subjects without re-running training.
        if write_labels:
            csv_path = os.path.join(
                out_dir, f"cluster_{k:02d}_samples_epoch_{epoch:03d}.csv")
            with open(csv_path, "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["grid_pos", "row", "col", "sample_id",
                            "true_label", "responsibility"])
                for pos, j in enumerate(idx):
                    sid = str(sample_ids[int(j)]) if int(j) < len(sample_ids) else f"#{int(j)}"
                    w.writerow([pos, pos // nrow, pos % nrow, sid,
                                int(Y_true[int(j)]), float(prob[int(j), k])])

        # Optional matplotlib variant with text labels.
        if do_labeled_image:
            n_imgs = int(idx.size)
            n_cols = nrow
            n_rows_g = (n_imgs + n_cols - 1) // n_cols
            # Compact figure: 1 inch per image cell, label adds a thin strip.
            fig_w = max(6, n_cols * 1.0)
            fig_h = max(2, n_rows_g * 1.25)
            fig, axes = plt.subplots(n_rows_g, n_cols,
                                     figsize=(fig_w, fig_h),
                                     squeeze=False)
            for r in range(n_rows_g):
                for c in range(n_cols):
                    ax = axes[r, c]
                    pos = r * n_cols + c
                    if pos < n_imgs:
                        j = int(idx[pos])
                        img_t = X_cpu[j]
                        # take channel 0 for visualisation
                        img2d = img_t[0].cpu().numpy() if img_t.dim() >= 3 else img_t.cpu().numpy()
                        ax.imshow(img2d, cmap="gray", vmin=0, vmax=1)
                        sid_short = str(sample_ids[j]) if j < len(sample_ids) else f"#{j}"
                        # truncate for layout sanity
                        if len(sid_short) > 18:
                            sid_short = sid_short[:7] + "…" + sid_short[-8:]
                        ax.set_title(sid_short, fontsize=6, pad=1)
                    ax.axis("off")
            fig.suptitle(f"cluster {k} — epoch {epoch}  (top-{n_imgs} by responsibility)",
                         fontsize=10)
            fig.tight_layout()
            out_lab = os.path.join(
                out_dir, f"cluster_{k:02d}_samples_labeled_epoch_{epoch:03d}.png")
            fig.savefig(out_lab, dpi=70, bbox_inches="tight")
            plt.close(fig)

        yk = Y_true[idx]
        counts = np.bincount(yk, minlength=10)
        top_digit = int(counts.argmax())
        purity = float(counts.max() / max(1, counts.sum()))

        row = {"cluster": k, "count": int(counts.sum()), "top_digit": top_digit, "purity": purity}
        for d in range(10):
            row[f"digit_{d}_count"] = int(counts[d])
        stats.append(row)

    return stats


@torch.no_grad()
def build_cluster_atlas_A(X_cpu, prob, out_dir, epoch,
                          conf_thresh=0.0, gamma_power=1.0,
                          max_per_cluster=5000,
                          robust="median"):
    """
    Scheme-A atlas: per cluster robust mean/median in pixel space using hard assignment from prob.
    
    Saves:
      atlas_epoch_XXX.png
      atlas_cluster_XX_epoch_XXX.png
      
    Args:
        X_cpu: [N,1,H,W] CPU tensor
        prob: [N,K] numpy probabilities
        out_dir: Output directory
        epoch: Current epoch
        conf_thresh: Confidence threshold
        gamma_power: Power for sharpening
        max_per_cluster: Max samples per cluster
        robust: "mean" or "median"
        
    Returns:
        atlases tensor [K,1,H,W] on CPU
    """
    from .clustering import sharpen_probs

    os.makedirs(out_dir, exist_ok=True)
    P = sharpen_probs(prob, gamma_power)
    hard = P.argmax(axis=1)
    conf = P.max(axis=1)

    # Derive image shape from the data (C, H, W) instead of hard-coding 1x28x28.
    # This makes the function work for OASIS (1,128,128) and any future size.
    img_shape = tuple(X_cpu.shape[1:])  # (C, H, W)
    empty_atlas = torch.zeros(*img_shape)

    N, K = P.shape
    atlases = []
    for k in range(K):
        idx = np.where(hard == k)[0]
        if idx.size == 0:
            atlas = empty_atlas
            atlases.append(atlas)
            vutils.save_image(atlas, os.path.join(out_dir, f"atlas_cluster_{k:02d}_epoch_{epoch:03d}.png"),
                              normalize=True, value_range=(0, 1))
            continue

        if conf_thresh > 0.0:
            idx = idx[conf[idx] >= conf_thresh]
            if idx.size == 0:
                atlas = empty_atlas
                atlases.append(atlas)
                vutils.save_image(atlas, os.path.join(out_dir, f"atlas_cluster_{k:02d}_epoch_{epoch:03d}.png"),
                                  normalize=True, value_range=(0, 1))
                continue

        if idx.size > max_per_cluster:
            scores = P[idx, k]
            order = np.argsort(-scores)
            idx = idx[order][:max_per_cluster]

        imgs = X_cpu[idx].float()  # [M, C, H, W]
        if robust == "median":
            atlas = imgs.median(dim=0).values
        else:
            atlas = imgs.mean(dim=0)

        atlases.append(atlas)
        vutils.save_image(atlas, os.path.join(out_dir, f"atlas_cluster_{k:02d}_epoch_{epoch:03d}.png"),
                          normalize=True, value_range=(0, 1))

    atlases = torch.stack(atlases, dim=0)  # [K, C, H, W]
    vutils.save_image(atlases, os.path.join(out_dir, f"atlas_epoch_{epoch:03d}.png"),
                      nrow=K, normalize=True, value_range=(0, 1))
    return atlases


def _gmm_component_cov(gmm, k: int):
    """Reconstruct a (D,D) covariance matrix for cluster k from a fitted
    sklearn GaussianMixture, regardless of covariance_type."""
    ct = gmm.covariance_type
    if ct == "full":
        return gmm.covariances_[k]                              # (D, D)
    elif ct == "diag":
        return np.diag(gmm.covariances_[k])                     # (D,) -> (D,D)
    elif ct == "spherical":
        D = gmm.means_.shape[1]
        return float(gmm.covariances_[k]) * np.eye(D)
    elif ct == "tied":
        return gmm.covariances_                                 # (D, D) shared
    else:
        raise ValueError(f"Unknown covariance_type: {ct}")


@torch.no_grad()
def save_refit_gmm_uncertainty_atlas(vae, gmm: GaussianMixture, out_dir: str,
                                     epoch: int, nrow: int,
                                     n_samples: int = 128, seed: int = 0,
                                     overlay_alpha: float = 0.7,
                                     unc_vmax: float = None,
                                     vmax_pct: float = 99.0,
                                     vcenter_pct: float = 90.0):
    """
    Per-cluster decoder uncertainty map, OVERLAID on the decoded mean image.

    For each refit GMM cluster k:
        1. Decode the cluster mean μ_k to get the "atlas" grayscale image.
        2. Draw n_samples z ~ N(μ_k, Σ_k); decode each → (N, C, H, W).
        3. Pixel-wise std over the N decoded samples → (C, H, W) uncertainty
           (Monte-Carlo estimate of the marginal pixel std under the prior).
        4. Render: grayscale atlas as background + RdBu_r uncertainty heatmap
           with alpha=overlay_alpha as foreground.  Matches the visual style
           of paper Fig.3 (mean atlas + uncertainty overlay).

    Outputs:
        gmm_refit_uncertainty_atlas_epoch_XXX.png       (grid, K panels)
        gmm_refit_uncertainty_cluster_XX_epoch_XXX.png  (per cluster)

    Args:
        vae: VAE with .decode(z)
        gmm: fitted sklearn GaussianMixture
        out_dir: output dir
        epoch: epoch index
        nrow: panels per row in the grid figure
        n_samples: number of latent draws per cluster (default 128)
        seed: RNG seed for reproducibility
        overlay_alpha: opacity of the heatmap on top of the grayscale atlas.
            Default 0.7 — colours dominate while the atlas anatomy still
            shows through. The colour scheme uses `coolwarm` diverging
            (saturated BLUE → WHITE → saturated RED), which pixel-matches
            the published Fig.3 reference colorbar.
        unc_vmax: hard upper bound for the colormap. If None, computed from
            `vmax_pct` percentile of the std distribution across all K
            clusters. Per-algo / per-call normalisation: clusters within one
            algo share the same scale; cross-algo magnitudes are NOT
            comparable on this scale.
        vmax_pct: percentile used for the diverging colormap's upper bound
            (the RED end). Default 99.
        vcenter_pct: percentile used for the diverging colormap's WHITE
            mid-point (via matplotlib TwoSlopeNorm). Default 90 — i.e. 90 % of
            pixels fall in the BLUE half (giving the published "all-blue
            background + red boundary highlights" look). Set to 50 for the
            classical symmetric vmin=0 / vmax linear behaviour.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as mpl_cm
    from matplotlib.colors import Normalize, TwoSlopeNorm

    os.makedirs(out_dir, exist_ok=True)
    # epoch can be int OR string like "best" — build a safe tag for filenames / CSV
    epoch_tag = str(epoch) if isinstance(epoch, str) else f"{epoch:03d}"
    device = next(vae.parameters()).device
    vae.eval()
    K, D = gmm.means_.shape
    rng = np.random.default_rng(seed)

    # 1) Decode the K cluster means -> grayscale atlas backgrounds.
    means_t = torch.tensor(gmm.means_, dtype=torch.float32, device=device)
    atlas = vae.decode(means_t).detach().cpu().clamp(0, 1).numpy()   # (K, C, H, W)

    # 2) Per-cluster MC std maps.
    unc_per_cluster = []
    for k in range(K):
        cov_k = _gmm_component_cov(gmm, k)
        cov_k = cov_k + 1e-8 * np.eye(D)        # PD jitter
        z_np = rng.multivariate_normal(gmm.means_[k], cov_k, size=n_samples)
        z = torch.from_numpy(z_np).float().to(device)
        decoded = vae.decode(z).detach().cpu().clamp(0, 1)           # (N, C, H, W)
        unc = decoded.std(dim=0).numpy()                              # (C, H, W)
        unc_per_cluster.append(unc)

    # Per-algo normalisation: all K cluster panels share the same vmax /
    # vcenter so within-algo cluster comparison is fair.
    all_u = np.concatenate([u.reshape(-1) for u in unc_per_cluster])
    if unc_vmax is None:
        unc_vmax = float(np.percentile(all_u, vmax_pct))
    unc_vmax = max(unc_vmax, 1e-6)
    vcenter = float(np.percentile(all_u, vcenter_pct))
    # Clip vcenter into the open interval (0, vmax) — TwoSlopeNorm rejects
    # the boundary cases.
    vcenter = float(np.clip(vcenter, 1e-7, unc_vmax - 1e-7))

    # `coolwarm` diverging: brighter saturated BLUE at vmin (#3B4CC0) →
    # WHITE midpoint → saturated RED at vmax (#B40426). Pixel-by-pixel
    # match with the published Fig.3 colorbar (verified by colorbar
    # extraction: max |ΔRGB| ≤ 11 across 11 sampled stops).
    # Using TwoSlopeNorm puts the WHITE midpoint at the `vcenter_pct`
    # quantile of the std distribution, so most pixels land in the BLUE
    # half — giving the "all-blue background + red boundary highlights"
    # look from Fig.3.
    CMAP_NAME = "coolwarm"
    norm = TwoSlopeNorm(vmin=0.0, vcenter=vcenter, vmax=unc_vmax)

    def _overlay(ax, atlas_img, unc_img, title=None):
        ax.imshow(atlas_img, cmap="gray", vmin=0, vmax=1)
        ax.imshow(unc_img, cmap=CMAP_NAME, norm=norm, alpha=overlay_alpha)
        ax.set_xticks([]); ax.set_yticks([])
        if title is not None:
            ax.set_title(title, fontsize=9)
        sm = mpl_cm.ScalarMappable(cmap=CMAP_NAME, norm=norm)
        sm.set_array([])
        return sm

    # Pre-compute per-cluster + per-image-region stats for both the
    # in-figure title strings (Layer 1) and the sidecar CSV (Layer 2).
    # These describe the RAW std distribution before TwoSlopeNorm scaling,
    # so cross-algo comparisons can later be made on absolute terms.
    cluster_stats = []   # one dict per cluster
    for k, u in enumerate(unc_per_cluster):
        flat = u.reshape(-1)
        cluster_stats.append({
            "epoch": epoch_tag,
            "cluster_id": int(k),
            "mean_std": float(flat.mean()),
            "p50":      float(np.percentile(flat, 50)),
            "p90":      float(np.percentile(flat, 90)),
            "p99":      float(np.percentile(flat, 99)),
            "max_std":  float(flat.max()),
            "vmax_used":    float(unc_vmax),
            "vcenter_used": float(vcenter),
        })

    norm_label = (f"vmin=0  vcenter={vcenter:.4f} (p{vcenter_pct:g})"
                  f"  vmax={unc_vmax:.4f} (p{vmax_pct:g})")

    # 3) Per-cluster overlay images.
    for k in range(K):
        st = cluster_stats[k]
        fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.4))
        im = _overlay(ax, atlas[k, 0], unc_per_cluster[k][0])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("std", fontsize=8)
        # Title carries norm + this cluster's raw stats so the colours can be
        # interpreted in absolute terms when picking samples for cross-algo
        # comparison later.
        ax.set_title(
            f"Cluster {k}  (epoch {epoch})\n"
            f"raw std: mean={st['mean_std']:.4f}  p99={st['p99']:.4f}  "
            f"max={st['max_std']:.4f}\n"
            f"{norm_label}",
            fontsize=8,
        )
        out_p = os.path.join(
            out_dir,
            f"gmm_refit_uncertainty_cluster_{k:02d}_epoch_{epoch_tag}.png")
        fig.savefig(out_p, dpi=120, bbox_inches="tight")
        plt.close(fig)

    # 4) Grid figure: K panels in nrow columns.
    n_cols = int(nrow) if nrow else K
    n_rows = int(np.ceil(K / n_cols))
    # Extra row-height to fit the 2-line suptitle + 2-line per-panel titles
    # without overlap. Use constrained_layout so matplotlib auto-arranges
    # both the panel titles and the suptitle.
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.8 * n_cols, 3.2 * n_rows + 0.6),
                             constrained_layout=True)
    if K == 1:
        axes_iter = [axes]
    elif n_rows == 1:
        axes_iter = list(axes) if hasattr(axes, "__iter__") else [axes]
    elif n_cols == 1:
        axes_iter = list(axes)
    else:
        axes_iter = [axes[i // n_cols, i % n_cols] for i in range(n_rows * n_cols)]
    last_im = None
    for i in range(n_rows * n_cols):
        ax = axes_iter[i]
        if i < K:
            st = cluster_stats[i]
            last_im = _overlay(
                ax, atlas[i, 0], unc_per_cluster[i][0],
                # Per-cluster sub-title (smaller font to keep things compact)
                title=(f"Cluster {i}  "
                       f"mean={st['mean_std']:.3f}  p99={st['p99']:.3f}"))
            # Override _overlay's fontsize=9 with a tighter one
            ax.title.set_fontsize(8)
        else:
            ax.axis("off")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes_iter[:K],
                            fraction=0.025, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("std", fontsize=9)
    # Suptitle: full normalisation info. constrained_layout will auto-make
    # room above the panel titles so this two-line text never overlaps.
    fig.suptitle(
        f"Refit GMM Uncertainty  (epoch {epoch}, n_samples={n_samples})\n"
        f"{norm_label}",
        fontsize=10,
    )
    grid_p = os.path.join(out_dir,
                          f"gmm_refit_uncertainty_atlas_epoch_{epoch_tag}.png")
    fig.savefig(grid_p, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Layer 2: append per-cluster stats to a sidecar CSV. One row per cluster,
    # appended each call. Header is written once on creation. Use this CSV to
    # later identify which (algo, epoch, cluster) combinations to include in
    # cross-algo comparison figures (e.g. "find epochs where v4.1's p99 std
    # bottoms out before v1.1 catches up").
    import csv as _csv
    stats_path = os.path.join(out_dir, "uncertainty_stats.csv")
    write_header = not os.path.exists(stats_path)
    with open(stats_path, "a", newline="") as f:
        cols = ["epoch", "cluster_id", "mean_std",
                "p50", "p90", "p99", "max_std",
                "vmax_used", "vcenter_used"]
        w = _csv.DictWriter(f, fieldnames=cols)
        if write_header:
            w.writeheader()
        for row in cluster_stats:
            w.writerow({k: row[k] for k in cols})


@torch.no_grad()
def save_student_t_means_decoded(vae, out_dir: str, epoch: int, nrow: int):
    """
    Decode VAE's own prior means (works for Gaussian or Student-t VAE) and
    save as an image grid. Named 'student_t' for legacy compatibility with
    v4.* training scripts; the implementation only requires vae.gmm_means and
    vae.decode, both of which exist in every VAE family used here.
    """
    os.makedirs(out_dir, exist_ok=True)
    vae.eval()
    means = vae.gmm_means.detach()                   # (K, D)
    imgs = vae.decode(means).detach().cpu()          # (K, C, H, W)
    vutils.save_image(
        imgs,
        os.path.join(out_dir, f"st_means_decoded_epoch_{epoch:03d}.png"),
        nrow=nrow, normalize=True, value_range=(0, 1),
    )


def save_latent_tsne(Z, Y, gmm, out_path, title="Latent Space t-SNE",
                     max_samples=10000):
    """Save t-SNE visualization of latent space with true labels, GMM contours,
    and cluster centers.

    Args:
        Z: [N, D] numpy array of latent embeddings
        Y: [N] numpy array of true labels
        gmm: fitted sklearn GaussianMixture (used to draw cluster centers)
        out_path: file path for the output PNG
        title: plot title
        max_samples: subsample to this many points for speed
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # Subsample for speed
    N = Z.shape[0]
    if N > max_samples:
        idx = np.random.choice(N, max_samples, replace=False)
        Z_sub, Y_sub = Z[idx], Y[idx]
    else:
        Z_sub, Y_sub = Z, Y

    # Run t-SNE on data points
    K = int(Y_sub.max()) + 1
    gmm_means = gmm.means_  # [K_gmm, D]

    # Concatenate GMM means so they get projected into the same t-SNE space
    Z_all = np.concatenate([Z_sub, gmm_means], axis=0)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(Z_all) - 1), init='random')
    Z_2d_all = tsne.fit_transform(Z_all)

    Z_2d = Z_2d_all[:len(Z_sub)]
    means_2d = Z_2d_all[len(Z_sub):]

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Scatter data points colored by true label
    cmap = plt.get_cmap("tab10", K)
    for k in range(K):
        mask = Y_sub == k
        ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1],
                   c=[cmap(k)], s=6, alpha=0.5, label=f"Class {k}")

    # Mark GMM centers
    ax.scatter(means_2d[:, 0], means_2d[:, 1],
               c="black", marker="X", s=200, edgecolors="white",
               linewidths=1.5, zorder=10, label="GMM centers")

    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=8, markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved t-SNE visualization to {out_path}")
