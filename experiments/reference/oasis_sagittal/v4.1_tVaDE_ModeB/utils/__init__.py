#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities package for CLAST.
"""

from .general import set_seed, ensure_dir, write_csv
from .clustering import (
    refit_gmm,
    hungarian_accuracy,
    eval_refit_metrics,
    eval_prior_student_t_metrics,
    collect_embeddings_and_images,
    sharpen_probs,
    overwrite_vae_prior_from_refit_gmm,
    predict_refit,
    predict_stage,
    cluster_size_entropy,
    safe_silhouette,
    safe_davies_bouldin,
    compute_label_free_metrics,
)
from .visualization import (
    save_grid,
    save_refit_gmm_means_atlas,
    save_refit_gmm_uncertainty_atlas,
    save_cluster_sample_grids,
    build_cluster_atlas_A,
    save_student_t_means_decoded,
    save_latent_tsne
)
from .compute_profile import (
    EpochProfiler,
    count_params,
    compute_flops,
    write_meta_json,
)

__all__ = [
    'set_seed',
    'ensure_dir',
    'write_csv',
    'refit_gmm',
    'hungarian_accuracy',
    'eval_refit_metrics',
    'eval_prior_student_t_metrics',
    'collect_embeddings_and_images',
    'sharpen_probs',
    'overwrite_vae_prior_from_refit_gmm',
    'predict_refit',
    'predict_stage',
    'cluster_size_entropy',
    'safe_silhouette',
    'safe_davies_bouldin',
    'compute_label_free_metrics',
    'save_grid',
    'save_refit_gmm_means_atlas',
    'save_refit_gmm_uncertainty_atlas',
    'save_cluster_sample_grids',
    'build_cluster_atlas_A',
    'save_student_t_means_decoded',
    'save_latent_tsne',
    'EpochProfiler',
    'count_params',
    'compute_flops',
    'write_meta_json',
]
