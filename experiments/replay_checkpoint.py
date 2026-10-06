"""Fresh evaluation of an archived checkpoint, not an exact historic RNG replay.

Report both native Student-t assignments and separately refitted GMM metrics.
Labels are used only for evaluation, never for fitting either model.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from runtime import DATASETS, load_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gamma', type=float, default=.12)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--workers', type=int, default=0)
    args = p.parse_args()
    if args.gamma < 0:
        p.error('gamma must be nonnegative')
    entry, device = load_reference(args.dataset, args.data_root, args.device, args.threads, args.workers)
    import numpy as np
    import torch
    from utils.clustering import (collect_embeddings_and_images, refit_gmm, eval_refit_metrics,
        eval_prior_student_t_metrics, predict_refit, predict_stage, compute_label_free_metrics)
    entry.set_seed(args.seed)
    cfg = entry.CFG()
    clean = entry.create_balanced_mnist_subset(cfg.num_samples_per_class)
    noisy = entry.NoisyDataset(clean, 'cauchy', args.gamma)
    _, _, train, test = entry.create_data_loaders(noisy, cfg.batch_size, .8)
    # The split precedes model initialization in the archived training entry.
    model = entry.StudentTVaDE(cfg.latent_dim, cfg.num_clusters).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    Z_train, Y_train, _ = collect_embeddings_and_images(model, train, device)
    Z_test, Y_test, _ = collect_embeddings_and_images(model, test, device)
    gmm = refit_gmm(Z_train, cfg.num_clusters, cfg.refit_covariance_type, cfg.refit_n_init)
    c_refit, c_stage = predict_refit(gmm, Z_test), predict_stage(model, Z_test, device)
    metrics = compute_label_free_metrics(Z_test, c_stage, c_refit, cfg.num_clusters, prefix='test_')
    if not args.dataset.startswith('oasis_'):
        for key, values in [('refit', eval_refit_metrics(gmm, Z_test, Y_test, cfg.num_clusters)),
                            ('stage', eval_prior_student_t_metrics(model, Z_test, Y_test, cfg.num_clusters, device))]:
            for name, value in zip(['accuracy','nmi','ari'], values):
                metrics[f'test_{key}_{name}'] = float(value)
    bn_counts = sorted({int(value) for key,value in state.items() if key.endswith('num_batches_tracked')})
    steps = math.ceil(len(train.dataset) / cfg.batch_size)
    result = dict(dataset=args.dataset, gamma=args.gamma, seed=args.seed, device=str(device),
                  package_model=model.__class__.__module__, checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  samples_train=len(train.dataset), samples_test=len(test.dataset),
                  batchnorm_training_batches=bn_counts,
                  inferred_training_epochs=[n/steps for n in bn_counts],
                  inference='Epoch inferred from batchnorm counters assuming unchanged batch size and one train pass per epoch',
                  metrics=metrics, evaluation='Fresh fixed-seed evaluation; historic stochastic noise/RNG state was not stored',
                  torch_version=torch.__version__, numpy_version=np.__version__)
    # Strict JSON uses null for undefined metrics of a collapsed assignment.
    result['metrics'] = {k:(float(v) if v is not None and np.isfinite(v) else None) for k,v in metrics.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
