"""Run the archived training workflow through the installed multi-cluster models.

The manuscript's optimizer/pretraining description differs from this workflow;
see docs/reproduction.md before interpreting a run as paper reproduction.
"""
import argparse
import json
from pathlib import Path
import sys
from runtime import DATASETS, load_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gamma', type=float, default=.12)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--samples-per-class', type=int, default=12800)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--gmm-dtype', choices=['float64', 'original'], default='float64',
                   help='Use float64 GMM covariance estimation to avoid float32 Cholesky failures; model training remains float32')
    p.add_argument('--train-silhouette-samples', type=int, default=0,
                   help='Optional fixed subsample for training-only Silhouette diagnostics; test metrics, training and selection remain full-data')
    args = p.parse_args()
    if args.gamma < 0 or args.epochs < 1 or args.samples_per_class < 2:
        p.error('gamma >= 0, epochs >= 1, samples-per-class >= 2 required')
    if args.train_silhouette_samples < 0 or args.train_silhouette_samples == 1:
        p.error('train-silhouette-samples must be zero or at least two')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    entry, _ = load_reference(args.dataset, args.data_root, args.device, args.threads, args.workers)
    if args.gmm_dtype == 'float64':
        import numpy as np
        original_refit = entry.refit_gmm
        def stable_refit(latent, *pos, **kw):
            return original_refit(np.asarray(latent, dtype=np.float64), *pos, **kw)
        entry.refit_gmm = stable_refit
    if args.train_silhouette_samples:
        import numpy as np
        from training import trainer
        from utils import clustering
        original_metrics = trainer.compute_label_free_metrics
        def sampled_training_metrics(z, stage, refit, k, prefix=''):
            if prefix != 'train_' or len(z) <= args.train_silhouette_samples:
                return original_metrics(z, stage, refit, k, prefix)
            # Local RandomState: this diagnostic must not consume the training RNG.
            indices = np.random.RandomState(0).choice(len(z), args.train_silhouette_samples, replace=False)
            result = original_metrics(z[indices], stage[indices], refit[indices], k, prefix)
            for name, labels in [('stage', stage), ('refit', refit)]:
                result[f'{prefix}{name}_size_entropy'] = clustering.cluster_size_entropy(labels, k)
                result[f'{prefix}{name}_davies_bouldin'] = clustering.safe_davies_bouldin(z, labels)
            result['train_silhouette_samples'] = len(indices)
            return result
        trainer.compute_label_free_metrics = sampled_training_metrics
    entry.CFG.epoch_num = args.epochs
    entry.CFG.seed = args.seed
    entry.CFG.num_samples_per_class = args.samples_per_class
    entry.PROJECT_ROOT = str(args.output)
    # Keep an additional best-refit checkpoint, since table selection and the
    # legacy recon checkpoint can select different epochs. This saves, not trains.
    original_evaluate = entry.evaluate_and_save
    best_refit = -float('inf')
    def evaluate(*pos, **kw):
        nonlocal best_refit
        import torch
        result = original_evaluate(*pos, **kw)
        model, _, _, _, cfg, epoch = pos[:6]
        row = result[0]
        folder = Path(cfg.results_dir)
        if epoch > int(.25 * args.epochs) and row['test_refit_accuracy'] > best_refit:
            best_refit = row['test_refit_accuracy']
            torch.save(model.state_dict(), folder/'best_refit_vae.pth')
            (folder/'best_refit_metadata.json').write_text(json.dumps({'epoch': epoch, 'selection': 'max test_refit_accuracy after 25% warmup', 'metrics': row}, indent=2)+'\n')
        previous = kw.get('best_test_recon')
        if previous is not None and result[3] < previous:
            (folder/'best_recon_metadata.json').write_text(json.dumps({'epoch': epoch, 'selection': 'min test reconstruction after 25% warmup', 'metrics': row}, indent=2)+'\n')
        return result
    entry.evaluate_and_save = evaluate
    (args.output/'run_config.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, indent=2)+'\n')
    sys.argv = [str(entry.__file__), '--noise_type', 'cauchy', '--noise_level', str(args.gamma)]
    entry.main()


if __name__ == '__main__':
    main()
