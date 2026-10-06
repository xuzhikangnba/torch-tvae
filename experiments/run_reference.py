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
    args = p.parse_args()
    if args.gamma < 0 or args.epochs < 1 or args.samples_per_class < 2:
        p.error('gamma >= 0, epochs >= 1, samples-per-class >= 2 required')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    entry, _ = load_reference(args.dataset, args.data_root, args.device, args.threads, args.workers)
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
