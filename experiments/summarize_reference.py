"""Compare complete fresh reference runs with the same-condition paper rows.

MRI PSNR uses a separately trained clean atlas and the historical montage crop.
This script never treats a single noise condition as a seven-condition average.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from audit_artifacts import legacy_cells


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_run(path):
    cfg = json.loads((path / 'run_config.json').read_text())
    files = list((path / 'robustness_eval/results').glob('v4.1_*/metrics_*.csv'))
    if len(files) != 1:
        raise ValueError(f'Expected one metrics CSV: {path}')
    frame = pd.read_csv(files[0])
    epochs = int(cfg['epochs'])
    if frame.epoch.tolist() != list(range(1, epochs + 1)):
        raise ValueError(f'Run is incomplete or has duplicate epochs: {path}')
    if epochs != 100:
        raise ValueError(f'Historical protocol requires 100 epochs: {path}')
    eligible = frame.loc[frame.epoch.gt(int(.25 * epochs))]
    metric = 'test_loss_recon' if cfg['dataset'].startswith('oasis_') else 'test_refit_accuracy'
    valid = eligible.loc[eligible[metric].notna()]
    if valid.empty:
        raise ValueError(f'No eligible evaluation: {path}')
    index = valid[metric].idxmin() if metric == 'test_loss_recon' else valid[metric].idxmax()
    selected = frame.loc[index]
    selected_metrics = {
        str(k): float(v) for k, v in selected.items()
        if k.startswith(('train_', 'test_')) and pd.notna(v)
        and not (cfg['dataset'].startswith('oasis_') and k.endswith(('_accuracy', '_ari', '_nmi')))
    }
    effective = {k: v for k, v in cfg.items() if k not in ('output', 'data_root')}
    # The launcher's earlier revision had no precision/subsampling options.
    # Keep the original manifest hash, and explicitly mark inferred defaults.
    inferred = {}
    for key, value in [('gmm_dtype', 'original'), ('train_silhouette_samples', 0)]:
        if key not in effective:
            effective[key] = value
            inferred[key] = 'Behavior of the launcher revision before this option was added'
    record = dict(
        dataset=cfg['dataset'], gamma=cfg['gamma'], selected_epoch=int(selected.epoch),
        selection=metric, completed_epochs=epochs, csv_sha256=digest(files[0]),
        run_config_sha256=digest(path / 'run_config.json'),
        configuration=effective, inferred_legacy_defaults=inferred,
        metrics=selected_metrics,
    )
    return cfg, files[0].parent, record


def psnr(noisy_folder, clean_folder, noisy_epoch, clean_epoch):
    def atlas(folder, epoch):
        return folder / 'atlas_refit_means' / f'gmm_refit_means_atlas_epoch_{epoch:03d}.png'
    noisy_path, clean_path = atlas(noisy_folder, noisy_epoch), atlas(clean_folder, clean_epoch)
    noisy, clean = legacy_cells(noisy_path), legacy_cells(clean_path)
    cost = np.array([[float(np.sum((a-b)**2)) for b in clean] for a in noisy])
    rr, cc = linear_sum_assignment(cost)
    values = []
    for i, j in zip(rr, cc):
        mse = float(np.mean((noisy[i]-clean[j])**2))
        values.append(dict(noisy_cluster=int(i), clean_cluster=int(j),
                           psnr=99.0 if mse < 1e-10 else 10 * math.log10(1/mse)))
    return dict(clusters=values, mean=float(np.mean([v['psnr'] for v in values])),
                cell_shape=list(noisy[0].shape), clean_epoch=clean_epoch,
                noisy_atlas_sha256=digest(noisy_path), clean_atlas_sha256=digest(clean_path),
                reference='own independently trained clean-condition refit atlas',
                matching='Hungarian minimum squared pixel error', data_range=1)


def compare(value, target, metric):
    delta = value-target
    # PSNR was not assigned a predeclared acceptance threshold.
    threshold = {'accuracy': .03, 'ari': .05, 'nmi': .05, 'silhouette': .05}.get(metric)
    relative = abs(delta/target) if target else None
    close = abs(delta) <= threshold if threshold is not None else None
    if metric == 'davies_bouldin':
        close = relative <= .15
    return dict(fresh=value, paper=target, difference=delta,
                absolute_difference=abs(delta), relative_difference=relative,
                within_exploratory_band=close)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, action='append', required=True)
    p.add_argument('--clean', type=Path, action='append', default=[])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    targets = json.loads((root / 'paper_targets.json').read_text())['targets']
    clean = {}
    for path in args.clean:
        cfg, folder, rec = read_run(path)
        if cfg['gamma'] != 0 or cfg['dataset'] in clean:
            raise ValueError('Clean references must be unique gamma=0 runs')
        clean[cfg['dataset']] = (cfg, folder, rec)
    runs = []
    for path in args.run:
        cfg, folder, rec = read_run(path)
        target = next(t for t in targets if t['dataset'] == cfg['dataset'] and t['gamma'] == cfg['gamma'])
        rec['comparison'] = {
            k: compare(rec['metrics'].get(k, float('nan')), v,
                       k.removeprefix('test_stage_').removeprefix('test_refit_'))
            for k, v in target['metrics'].items()
        }
        if cfg['dataset'] in clean:
            ccfg, cfolder, crec = clean[cfg['dataset']]
            for key in ('seed', 'epochs', 'samples_per_class', 'gmm_dtype', 'workers'):
                if cfg.get(key) != ccfg.get(key):
                    raise ValueError(f'Clean/noisy configuration mismatch: {key}')
            rec['psnr'] = psnr(folder, cfolder, rec['selected_epoch'], crec['selected_epoch'])
        runs.append(rec)
    mri = []
    for gamma in sorted({r['gamma'] for r in runs}):
        part = [r for r in runs if r['gamma'] == gamma and r['dataset'].startswith('oasis_')]
        if len(part) != 3 or {r['dataset'] for r in part} != {'oasis_coronal', 'oasis_axial', 'oasis_sagittal'}:
            continue
        result = dict(gamma=gamma, views=3, comparison={})
        for metric in ('silhouette', 'davies_bouldin'):
            key = f'test_refit_{metric}'
            values = [r['metrics'].get(key, float('nan')) for r in part]
            if not np.isfinite(values).all():
                raise ValueError('A fresh MRI view has undefined clustering metrics; report collapse separately')
            expected = [r['comparison'][key]['paper'] for r in part]
            result['comparison'][metric] = compare(float(np.mean(values)), float(np.mean(expected)), metric)
        if all('psnr' in r for r in part):
            table = pd.read_csv(root / 'audit_results/main_table_crosscheck.csv')
            expected = table.loc[table.method.eq('Ours') & table.gamma.eq(gamma) & table.metric.eq('psnr')]
            if len(expected) == 1:
                result['comparison']['psnr'] = compare(float(np.mean([r['psnr']['mean'] for r in part])),
                                                       float(expected.iloc[0].computed), 'psnr')
        mri.append(result)
    report = dict(
        protocol='Recovered historical configuration; same-condition comparisons',
        scope='Only listed runs. Does not verify the seven-noise average or all paper claims.',
        selection_note='Digits/clothing use held-out refit labels for checkpoint selection, then report native scores.',
        psnr_note='Self-clean decoded prototypes, historical 128x127 montage crop; not patient-level ground truth.',
        runs=runs, clean_runs=[v[2] for v in clean.values()], mri_aggregates=mri,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(mri, indent=2))


if __name__ == '__main__':
    main()
