"""Audit the original result archive against the accepted paper.

This checks archived evidence, separately from fresh checkpoint evaluation.
The source archive and MRI images are private inputs and are not bundled.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch

METHODS = {'v1.1':'VaDE','v1.2':'Contra-VaDE','v1.3':'LDDPM-VaDE','v4.0':'t-VaDE-log',
           'v4.0a':'t-VaDE+EM','v4.1a':'t-VaDE+2Ph','v4.1':'Ours'}
VIEWS = ['exp16_oasis_coronal_128','exp17_oasis_axial_128','exp18_oasis_sagittal_128']


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def legacy_cells(path):
    """Preserve the historical K=2 montage crop; not full 128x128 PSNR."""
    a = np.asarray(Image.open(path).convert('L'), dtype=np.float32)/255
    h,w = a.shape
    cells = [a[2:h-2, round(i*w/2)+2:round((i+1)*w/2)-2] for i in range(2)]
    hh,ww = min(x.shape[0] for x in cells),min(x.shape[1] for x in cells)
    return [x[:hh,:ww] for x in cells]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--paper',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    root,out=args.source_root.resolve(),args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    best=pd.read_csv(root/'paper_results/tables/best_epoch_all.csv')
    best['gamma']=pd.to_numeric(best.noise_tag.str.extract(r'^cauchy_g([0-9.]+)$',expand=False),errors='coerce')
    paper=args.paper.read_text()
    sources=[]; raw_checks=[]
    for row in best.itertuples():
        run=root/row.exp/'robustness_eval/results'/f'{row.algo}_{row.noise_tag}'
        paths=sorted(run.glob('metrics_*.csv'))
        if len(paths)!=1:
            raise RuntimeError(f'Ambiguous raw metrics: {run}')
        path=paths[0];raw=pd.read_csv(path)
        eligible=raw.loc[raw.epoch.gt(int(.25*raw.epoch.max()))]
        selected=eligible.loc[eligible.test_refit_accuracy.idxmax() if row.criterion_used=='acc'
                              else eligible.test_loss_recon.idxmin()]
        same_epoch=int(selected.epoch)==row.best_epoch
        keys=[k for k in best.columns if k.startswith('test_') and k in raw]
        same_metrics=all(np.isclose(selected[k],getattr(row,k),equal_nan=True,atol=1e-10,rtol=0) for k in keys)
        assert same_epoch and same_metrics,(path,row.best_epoch,selected.epoch)
        raw_checks.append(dict(exp=row.exp,algo=row.algo,noise_tag=row.noise_tag,
                               epoch=int(selected.epoch),metrics_verified=len(keys),matches=True))
        sources.append(dict(path=str(path.relative_to(root)),sha256=digest(path)))
    pd.DataFrame(raw_checks).to_csv(out/'raw_log_checks.csv',index=False)
    pd.DataFrame(sources).to_csv(out/'raw_log_sources.csv',index=False)
    # The model-native (stage) scores are reported at the refit-selected epoch.
    clustering=[]
    for dataset,exp,target in [('MNIST','exp2_gamma7','0.798'),('Fashion-MNIST','exp12_fashion_mnist','0.555')]:
        for algo,method in METHODS.items():
            part=best.loc[best.exp.eq(exp)&best.algo.eq(algo)&best.gamma.gt(0)&best.gamma.le(.14)]
            assert len(part)==7
            entry=dict(dataset=dataset,algo=algo,method=method,conditions=7,
                       selection='max test_refit_accuracy after 25% warmup',reported_metric='stage')
            for metric in ['accuracy','ari','nmi']:
                entry[f'mean_stage_{metric}']=float(part[f'test_stage_{metric}'].mean())
                entry[f'mean_refit_{metric}_not_reported']=float(part[f'test_refit_{metric}'].mean())
            if algo=='v4.1':
                assert target in paper and f"{entry['mean_stage_accuracy']:.3f}"==target
                entry['paper_ACC']=target
            clustering.append(entry)
    pd.DataFrame(clustering).to_csv(out/'clustering_stage_metrics.csv',index=False)
    # Recompute all PSNRs used by main Table 1 from the atlas PNGs.
    psnr=[]
    for exp in VIEWS:
        archive=pd.read_csv(root/f'paper_results/tables/atlas_metrics_{exp}.csv')
        for row in best.loc[best.exp.eq(exp)&best.gamma.isin([.02,.08,.12])].itertuples():
            selected=archive.loc[archive.algo.eq(row.algo)&archive.noise_tag.eq(row.noise_tag)
                                 &archive.epoch.eq(row.best_epoch)&archive.ref_scheme.eq('self')].sort_values('cluster_id_noisy')
            assert len(selected)==2
            rec=selected.iloc[0]
            base=root/exp/'robustness_eval/results'
            noisy_path=base/f'{row.algo}_{row.noise_tag}'/'atlas_refit_means'/f'gmm_refit_means_atlas_epoch_{int(row.best_epoch):03d}.png'
            clean_path=base/f'{row.algo}_{rec.ref_noise_tag}'/'atlas_refit_means'/f'gmm_refit_means_atlas_epoch_{int(rec.ref_epoch):03d}.png'
            noisy,clean=legacy_cells(noisy_path),legacy_cells(clean_path)
            cost=np.array([[float(np.sum((a-b)**2)) for b in clean] for a in noisy])
            rr,cc=linear_sum_assignment(cost)
            pi=dict(zip(rr,cc))
            for s in selected.itertuples():
                i=int(s.cluster_id_noisy);j=pi[i]
                assert j==s.cluster_id_ref_matched
                mse=float(np.mean((noisy[i]-clean[j])**2))
                computed=99 if mse<1e-10 else 10*math.log10(1/mse)
                assert abs(computed-s.psnr_cluster)<1e-4
                psnr.append(dict(exp=exp,algo=row.algo,gamma=row.gamma,epoch=row.best_epoch,cluster=i,
                                 clean_cluster=int(j),psnr=float(s.psnr_cluster),recomputed_psnr=computed,
                                 included=bool(np.isfinite(row.test_refit_silhouette)),
                                 noisy_sha256=digest(noisy_path),clean_sha256=digest(clean_path)))
    psnr=pd.DataFrame(psnr)
    psnr.to_csv(out/'psnr_pixel_recomputation.csv',index=False)
    cells=[]
    table=paper.split(r'\label{tab:main}',1)[1].split(r'\end{table}',1)[0]
    aliases={v:k for k,v in METHODS.items()}
    for line in table.splitlines():
        if line.lstrip().startswith('%') or '&' not in line:continue
        fields=line.split('&')
        label=re.sub(r'\\textbf\{([^}]+)\}',r'\1',fields[0].strip().replace('$',''))
        if label not in aliases:continue
        algo=aliases[label]
        for n,gamma in enumerate([0,.02,.08,.12]):
            values=best.loc[best.exp.isin(VIEWS)&best.algo.eq(algo)&best.gamma.eq(gamma)]
            assert len(values)==3
            valid=values.loc[values.test_refit_silhouette.notna()]
            for k,metric in enumerate(['silhouette','davies_bouldin','psnr']):
                if metric=='psnr' and gamma==0:continue
                value=(psnr.loc[psnr.algo.eq(algo)&psnr.gamma.eq(gamma)&psnr.included]
                       .groupby('exp').psnr.mean().mean() if metric=='psnr' else valid[f'test_refit_{metric}'].mean())
                printed=re.search(r'\d+\.\d+',fields[1+3*n+k])[0]
                digits=2 if metric=='psnr' else 3
                match=f'{value:.{digits}f}'==printed
                assert match,(label,gamma,metric,value,printed)
                cells.append(dict(method=label,gamma=gamma,metric=metric,computed=float(value),paper=printed,matches=match))
        part=best.loc[best.algo.eq(algo)]
        count=int((part.test_refit_silhouette.isna()|part.test_refit_size_entropy.lt(.1)).sum())
        rate=count/len(part)*100
        for index,metric,value in [(13,'collapse_count',count),(14,'collapse_rate_percent',rate)]:
            printed=re.search(r'\d+(?:\.\d+)?',fields[index])[0]
            match=(str(value)==printed if metric=='collapse_count' else f'{value:.1f}'==printed)
            cells.append(dict(method=label,metric=metric,computed=value,paper=printed,matches=match))
    pd.DataFrame(cells).to_csv(out/'main_table_crosscheck.csv',index=False)
    weights=[]
    for row in best.loc[best.algo.eq('v4.1')&best.exp.isin(['exp2_gamma7','exp12_fashion_mnist']+VIEWS)&best.gamma.notna()].itertuples():
        path=root/row.exp/'robustness_eval/results'/f'v4.1_{row.noise_tag}'/'best_vae.pth'
        state=torch.load(path,map_location='cpu',weights_only=True)
        counts=sorted({int(v) for k,v in state.items() if k.endswith('num_batches_tracked')})
        steps=7 if row.exp in VIEWS else 1500
        inferred=[v/steps for v in counts]
        weights.append(dict(exp=row.exp,gamma=row.gamma,table_epoch=row.best_epoch,
                            inferred_checkpoint_epoch=inferred[0],counter_values=counts,
                            matches_selected_epoch=len(inferred)==1 and inferred[0]==row.best_epoch,
                            checkpoint_sha256=digest(path)))
    pd.DataFrame(weights).to_csv(out/'checkpoint_epoch_audit.csv',index=False)
    report=dict(paper_path=str(args.paper),paper_sha256=digest(args.paper),raw_logs_verified=len(raw_checks),
                table_cells_verified=sum(x['matches'] for x in cells),table_cells_checked=len(cells),
                psnr_cluster_images_recomputed=len(psnr),
                stage_ACC_targets_recovered=True,
                prior_audit_correction='0.798/0.555 are native stage scores, not refit GMM scores; earlier audit compared the wrong columns',
                checkpoint_epoch_matches=sum(x['matches_selected_epoch'] for x in weights),checkpoints_checked=len(weights),
                checkpoint_epoch_inference='Batchnorm counters / training batches per epoch, assuming unchanged batch size and workflow',
                verified_scope='Archive aggregation and per-image PSNR recomputation. Fresh checkpoint evaluation reported separately.',
                full_paper_reproduction_passed=False,
                unresolved=['Manuscript Adam / 50 pretraining epochs / 500 iterations differs from archived AdamW / 100 epochs / no pretraining',
                            'Selected-epoch checkpoints and historical stochastic-noise RNG states are not all available',
                            'Historical PSNR uses 128x127 cropped montage cells against method-specific clean atlases'])
    (out/'artifact_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
