"""Signatures by class, severity (full vs partial), Holm-corrected paired comparisons, domain gap, model cost."""
import json, time, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, os
from common import *
from scipy import stats
out = {}
# 1. signature signs: median Vrel, Irel, loss by class and site
sig = {}
for site, d in (('Brazil', Wn), ('Reunion', Rn)):
    for y, nm in ((0, 'healthy'), (1, 'shaded')):
        g = d[d.y == y]; sig[f'{site}_{nm}'] = dict(n=int(len(g)), loss=float(g.loss.median()), Vrel=float(g.Vrel.median()), Irel=float(g.Irel.median()))
out['signature'] = sig
for k, v in sig.items(): print(k, {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()})
# 2. severity (Brazil): full vs partial shading, AUC of the relative loss alone (block bootstrap by cycle)
sv = Wn[Wn.cls.isin(['Full_Shading', 'Partial_shading'])].copy(); sv['y'] = (sv.cls == 'Full_Shading').astype(int)
a = roc_auc_score(sv.y, sv.loss); bs = block_boot(sv, sv.loss.to_numpy(), 'blk', lambda y, p: roc_auc_score(y, p), 400)
out['severity_loss_auc'] = dict(auc=float(a), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], median_loss_full=float(sv[sv.y == 1].loss.median()), median_loss_partial=float(sv[sv.y == 0].loss.median()),
                                  n_full=int(sv.y.sum()), n_partial=int((1 - sv.y).sum())); print('severity', out['severity_loss_auc'])
trB, teB = Wn[Wn.split == 'train'], Wn[Wn.split == 'test']; teR = Rn[(Rn.y == 1) | (Rn.split == 'test')]; trR = Rn[Rn.split != 'test']
# 3. Holm-corrected paired comparisons (XGB, AUC): physics-normalised vs raw, in both transfer directions; and within-site logistic vs xgb (normalised)
def pboot(tr, te, key, colsA, colsB, mA='xgb', mB='xgb', n=800):
    sa, sb = fitpred(mA, colsA, tr, te), fitpred(mB, colsB, tr, te); ids = te[key].to_numpy(); u = np.unique(ids); idx = {k: np.where(ids == k)[0] for k in u}; y = te.y.to_numpy(); d = []
    for _ in range(n):
        ii = np.concatenate([idx[k] for k in rng.choice(u, len(u))])
        if y[ii].min() == y[ii].max(): continue
        d.append(roc_auc_score(y[ii], sa[ii]) - roc_auc_score(y[ii], sb[ii]))
    obs = roc_auc_score(y, sa) - roc_auc_score(y, sb); p = min(1.0, 2 * min(np.mean(np.array(d) >= 0), np.mean(np.array(d) <= 0)))
    return dict(diff=float(obs), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], p=float(max(p, 1 / n)))
nf, rf = FS['physics-normalised (loss,Vrel,Irel)'], FS['raw (G,T,V,I,P)']; lo = FS['loss only']
cmp = {'B->R xgb: normalised vs raw': pboot(Wn, teR, 'blk', nf, rf), 'R->B xgb: normalised vs raw': pboot(trR, Wn, 'blk', nf, rf),
       'B->R xgb: loss-only vs normalised(3)': pboot(Wn, teR, 'blk', lo, nf), 'R->B logreg: loss-only vs normalised(3)': pboot(trR, Wn, 'blk', lo, nf, 'logreg', 'logreg'),
       'within-Brazil: xgb normalised vs logreg raw': pboot(trB, teB, 'blk', nf, rf, 'xgb', 'logreg'), 'within-Brazil: xgb normalised+GT vs xgb raw': pboot(trB, teB, 'blk', FS['normalised + G,T'], rf)}
ps = np.array([v['p'] for v in cmp.values()]); order = np.argsort(ps); holm = np.empty_like(ps); m = len(ps); run_max = 0
for rank, i in enumerate(order): run_max = max(run_max, min(1.0, ps[i] * (m - rank))); holm[i] = run_max
for (k, v), h in zip(cmp.items(), holm): v['p_holm'] = float(h); print(k, {a: (round(b, 3) if isinstance(b, float) else [round(c, 3) for c in b]) for a, b in v.items()})
out['paired_holm'] = cmp
# 4. domain gap and cost
r = json.load(open(os.path.join(RES, 'detect_transfer.json')))
out['domain_gap'] = {'xgb normalised': dict(within_brazil=r['A_within_brazil']['xgb | physics-normalised (loss,Vrel,Irel)']['auc'], brazil_to_reunion=r['C_brazil_to_reunion']['xgb | physics-normalised (loss,Vrel,Irel)']['auc'],
                                               within_reunion=r['B_within_reunion']['xgb | physics-normalised (loss,Vrel,Irel)']['auc'], reunion_to_brazil=r['D_reunion_to_brazil']['xgb | physics-normalised (loss,Vrel,Irel)']['auc'])}
cost = {}
for mn in ('logreg', 'xgb', 'mlp'):
    t = time.time(); mod = mk(mn); cols = FS['normalised + G,T']
    mod.fit(trB[cols], trB.y, **({'sample_weight': np.where(trB.y == 1, (trB.y == 0).sum() / trB.y.sum(), 1.0)} if mn == 'xgb' else {})); tt = time.time() - t
    t = time.time(); mod.predict_proba(teB[cols]); ti = (time.time() - t) / len(teB) * 1e6
    if mn == 'logreg': npar = len(cols) + 1
    elif mn == 'mlp': mm = mod[-1]; npar = int(sum(c.size for c in mm.coefs_) + sum(b.size for b in mm.intercepts_))
    else: npar = int(mod.get_booster().trees_to_dataframe().shape[0])
    cost[mn] = dict(params_or_nodes=npar, train_s=float(tt), infer_us_per_sample=float(ti))
out['cost'] = cost; print('cost', cost)
json.dump(out, open(os.path.join(RES, 'extra_analyses.json'), 'w'), indent=1, default=float)
