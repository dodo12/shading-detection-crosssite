"""Experiments A-G: within-site detection, leave-fault-day-out (Réunion), cross-site transfer in both directions, paired comparison, SHAP, robustness."""
from common import *
res = {}
# ---------- Exp A: within Brazil (chronological, cycle-blocked); one model x feature set grid
trB, teB = Wn[Wn.split == 'train'], Wn[Wn.split == 'test']; A = {}
for mn in ('logreg', 'xgb', 'mlp'):
    for fn, cols in FS.items():
        s = fitpred(mn, cols, trB, teB); m = metrics(teB.y.to_numpy(), s); bs = block_boot(teB, s, 'blk', lambda y, p: roc_auc_score(y, p), 400)
        A[f'{mn} | {fn}'] = dict(m, auc_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
res['A_within_brazil'] = A; res['A_n'] = dict(train=len(trB), test=len(teB), test_pos=int(teB.y.sum()))
# ---------- Exp B: within Réunion (chronological: train/val healthy+... -> shading hours are in the training period, so use leave-fault-day-out)
fd = sorted(Rn.loc[Rn.y == 1, 'date'].unique()); Bres = {}
for mn in ('logreg', 'xgb'):
    for fn, cols in FS.items():
        S = pd.Series(np.nan, index=Rn.index)
        for d_ in fd:
            ex = (Rn.date >= d_ - pd.Timedelta(days=1)) & (Rn.date <= d_ + pd.Timedelta(days=1)); tr = Rn[~ex & (Rn.split != 'test')]; te = Rn[Rn.date == d_]
            S[te.index] = fitpred(mn, cols, tr, te)
        te = Rn[S.notna() | (Rn.split == 'test')]; trf = Rn[(Rn.split != 'test') & ~Rn.date.isin(fd)]
        mfull = fitpred(mn, cols, Rn[Rn.split != 'test'], Rn[Rn.split == 'test']); S[Rn.split == 'test'] = mfull
        ev = Rn[S.notna()]; s = S[S.notna()].to_numpy(); m = metrics(ev.y.to_numpy(), s); bs = block_boot(ev, s, 'blk', lambda y, p: roc_auc_score(y, p), 400)
        Bres[f'{mn} | {fn}'] = dict(m, auc_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=len(ev))
res['B_within_reunion'] = Bres
# ---------- Exp C/D: cross-site transfer
def cross(tr, te, key, tag):
    out = {}
    for mn in ('logreg', 'xgb'):
        for fn, cols in FS.items():
            s = fitpred(mn, cols, tr, te); m = metrics(te.y.to_numpy(), s); bs = block_boot(te, s, key, lambda y, p: roc_auc_score(y, p), 400)
            out[f'{mn} | {fn}'] = dict(m, auc_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], thr_pos_rate=float((s > 0.5).mean()))
    return out
teR = Rn[(Rn.y == 1) | (Rn.split == 'test')]
res['C_brazil_to_reunion'] = cross(Wn, teR, 'blk', 'B->R'); res['C_n'] = dict(test_pos=int(teR.y.sum()), test_neg=int((teR.y == 0).sum()))
trR = Rn[Rn.split != 'test']; res['D_reunion_to_brazil'] = cross(trR, Wn, 'blk', 'R->B'); res['D_n'] = dict(train=len(trR), train_pos=int(trR.y.sum()))
# ---------- domain gap and paired bootstrap physics vs raw (cross-site, XGB)
def paired(tr, te, key, colsA, colsB, mn='xgb', n=600):
    sa, sb = fitpred(mn, colsA, tr, te), fitpred(mn, colsB, tr, te); ids = te[key].to_numpy(); u = np.unique(ids); idx = {k: np.where(ids == k)[0] for k in u}; y = te.y.to_numpy(); d = []
    for _ in range(n):
        ii = np.concatenate([idx[k] for k in rng.choice(u, len(u))])
        if y[ii].min() == y[ii].max(): continue
        d.append(roc_auc_score(y[ii], sa[ii]) - roc_auc_score(y[ii], sb[ii]))
    obs = roc_auc_score(y, sa) - roc_auc_score(y, sb); return dict(diff=float(obs), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
nf, rf = FS['physics-normalised (loss,Vrel,Irel)'], FS['raw (G,T,V,I,P)']
res['E_paired_physics_vs_raw'] = {'brazil_to_reunion': paired(Wn, teR, 'blk', nf, rf), 'reunion_to_brazil': paired(trR, Wn, 'blk', nf, rf)}
# ---------- feature relevance within Brazil (xgb gain + mean |SHAP|), normalised set + G,T
cols = FS['normalised + G,T']; m = mk('xgb'); m.fit(trB[cols], trB.y, sample_weight=np.where(trB.y == 1, (trB.y == 0).sum() / trB.y.sum(), 1.0))
contrib = m.get_booster().predict(xgb.DMatrix(teB[cols]), pred_contribs=True)[:, :-1]; res['F_mean_abs_shap_brazil'] = dict(zip(cols, np.abs(contrib).mean(0).astype(float)))
# ---------- robustness: multiplicative noise on G and Ta, and missing G (median imputation) at 1/5/10/20 %, within Brazil, physics-normalised vs raw (xgb)
def perturb(d, kind, lvl):
    d = d.copy(); n = len(d)
    if kind == 'noise': d['G'] = d.G * (1 + rng.normal(0, lvl, n)); d['Ta'] = d.Ta * (1 + rng.normal(0, lvl, n))
    else: miss = rng.random(n) < lvl; d.loc[miss, 'G'] = trB.G.median()
    d = add_norm(d.assign(G=d.G.clip(lower=1)), refB); return d
rob = {}
for kind in ('noise', 'missing'):
    for lvl in (0.0, 0.01, 0.05, 0.10, 0.20):
        t = perturb(teB, kind, lvl); row = {}
        for fn in ('raw (G,T,V,I,P)', 'physics-normalised (loss,Vrel,Irel)', 'normalised + G,T'):
            cols = FS[fn]; s = fitpred('xgb', cols, trB, t); row[fn] = float(roc_auc_score(t.y, s))
        rob[f'{kind}_{lvl}'] = row
res['G_robustness_brazil_auc'] = rob
json.dump(res, open(os.path.join(RES, 'detect_transfer.json'), 'w'), indent=1, default=float)
def show(title, d):
    print('\n' + title)
    for k, v in d.items(): print(f'  {k:52s} AUC {v["auc"]:.3f} [{v["auc_ci"][0]:.3f},{v["auc_ci"][1]:.3f}]  AP {v["ap"]:.3f} (prev {v["prev"]:.2f})  bal-acc {v["bal"]:.3f}')
show('A. within Brazil (chronological, n=%s)' % res['A_n'], A); show('B. within Réunion (leave-fault-day-out; test = Jan-Apr 2023)', Bres)
show('C. TRAIN Brazil -> TEST Réunion %s' % res['C_n'], res['C_brazil_to_reunion']); show('D. TRAIN Réunion -> TEST Brazil %s' % res['D_n'], res['D_reunion_to_brazil'])
print('\nE. paired AUC difference (physics-normalised - raw), xgb:', json.dumps(res['E_paired_physics_vs_raw']))
print('F. mean |SHAP| (Brazil, xgb):', {k: round(v, 3) for k, v in res['F_mean_abs_shap_brazil'].items()})
print('G. robustness (XGB AUC):'); [print('  ', k, {a: round(b, 3) for a, b in v.items()}) for k, v in rob.items()]
