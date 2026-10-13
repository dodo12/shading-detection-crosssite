"""Phase-1 feasibility checks on the Brazilian window features (not paper results): is each task observable from window-level means/ripple?
Split: chronological by cycle index within each file (first 70 % of cycles = train, last 30 % = test); cycle-grouped so the 6 segments of a cycle never straddle the split."""
import pandas as pd, numpy as np, json, os, warnings; warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); DATA = os.path.join(ROOT, 'data', 'derived'); RES = os.path.join(ROOT, 'results')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
import xgboost as xgb
W = pd.read_csv(os.path.join(DATA, 'ppgen_windows.csv')); W['cond'] = W.condition.str.replace('_csv', '')
W['cls'] = W.cond.str.extract(r'^\d+_(\w+?)(?:_Panel|$)')[0]; W['panel'] = W.cond.str.extract(r'Panel(\d)')[0].astype(float)
W = W[(W.G_mean > 5) & (W.T_mean > 5)].copy()
W['PG'] = W.P_mean / W.G_mean; W['VG'] = W.V_mean / W.G_mean; W['IG'] = W.I_mean / W.G_mean
ncyc = W.groupby('cond').cycle.transform('max') + 1; W['split'] = np.where(W.cycle < 0.7 * ncyc, 'train', 'test')
FE = {'G,T only': ['G_mean', 'T_mean'],
      'means (V,I,P)': ['V_mean', 'I_mean', 'P_mean'],
      'means + G,T': ['V_mean', 'I_mean', 'P_mean', 'G_mean', 'T_mean', 'PG'],
      'ripple only (std/min/max)': ['V_std', 'I_std', 'P_std', 'V_min', 'V_max', 'I_min', 'I_max'],
      'all window features': ['V_mean', 'V_std', 'V_min', 'V_max', 'I_mean', 'I_std', 'I_min', 'I_max', 'P_mean', 'P_std', 'G_mean', 'T_mean', 'PG', 'VG', 'IG']}
def run(sub, ycol, name):
    out = {}
    tr, te = sub[sub.split == 'train'], sub[sub.split == 'test']; classes = sorted(sub[ycol].unique()); chance = 1 / len(classes)
    for fn, cols in FE.items():
        res = {}
        for mn, mdl in (('logreg', make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                        ('xgb', xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=4))):
            y_tr = pd.Categorical(tr[ycol], categories=classes).codes; y_te = pd.Categorical(te[ycol], categories=classes).codes
            mdl.fit(tr[cols], y_tr); p = mdl.predict(te[cols]); res[mn] = dict(acc=float(accuracy_score(y_te, p)), bal_acc=float(balanced_accuracy_score(y_te, p)), macro_f1=float(f1_score(y_te, p, average='macro')))
        out[fn] = res
    print(f'\n== {name}  (n_train={len(tr)}, n_test={len(te)}, classes={len(classes)}, chance={chance:.3f})')
    for fn, r in out.items(): print(f'  {fn:28s} logreg bal-acc {r["logreg"]["bal_acc"]:.3f} | xgb bal-acc {r["xgb"]["bal_acc"]:.3f} (macro-F1 {r["xgb"]["macro_f1"]:.3f})')
    return dict(chance=chance, n_train=len(tr), n_test=len(te), results=out)
R = {}
R['5class_all'] = run(W, 'cls', '5 classes: Standard / Full / Partial / Short-circuit / Arc')
sh = W[W.cls.isin(['Standard', 'Full_Shading', 'Partial_shading'])]
R['3class_shading'] = run(sh, 'cls', '3 classes: Standard / Full shading / Partial shading')
b = sh.assign(y=(sh.cls != 'Standard').astype(int)); R['binary_shading'] = run(b, 'y', 'binary: shaded vs standard (class imbalance ~ 1:9)')
R['loc_full'] = run(W[W.cls == 'Full_Shading'], 'panel', 'LOCALISATION, full shading: which panel (1-6)')
R['loc_partial'] = run(W[W.cls == 'Partial_shading'], 'panel', 'LOCALISATION, partial shading: which panel (1-6)')
json.dump(R, open(os.path.join(RES, 'feasibility_localisation.json'), 'w'), indent=1)
# permutation-style check: are panel differences in simple means significant? (full shading, per-panel medians of P/G, V, I)
f = W[W.cls == 'Full_Shading']; print('\nFull shading per-panel medians (PG, V, I, Vstd, Istd):'); print(f.groupby('panel')[['PG', 'V_mean', 'I_mean', 'V_std', 'I_std']].median().round(3).to_string())
from scipy import stats
for c in ('PG', 'V_mean', 'I_mean', 'V_std', 'I_std'):
    H = stats.kruskal(*[g[c].values for _, g in f.groupby('panel')]); print(f'  Kruskal-Wallis across 6 panels, {c}: H={H.statistic:.1f}, p={H.pvalue:.3g}')
