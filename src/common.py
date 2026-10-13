"""Shared data loading, site-specific healthy reference (normalisation) and evaluation helpers.
Imported by detect_transfer.py, extra_analyses.py, feasibility.py and make_figures.py."""
import sys, os, json, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
import xgboost as xgb
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); DATA = os.path.join(ROOT, 'data', 'derived'); RES = os.path.join(ROOT, 'results'); FIG = os.path.join(ROOT, 'figures')
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(42); GMIN = 300.0; B = 1000
# ------------------------------------------------------------------ Brazil
W = pd.read_csv(os.path.join(DATA, 'ppgen_windows.csv')); W['cond'] = W.condition.str.replace('_csv', '')
W['cls'] = W.cond.str.extract(r'^\d+_(\w+?)(?:_Panel|$)')[0]
W = W[(W.G_mean > 5) & (W.T_mean > 5) & W.cls.isin(['Standard', 'Full_Shading', 'Partial_shading'])].copy()
W = W.rename(columns={'G_mean': 'G', 'T_mean': 'Ta', 'V_mean': 'V', 'I_mean': 'I', 'P_mean': 'P'})
ncyc = W.groupby('cond').cycle.transform('max') + 1; W['split'] = np.where(W.cycle < 0.7 * ncyc, 'train', 'test'); W['y'] = (W.cls != 'Standard').astype(int)
# ------------------------------------------------------------------ Réunion (hourly, line 1 = shaded line)
D = pd.read_csv(os.path.join(DATA, 'reunion_hourly_qc.csv'), parse_dates=['date']); D = D.set_index(pd.to_datetime(D.hour_utc, utc=True)).sort_index()   # built by prepare_reunion.py
R = D[D.qc & (D.G >= GMIN)].copy(); R = R.rename(columns={'V1': 'V', 'I1': 'I', 'P1': 'P'}); R['y'] = R.is_fault.astype(int)
# ------------------------------------------------------------------ site-specific healthy baseline (identical functional forms)
def Xp(d): return np.c_[d.G, d.G * d.Ta, d.G ** 2]
def Xv(d): return np.c_[np.log(d.G), d.Ta, np.log(d.G) ** 2]
def fit_ref(h):
    return {k: LinearRegression().fit(f(h), h[k]) for k, f in (('P', Xp), ('I', Xp), ('V', Xv))}
def add_norm(d, ref):
    d = d.copy(); d['P_ref'] = ref['P'].predict(Xp(d)); d['V_ref'] = ref['V'].predict(Xv(d)); d['I_ref'] = ref['I'].predict(Xp(d))
    d['loss'] = 1 - d.P / d.P_ref; d['Vrel'] = d.V / d.V_ref; d['Irel'] = d.I / d.I_ref; return d
hb = W[(W.cls == 'Standard') & (W.split == 'train') & (W.G >= GMIN)]; refB = fit_ref(hb)
hr = R[(R.y == 0) & (R.split.isin(['train', 'val']))]; refR = fit_ref(hr)
Wn = add_norm(W[W.G >= GMIN], refB); Rn = add_norm(R, refR)
if __name__ == '__main__': print('Brazil windows G>=300:', len(Wn), '| shaded', int(Wn.y.sum()), '| Réunion hours G>=300:', len(Rn), '| labelled', int(Rn.y.sum()))
FS = {'raw (G,T,V,I,P)': ['G', 'Ta', 'V', 'I', 'P'], 'physics-normalised (loss,Vrel,Irel)': ['loss', 'Vrel', 'Irel'], 'loss only': ['loss'],
      'normalised + G,T': ['loss', 'Vrel', 'Irel', 'G', 'Ta']}
def mk(name):
    if name == 'logreg': return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced'))
    if name == 'mlp': return make_pipeline(StandardScaler(), MLPClassifier((32, 32), max_iter=600, early_stopping=True, random_state=42))
    return xgb.XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.08, subsample=0.8, colsample_bytree=0.9, random_state=42, n_jobs=4)
def fitpred(name, cols, tr, te):
    m = mk(name); w = None
    if name == 'xgb': w = np.where(tr.y == 1, (tr.y == 0).sum() / max(tr.y.sum(), 1), 1.0)
    m.fit(tr[cols], tr.y, **({'sample_weight': w} if name == 'xgb' else {})); return m.predict_proba(te[cols])[:, 1]
def metrics(y, s): return dict(auc=float(roc_auc_score(y, s)), ap=float(average_precision_score(y, s)), prev=float(np.mean(y)), bal=float(balanced_accuracy_score(y, s > 0.5)))
def block_boot(te, s, key, fn, n=B):
    ids = te[key].to_numpy(); u = np.unique(ids); idx = {k: np.where(ids == k)[0] for k in u}; y = te.y.to_numpy(); out = []
    for _ in range(n):
        ii = np.concatenate([idx[k] for k in rng.choice(u, len(u))])
        if y[ii].min() == y[ii].max(): continue
        out.append(fn(y[ii], s[ii]))
    return out
Wn['blk'] = Wn.cycle // 5; Rn['blk'] = Rn.date.astype(str)

