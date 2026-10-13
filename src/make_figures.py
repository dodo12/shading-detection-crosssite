import json, os, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from common import *
from figstyle import *
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
trB, teB = Wn[Wn.split == 'train'], Wn[Wn.split == 'test']; teR = Rn[(Rn.y == 1) | (Rn.split == 'test')]; trR = Rn[Rn.split != 'test']
E = json.load(open(os.path.join(RES, 'detect_transfer.json'))); X = json.load(open(os.path.join(RES, 'extra_analyses.json'))); FJ = json.load(open(os.path.join(RES, 'feasibility_localisation.json'))); AR = json.load(open(os.path.join(DATA, 'ppgen_audit_report.json')))
facts = {}
# ---------------- facts
W0 = pd.read_csv(os.path.join(DATA, 'ppgen_windows.csv')); W0['cond'] = W0.condition.str.replace('_csv', ''); W0['cls'] = W0.cond.str.extract(r'^\d+_(\w+?)(?:_Panel|$)')[0]
facts['brazil_windows_total'] = int(len(W0)); facts['brazil_invalid'] = int(((W0.G_mean <= 5) | (W0.T_mean <= 5)).sum())
facts['brazil_rows_total'] = int(sum(v['rows'] for v in AR['files'].values())); facts['brazil_cycles_min'] = int(min(v['cycles'] for v in AR['files'].values())); facts['brazil_cycles_max'] = int(max(v['cycles'] for v in AR['files'].values()))
facts['brazil_class_counts_G300'] = Wn.groupby('cls').size().to_dict(); facts['brazil_G300'] = int(len(Wn)); facts['brazil_std_cal'] = int(len(hb)); facts['reunion_cal'] = int(len(hr))
facts['brazil_train'] = int(len(trB)); facts['brazil_test'] = int(len(teB)); facts['reunion_test_neg'] = int((teR.y == 0).sum()); facts['reunion_pos'] = int(teR.y.sum()); facts['reunion_pos_days'] = int(Rn.loc[Rn.y == 1, 'date'].nunique())
fc = Rn[Rn.y == 1].assign(g=lambda d: d.fault_code.map(lambda c: '2.x' if int(c) == 2 else ('3.x' if int(c) == 3 else '4'))); facts['reunion_pos_by_type'] = fc.groupby('g').size().to_dict()
sig = {}
for site, d in (('Brazil', Wn), ('Reunion', Rn)):
    for nm, g in ([('healthy', d[d.y == 0])] + [('shaded', d[d.y == 1])] + ([('full', d[d.cls == 'Full_Shading']), ('partial', d[d.cls == 'Partial_shading'])] if site == 'Brazil' else [('2.x', fc[fc.g == '2.x']), ('3.x', fc[fc.g == '3.x']), ('4', fc[fc.g == '4'])])):
        sig[f'{site}|{nm}'] = dict(n=int(len(g)), loss=float(g.loss.median()), Vrel=float(g.Vrel.median()), Irel=float(g.Irel.median()))
facts['signature'] = sig
json.dump(facts, open(os.path.join(RES, 'facts.json'), 'w'), indent=1, default=float); print(json.dumps(facts, indent=0, default=float)[:1800])
# ---------------- Fig 2: dataset overview
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.5))
axs[0].hist(Wn.G, bins=30, color=C['ml'], alpha=0.8); axs[0].set_xlabel('G (W m$^{-2}$), Brazil windows'); axs[0].set_ylabel('Windows (G ≥ 300)'); panel(axs[0], 'a')
axs[1].hist(Rn.G, bins=30, color=C['hyb'], alpha=0.8); axs[1].set_xlabel('GTI (W m$^{-2}$), Réunion hours'); axs[1].set_ylabel('Hours (G ≥ 300)'); panel(axs[1], 'b')
cl = ['Standard', 'Full_Shading', 'Partial_shading']; cn = [int((Wn.cls == c).sum()) for c in cl]; axs[2].bar(['Brazil\nstandard', 'Brazil\nfull', 'Brazil\npartial', 'Réunion\nhealthy', 'Réunion\nshaded'], cn + [int((Rn.y == 0).sum()), int((Rn.y == 1).sum())], color=[C['grey'], C['fault'], C['phys'], C['grey'], C['fault']]); axs[2].set_yscale('log'); axs[2].set_ylabel('Samples (log scale)'); axs[2].tick_params(axis='x', labelsize=6.3); panel(axs[2], 'c')
fig.tight_layout(); save(fig, 'fig02_datasets')
# ---------------- Fig 3: signatures
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.9)); groups = [('Brazil\nstandard', Wn[Wn.cls == 'Standard'], C['grey']), ('Brazil\nfull', Wn[Wn.cls == 'Full_Shading'], C['fault']), ('Brazil\npartial', Wn[Wn.cls == 'Partial_shading'], C['phys']), ('Réunion\nhealthy', Rn[Rn.y == 0], C['grey']), ('Réunion\nshaded', Rn[Rn.y == 1], C['fault'])]
for ax, (col, lab, s) in zip(axs, (('loss', 'Relative loss ℓ (–)', 'a'), ('Vrel', 'Relative voltage V$_{rel}$ (–)', 'b'), ('Irel', 'Relative current I$_{rel}$ (–)', 'c'))):
    bp = ax.boxplot([g[col].values for _, g, _ in groups], labels=[n for n, _, _ in groups], showfliers=False, patch_artist=True, medianprops=dict(color='k'))
    for p_, (_, _, c) in zip(bp['boxes'], groups): p_.set_facecolor(c); p_.set_alpha(0.75)
    ax.axhline(1.0 if col != 'loss' else 0.0, color='k', lw=0.5, ls=':'); ax.set_ylabel(lab); ax.tick_params(axis='x', labelsize=6.2); panel(ax, s)
fig.tight_layout(); save(fig, 'fig03_signatures')
# ---------------- Fig 4: within-site AUC
def barauc(ax, d, models, fsets, labs, cols):
    w = 0.8 / len(fsets)
    for k, (fs, lb, c) in enumerate(zip(fsets, labs, cols)):
        v = [d[f'{m} | {fs}']['auc'] for m in models]; lo = [d[f'{m} | {fs}']['auc'] - d[f'{m} | {fs}']['auc_ci'][0] for m in models]; hi = [d[f'{m} | {fs}']['auc_ci'][1] - d[f'{m} | {fs}']['auc'] for m in models]
        ax.bar(np.arange(len(models)) + k * w - 0.4 + w / 2, v, w, yerr=[lo, hi], capsize=1.5, color=c, label=lb, error_kw=dict(lw=0.6))
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models); ax.set_ylabel('ROC-AUC'); ax.set_ylim(0.4, 1.02)
FSN = ['raw (G,T,V,I,P)', 'physics-normalised (loss,Vrel,Irel)', 'loss only', 'normalised + G,T']; FL = ['raw', 'normalised', 'loss only', 'normalised + G,T']; FC = [C['grey'], C['hyb'], C['phys'], C['ml']]
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7), sharey=True)
barauc(axs[0], E['A_within_brazil'], ['logreg', 'xgb', 'mlp'], FSN, FL, FC); axs[0].set_title('Within Brazil (cycle-blocked chronological split)', fontsize=7.3); panel(axs[0], 'a'); axs[0].legend(frameon=False, fontsize=6, ncol=2, loc='lower left')
barauc(axs[1], E['B_within_reunion'], ['logreg', 'xgb'], FSN, FL, FC); axs[1].set_title('Within Réunion (leave-fault-day-out)', fontsize=7.3); panel(axs[1], 'b'); axs[1].set_ylabel('')
fig.tight_layout(); save(fig, 'fig04_within_site')
# ---------------- Fig 5: cross-site
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7), sharey=True)
barauc(axs[0], E['C_brazil_to_reunion'], ['logreg', 'xgb'], FSN[:3], FL[:3], FC[:3]); axs[0].set_title('Train Brazil → test Réunion', fontsize=7.3); panel(axs[0], 'a'); axs[0].axhline(0.5, color='k', lw=0.5, ls=':')
barauc(axs[1], E['D_reunion_to_brazil'], ['logreg', 'xgb'], FSN[:3], FL[:3], FC[:3]); axs[1].set_title('Train Réunion → test Brazil', fontsize=7.3); panel(axs[1], 'b'); axs[1].set_ylabel(''); axs[1].axhline(0.5, color='k', lw=0.5, ls=':'); axs[1].legend(frameon=False, fontsize=6.3, loc='lower right')
fig.tight_layout(); save(fig, 'fig05_cross_site')
# ---------------- Fig 6: severity
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw=dict(width_ratios=[1, 1.1]))
sv = Wn[Wn.cls.isin(['Full_Shading', 'Partial_shading'])]
axs[0].hist(sv[sv.cls == 'Partial_shading'].loss * 100, bins=40, alpha=0.7, color=C['phys'], label='Partial'); axs[0].hist(sv[sv.cls == 'Full_Shading'].loss * 100, bins=40, alpha=0.7, color=C['fault'], label='Full'); axs[0].set_xlabel('Relative loss ℓ (%), Brazil'); axs[0].set_ylabel('Windows'); axs[0].legend(frameon=False); panel(axs[0], 'a')
bp = axs[1].boxplot([fc[fc.g == g].loss.values * 100 for g in ('2.x', '3.x', '4')], labels=['Whole modules\n(2.x)', 'Module portion\n(3.x)', 'Intermittent\n(4)'], patch_artist=True, medianprops=dict(color='k'))
for p_ in bp['boxes']: p_.set_facecolor(C['fault']); p_.set_alpha(0.75)
axs[1].set_ylabel('Relative loss ℓ (%), Réunion'); axs[1].tick_params(axis='x', labelsize=6.5); panel(axs[1], 'b'); fig.tight_layout(); save(fig, 'fig06_severity')
# ---------------- Fig 7: localisation
fig, ax = plt.subplots(figsize=(W1 * 1.5, 2.6)); labs = []; vals = []
for k, lb in (('loc_full', 'Full shading'), ('loc_partial', 'Partial shading')):
    for fs in ('G,T only', 'means + G,T', 'ripple only (std/min/max)', 'all window features'):
        for mdl in ('logreg', 'xgb'): pass
    r = FJ[k]['results']; labs.append(lb); vals.append([r[fs]['xgb']['bal_acc'] for fs in ('G,T only', 'means + G,T', 'ripple only (std/min/max)', 'all window features')])
xx = np.arange(4); ax.bar(xx - 0.2, vals[0], 0.4, color=C['fault'], label='Full shading'); ax.bar(xx + 0.2, vals[1], 0.4, color=C['phys'], label='Partial shading'); ax.axhline(1 / 6, color='k', ls='--', lw=0.8, label='Chance (1/6)')
ax.set_xticks(xx); ax.set_xticklabels(['G, T', 'Means +\nG, T', 'Ripple', 'All window\nfeatures'], fontsize=6.8); ax.set_ylabel('Balanced accuracy (6 panels)'); ax.set_ylim(0, 0.5); ax.legend(frameon=False, fontsize=6.5); save(fig, 'fig07_localisation')
# ---------------- Fig 8: robustness
rob = E['G_robustness_brazil_auc']; lv = [0.0, 0.01, 0.05, 0.10, 0.20]; fig, axs = plt.subplots(1, 2, figsize=(W2, 2.5), sharey=True)
for ax, kind, ttl, s in ((axs[0], 'noise', 'Multiplicative noise on G and T$_a$', 'a'), (axs[1], 'missing', 'Missing irradiance (median imputation)', 'b')):
    for fs, lb, c in zip(['raw (G,T,V,I,P)', 'physics-normalised (loss,Vrel,Irel)', 'normalised + G,T'], ['raw', 'normalised', 'normalised + G,T'], [C['grey'], C['hyb'], C['ml']]):
        ax.plot([l * 100 for l in lv], [rob[f'{kind}_{l}'][fs] for l in lv], 'o-', ms=3, color=c, label=lb)
    ax.set_xlabel('Perturbation level (%)'); ax.set_title(ttl, fontsize=7.3); panel(ax, s)
axs[0].set_ylabel('ROC-AUC (XGBoost, within Brazil)'); axs[0].legend(frameon=False, fontsize=6.5); fig.tight_layout(); save(fig, 'fig08_robustness')
# ---------------- Fig 9: SHAP
sh = E['F_mean_abs_shap_brazil']; o = sorted(sh, key=sh.get); lab = {'loss': 'Relative loss ℓ', 'Vrel': 'V$_{rel}$', 'Irel': 'I$_{rel}$', 'G': 'G', 'Ta': 'T$_a$'}
fig, ax = plt.subplots(figsize=(W1 * 1.3, 2.3)); ax.barh([lab[k] for k in o], [sh[k] for k in o], color=C['ml']); ax.set_xlabel('Mean |SHAP| (log-odds), XGBoost, within Brazil'); save(fig, 'fig09_shap')
# ---------------- Fig 1: workflow
def box(ax, x, y, w, h, t, fc='#F2F2F2', fs=7, bold=False): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06', fc=fc, ec='#444', lw=0.8)); ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', linespacing=1.25)
def arr(ax, p, q): ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=8, lw=0.9, color='#444'))
fig, ax = plt.subplots(figsize=(W2, 3.4)); ax.set_xlim(0, 14); ax.set_ylim(0, 6.8); ax.axis('off')
box(ax, 0.1, 4.6, 3.4, 1.8, 'Brazil: PPGEN-SILVA2020\n6-panel string, 100 kS/s windows\nlabels: standard / full / partial\nshading of panel 1-6 (+ faults)', fc='#E8F1FA', fs=6.6)
box(ax, 0.1, 0.5, 3.4, 1.8, 'La Réunion: 4 kW rooftop plant\nhourly means (0.2 Hz, 1 Hz source)\nshading labels on line 1,\nline 2 never shaded', fc='#E5F5EC', fs=6.6)
box(ax, 4.3, 2.5, 3.0, 1.8, 'Site-specific healthy\nreference (fit on healthy\ntraining data only):\nP$_{ref}$, I$_{ref}$, V$_{ref}$ (G, T$_a$)', fc='#FCE9C7', fs=6.6)
box(ax, 8.1, 2.5, 2.4, 1.8, 'Normalised features\nℓ = 1 − P/P$_{ref}$\nV$_{rel}$, I$_{rel}$', fc='#FCE9C7', fs=6.8, bold=True)
box(ax, 11.2, 4.6, 2.7, 1.8, 'Detectors\nlogistic, XGBoost, MLP', fc='#D5EFE6', fs=6.8)
box(ax, 11.2, 2.5, 2.7, 1.5, 'Within-site and\ncross-site transfer\n(both directions)', fc='#F2F2F2', fs=6.6)
box(ax, 11.2, 0.5, 2.7, 1.6, 'Severity, localisation,\nablation, robustness,\ninterpretability', fc='#F2F2F2', fs=6.6)
for p, q in [((3.5, 5.5), (4.3, 3.9)), ((3.5, 1.4), (4.3, 2.9)), ((7.3, 3.4), (8.1, 3.4)), ((10.5, 3.7), (11.2, 5.3)), ((12.55, 4.6), (12.55, 4.0)), ((12.55, 2.5), (12.55, 2.1))]: arr(ax, p, q)
fig.savefig(os.path.join(FIG, 'fig01_workflow.png'), bbox_inches='tight'); fig.savefig(os.path.join(FIG, 'fig01_workflow.pdf'), bbox_inches='tight'); plt.close(fig)
print('figures done')
