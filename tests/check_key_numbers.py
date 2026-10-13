"""Smoke test: the main reported numbers must be reproduced (tolerance 1e-3). Run after `make analysis`."""
import json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); RES = os.path.join(ROOT, 'results')
E = json.load(open(f'{RES}/detect_transfer.json')); X = json.load(open(f'{RES}/extra_analyses.json')); F = json.load(open(f'{RES}/facts.json'))
RAW, NRM = 'raw (G,T,V,I,P)', 'physics-normalised (loss,Vrel,Irel)'
expected = {
    'Brazil examples with G>=300': (F['brazil_G300'], 12660),
    'Réunion shaded hours (G>=300)': (F['reunion_pos'], 80),
    'XGBoost raw, Brazil->Réunion AUC': (E['C_brazil_to_reunion'][f'xgb | {RAW}']['auc'], 0.497),
    'XGBoost normalised, Brazil->Réunion AUC': (E['C_brazil_to_reunion'][f'xgb | {NRM}']['auc'], 0.974),
    'XGBoost raw, Réunion->Brazil AUC': (E['D_reunion_to_brazil'][f'xgb | {RAW}']['auc'], 0.472),
    'XGBoost normalised, Réunion->Brazil AUC': (E['D_reunion_to_brazil'][f'xgb | {NRM}']['auc'], 0.915),
    'Severity AUC (full vs partial)': (X['severity_loss_auc']['auc'], 0.991),
    'Paired dAUC B->R (normalised - raw)': (X['paired_holm']['B->R xgb: normalised vs raw']['diff'], 0.476),
}
bad = [(k, a, b) for k, (a, b) in expected.items() if abs(a - b) > 1e-3]
for k, (a, b) in expected.items(): print(f'{"OK " if abs(a - b) <= 1e-3 else "FAIL"} {k}: {a:.4f} (expected {b})')
sys.exit(1 if bad else 0)
