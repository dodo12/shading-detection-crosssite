"""Build the quality-controlled hourly La Réunion table used by the detection experiments.

Input : data/derived/reunion_hourly_master.csv  (hourly table derived from the open La Réunion records, Graillet et al. 2023)
Output: data/derived/reunion_hourly_qc.csv      (one row per hour with the QC flag, split and shading label)
        results/reunion_qc_log.csv               (hours retained after each sequential step)
Steps are applied in this order and are identical to those described in the manuscript:
  inverter data present -> >=90% inverter records -> >=90% meteorological coverage -> GTI >= 100 W m-2 -> V>100 V and I>0 (both lines)
  -> AC/DC ratio in [0.85, 1.10] (both lines) -> hours only partly labelled (0 < label share <= 50%) excluded
  -> specific yield P/G >= 50% of the line median (both lines).
Line 1 is the line subjected to the shading experiments; line 2 is never shaded.
"""
import os
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); DATA = os.path.join(ROOT, 'data', 'derived'); RES = os.path.join(ROOT, 'results'); os.makedirs(RES, exist_ok=True)
N_INV_MIN, COV_MIN, G_MODEL = 540, 90.0, 100.0
TRAIN_END, VAL_END = pd.Timestamp('2022-09-30'), pd.Timestamp('2022-12-31')

m = pd.read_csv(os.path.join(DATA, 'reunion_hourly_master.csv'), parse_dates=['hour_utc', 'hour_reunion'])
m['hour_utc'] = pd.to_datetime(m['hour_utc'], utc=True); m = m.set_index('hour_utc').sort_index()
m['date'] = pd.to_datetime(m['date_reunion'])
for k in (1, 2):
    m[f'I{k}'] = m[f'inv{k}_Ia_mean']; m[f'V{k}'] = m[f'inv{k}_Va_mean']; m[f'P{k}'] = m[f'I{k}'] * m[f'V{k}']   # DC power = mean current x mean voltage
    m[f'PAC{k}'] = m[f'inv{k}_Pg_mean']
m['G'] = m['GTI_mean']; m['Ta'] = m['TA_mean']
m['fault'] = m['fault_pct'].fillna(0.0); m['fault_code'] = m['fault_code_dominant']
m['is_fault'] = m['fault'] > 0.5                              # more than half of the hour's records carry a shading label
m['transition'] = (m['fault'] > 0) & (m['fault'] <= 0.5)       # partly labelled hours: excluded from both classes
log = []
def step(name, mask): log.append((name, int(mask.sum()))); return mask
ok = step('all hourly records', pd.Series(True, index=m.index))
ok = step('inverter data present', ok & (m['has_inverter'] == 1))
ok = step(f'inverter records >= 90% (n >= {N_INV_MIN})', ok & (m['inv1_n'] >= N_INV_MIN))
ok = step('meteorological coverage >= 90%', ok & (m['couverture_dt1_pct'] >= COV_MIN))
ok = step(f'GTI >= {G_MODEL:.0f} W m-2', ok & (m['G'] >= G_MODEL))
ok = step('V > 100 V and I > 0 (both lines)', ok & (m['V1'] > 100) & (m['V2'] > 100) & (m['I1'] > 0) & (m['I2'] > 0))
for k in (1, 2):
    r = m[f'PAC{k}'] / m[f'P{k}']; ok = step(f'line {k}: 0.85 <= AC/DC <= 1.10', ok & (r >= 0.85) & (r <= 1.10))
ok = step('exclude partly labelled hours', ok & ~m['transition'])
for k in (1, 2):
    y = m[f'P{k}'] / m['G']; med = y[ok & (m['G'] >= 300)].median(); ok = step(f'line {k}: specific yield P/G >= 50% of median', ok & (y >= 0.5 * med))
m['qc'] = ok
m['split'] = np.where(m['date'] <= TRAIN_END, 'train', np.where(m['date'] <= VAL_END, 'val', 'test'))
cols = ['date', 'split', 'G', 'Ta', 'V1', 'I1', 'P1', 'V2', 'I2', 'P2', 'is_fault', 'fault', 'fault_code', 'qc']
out = m[cols].reset_index(); out['hour_utc'] = out['hour_utc'].dt.strftime('%Y-%m-%d %H:%M:%S%z')
out.to_csv(os.path.join(DATA, 'reunion_hourly_qc.csv'), index=False)
pd.DataFrame(log, columns=['step', 'hours_retained']).to_csv(os.path.join(RES, 'reunion_qc_log.csv'), index=False)
print(pd.DataFrame(log, columns=['step', 'hours_retained']).to_string(index=False)); print('written', len(out), 'rows;', int(m['qc'].sum()), 'hours retained')
