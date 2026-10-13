"""
Audit and window-level feature extraction for PPGEN-SILVA2020 (Mendeley Data, doi:10.17632/g6m3pbh5kw.1).

Usage:   python audit_extract_ppgen.py  <folder_with_the_20_csv_files>  [output_folder]

What it does (nothing is assumed silently; every assumption is checked and reported):
  1. Detects a header line, counts rows per file, infers the window length (rows per cycle) from the
     row count, and checks it against 220 cycles.  Description says 9,960 samples per window, but
     99.96 ms x 100 kS/s = 9,996 = 6 x 1,666; the script reports which value the files actually follow.
  2. Checks columns (V, I, irradiance, ambient temperature): NaN counts, ranges, whether irradiance and
     temperature are constant inside a window.
  3. Cuts each cycle into 6 examples of 1,666 samples (the dataset's own definition) and writes one row of
     summary features per example (means, ripple, power) -> windows.csv  (small enough to upload).
  4. Writes audit_report.json with all checks.
No labels are invented: the condition label is the file name; map names to classes in the next step.
"""
import sys, os, json, glob
import numpy as np, pandas as pd

SEG = 1666; SEG_PER_CYCLE = 6; CYCLES_EXPECTED = 220
folder = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else 'ppgen_audit_out'; os.makedirs(out, exist_ok=True)
files = sorted(glob.glob(os.path.join(folder, '**', '*.csv'), recursive=True))
report = {'n_files': len(files), 'files': {}}; rows_out = []

def has_header(path):
    with open(path, 'r', errors='replace') as f: first = f.readline().strip()
    try: [float(x) for x in first.replace(';', ',').split(',')[:4]]; return False
    except ValueError: return True

for p in files:
    name = os.path.splitext(os.path.basename(p))[0]; hh = has_header(p)
    df = pd.read_csv(p, header=0 if hh else None, sep=None, engine='python')
    df = df.iloc[:, :4]; df.columns = ['V', 'I', 'G', 'T']; df = df.apply(pd.to_numeric, errors='coerce')
    n = len(df); info = dict(header=hh, rows=n, nan=df.isna().sum().to_dict(),
                             ranges={c: [float(df[c].min()), float(df[c].max())] for c in df.columns})
    win = None
    for w in (9996, 9960):
        if n % w == 0: win = w; info['rows_per_cycle_consistent_with'] = w; info['cycles'] = n // w; break
    if win is None: info['warning'] = f'row count {n} divisible by neither 9996 nor 9960'; win = SEG * SEG_PER_CYCLE
    ncyc = n // win; info['cycles_used'] = int(ncyc); info['cycles_match_220'] = bool(ncyc == CYCLES_EXPECTED)
    P = df['V'] * df['I']
    cyc_const = []
    for c in range(ncyc):
        blk = df.iloc[c * win:(c + 1) * win]; cyc_const.append((blk['G'].nunique() <= 5, blk['T'].nunique() <= 5))
        for s in range(SEG_PER_CYCLE):
            sl = slice(c * win + s * SEG, c * win + (s + 1) * SEG); b = df.iloc[sl]; pb = P.iloc[sl]
            if len(b) < SEG: continue
            v, i = b['V'].to_numpy(), b['I'].to_numpy()
            rows_out.append(dict(condition=name, cycle=c, segment=s, V_mean=v.mean(), V_std=v.std(), V_min=v.min(), V_max=v.max(),
                                 I_mean=i.mean(), I_std=i.std(), I_min=i.min(), I_max=i.max(), P_mean=pb.mean(), P_std=pb.std(),
                                 G_mean=b['G'].mean(), G_std=b['G'].std(), T_mean=b['T'].mean(), T_std=b['T'].std()))
    info['G_const_in_window_share'] = float(np.mean([a for a, _ in cyc_const])) if cyc_const else None
    info['T_const_in_window_share'] = float(np.mean([b_ for _, b_ in cyc_const])) if cyc_const else None
    report['files'][name] = info; print(name, n, 'rows;', info.get('cycles_used'), 'cycles')
W = pd.DataFrame(rows_out); W.to_csv(os.path.join(out, 'windows.csv'), index=False)
report['windows_written'] = int(len(W)); report['examples_per_condition'] = W.groupby('condition').size().to_dict() if len(W) else {}
json.dump(report, open(os.path.join(out, 'audit_report.json'), 'w'), indent=1, default=float)
print('done:', len(W), 'windows ->', out)
