"""
PPGEN-SILVA2020, second pass: waveform-level features for the shaded-panel LOCALISATION question.

Usage: python extract_ppgen_spectral.py <folder_with_the_20_csv_files> [output_folder] [raw_segments_per_condition=250]

Outputs (small enough to upload):
  windows_spectral.csv   one row per 1,666-sample example: AC RMS, skewness, kurtosis, crest factor of V and I,
                         V-I AC correlation, dominant frequency/amplitude, and log10 power in 16 log-spaced bands
                         (60 Hz .. 50 kHz) of V and I  (fs = 100 kS/s assumed from the dataset description).
  raw_segments.npz       a SUBSAMPLE of raw segments (float16): X[n,1666,2] (V,I), plus condition, cycle, segment, G, T.
                         Used only for a deep-learning check on a few hundred segments per condition.
Same layout assumptions as the first script (no header, 4 columns V,I,G,T, 9,996 rows per cycle); they are verified and printed.
"""
import sys, os, glob, json
import numpy as np, pandas as pd
from scipy import stats, signal

FS = 100_000; SEG = 1666; SPC = 6; WIN = SEG * SPC
folder = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else 'ppgen_spectral_out'; K = int(sys.argv[3]) if len(sys.argv) > 3 else 250
os.makedirs(out, exist_ok=True)
edges = np.logspace(np.log10(60), np.log10(50_000), 17)
freqs = np.fft.rfftfreq(SEG, 1 / FS); hann = signal.windows.hann(SEG)
def band_feats(x, tag):
    xc = x - x.mean(); P = np.abs(np.fft.rfft(xc * hann)) ** 2 / (hann ** 2).sum()
    d = {f'{tag}_band{b:02d}': np.log10(P[(freqs >= lo) & (freqs < hi)].sum() + 1e-12) for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))}
    k = int(np.argmax(P[1:])) + 1; d[f'{tag}_f_dom'] = float(freqs[k]); d[f'{tag}_a_dom'] = float(np.sqrt(P[k])); d[f'{tag}_ac_rms'] = float(xc.std())
    d[f'{tag}_skew'] = float(stats.skew(xc)); d[f'{tag}_kurt'] = float(stats.kurtosis(xc)); d[f'{tag}_crest'] = float(np.abs(xc).max() / (xc.std() + 1e-12))
    return d
files = sorted(glob.glob(os.path.join(folder, '**', '*.csv'), recursive=True)); rows = []; raw = dict(X=[], cond=[], cycle=[], seg=[], G=[], T=[])
for p in files:
    name = os.path.splitext(os.path.basename(p))[0].replace('_csv', '')
    df = pd.read_csv(p, header=None, sep=None, engine='python', comment=None)
    try: df = df.apply(pd.to_numeric)
    except Exception: df = pd.read_csv(p, header=0, sep=None, engine='python').apply(pd.to_numeric)
    A = df.iloc[:, :4].to_numpy(dtype=np.float64); n = len(A); win = 9996 if n % 9996 == 0 else (9960 if n % 9960 == 0 else WIN); ncyc = n // win
    pick = set(np.unique(np.linspace(0, ncyc - 1, min(K // SPC + 1, ncyc)).astype(int)).tolist())
    for c in range(ncyc):
        for s in range(SPC):
            seg = A[c * win + s * SEG: c * win + (s + 1) * SEG]
            if len(seg) < SEG or seg[:, 2].mean() <= 5 or seg[:, 3].mean() <= 5: continue
            v, i = seg[:, 0], seg[:, 1]; d = dict(condition=name, cycle=c, segment=s, G=float(seg[:, 2].mean()), T=float(seg[:, 3].mean()))
            d.update(band_feats(v, 'V')); d.update(band_feats(i, 'I')); d['VI_ac_corr'] = float(np.corrcoef(v - v.mean(), i - i.mean())[0, 1]) if v.std() > 0 and i.std() > 0 else 0.0
            rows.append(d)
            if c in pick:
                raw['X'].append(np.stack([v, i], 1).astype(np.float16)); raw['cond'].append(name); raw['cycle'].append(c); raw['seg'].append(s); raw['G'].append(d['G']); raw['T'].append(d['T'])
    print(name, n, 'rows;', ncyc, 'cycles;', len(rows), 'cumulative examples', flush=True)
pd.DataFrame(rows).to_csv(os.path.join(out, 'windows_spectral.csv'), index=False)
np.savez_compressed(os.path.join(out, 'raw_segments.npz'), X=np.stack(raw['X']), cond=np.array(raw['cond']), cycle=np.array(raw['cycle']), seg=np.array(raw['seg']), G=np.array(raw['G'], dtype=np.float32), T=np.array(raw['T'], dtype=np.float32))
print('done:', len(rows), 'examples;', len(raw['cond']), 'raw segments ->', out)
