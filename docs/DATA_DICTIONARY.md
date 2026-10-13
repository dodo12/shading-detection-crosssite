# Data dictionary (`data/derived/`)

## `ppgen_windows.csv` — one row per 1,666-sample example of the Brazilian dataset
Produced by `src/audit_extract_ppgen.py` from the 20 raw CSV files. Each cycle of 9,996 rows gives six examples.

| Column | Meaning |
|---|---|
| `condition` | file name of the operating condition (e.g. `01_Full_Shading_Panel1_csv`) |
| `cycle` | cycle index inside the file (0-based) |
| `segment` | example index inside the cycle (0–5) |
| `V_mean`, `V_std`, `V_min`, `V_max` | voltage statistics over the 1,666 samples (V) |
| `I_mean`, `I_std`, `I_min`, `I_max` | current statistics (A) |
| `P_mean`, `P_std` | mean and standard deviation of the instantaneous product V × I (W) |
| `G_mean`, `G_std` | irradiance (W m⁻²); constant inside a window |
| `T_mean`, `T_std` | ambient temperature (°C); constant inside a window |

## `ppgen_audit_report.json` — per-file audit (rows, cycles, ranges, NaN counts, constancy of G and T inside windows). See table S1.

## `reunion_hourly_master.csv` — hourly table derived from the open La Réunion records
Hours are stamped by their start in UTC (`hour_utc`); `hour_reunion` is local time (UTC+4).

| Column | Meaning |
|---|---|
| `hour_utc`, `hour_reunion`, `date_reunion`, `month` | time stamps and calendar fields |
| `n_dt1`, `couverture_dt1_pct`, `flag_couverture_dt1_faible` | number of one-second meteorological records in the hour, coverage (%) and low-coverage flag |
| `GTI_mean/std/min/max`, `GTI_range_intrah` | global tilted irradiance statistics within the hour (W m⁻²) |
| `DTI_mean` | diffuse tilted irradiance (W m⁻²) |
| `TA_mean`, `TA_std`; `TPV_mean`, `TPV_std` | ambient and back-surface module temperature (°C) |
| `inv{1,2}_n` | number of inverter records in the hour |
| `inv{1,2}_Pg_mean/min/max` | AC power injected into the grid (W) |
| `inv{1,2}_Ia_mean`, `inv{1,2}_Va_mean` | PV-side current (A) and voltage (V) |
| `inv{1,2}_Ig_mean`, `inv{1,2}_Vg_mean`, `inv{1,2}_Fg_mean`, `inv{1,2}_Eg_mean` | grid current, voltage, frequency and injected energy |
| `fault_pct`, `fault_code_dominant`, `fault_label` | share of the hour's records carrying a shading label, dominant label code and its text (line 1) |
| `has_inverter` | 1 if inverter data exist for the hour |
| `Pg_total_*`, `Pg_range_intrah`, `PR_proxy`, `is_daytime` | derived summaries provided with the table |

## `reunion_hourly_qc.csv` — output of `src/prepare_reunion.py`

| Column | Meaning |
|---|---|
| `hour_utc`, `date` | start of the hour (UTC) and local calendar date |
| `split` | `train` (to 30 Sep 2022), `val` (to 31 Dec 2022) or `test` |
| `G`, `Ta` | irradiance and ambient temperature used by the analysis |
| `V1`, `I1`, `P1`; `V2`, `I2`, `P2` | line voltage (V), current (A) and DC power (W) = mean current × mean voltage |
| `is_fault`, `fault`, `fault_code` | shading label of line 1 (more than half of the hour labelled), label share, dominant code |
| `qc` | passes all quality-control steps (`results/reunion_qc_log.csv`) |
