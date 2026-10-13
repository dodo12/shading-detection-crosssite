# Reproducibility

## Environment
Python 3.12; package versions in `requirements.txt` (numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, scikit-learn 1.8.0, xgboost 3.4.1, matplotlib 3.10.8). All random seeds are fixed (42).

## Pipeline
```bash
make all
```
| Step | Command | Output |
|---|---|---|
| 1 | `python src/prepare_reunion.py` | `data/derived/reunion_hourly_qc.csv`, `results/reunion_qc_log.csv` |
| 2 | `python src/feasibility_localisation.py` | `results/feasibility_localisation.json` |
| 3 | `python src/detect_transfer.py` | `results/detect_transfer.json` |
| 4 | `python src/extra_analyses.py` | `results/extra_analyses.json` |
| 5 | `python src/make_figures.py` | `figures/*.png`, `figures/*.pdf`, `results/facts.json` |
| 6 | `python src/make_supplementary_tables.py` | `results/tables/*.csv`, `docs/SUPPLEMENTARY_TABLES.md` |
| 7 | `python tests/check_key_numbers.py` | pass/fail on the main numbers |

Step 3 must run before steps 4–6 (they read its output). Steps 2–4 take the derived inputs only; the raw third-party files are needed only to regenerate `ppgen_windows.csv` (`src/audit_extract_ppgen.py`).

## Regenerating the Brazilian window features from the raw files
Download the 20 CSV files from the Mendeley Data page into `data/raw/`, then:
```bash
python src/audit_extract_ppgen.py data/raw data/raw_out
cp data/raw_out/windows.csv data/derived/ppgen_windows.csv
cp data/raw_out/audit_report.json data/derived/ppgen_audit_report.json
```
The script checks and reports its structural assumptions (no header, four columns, 9,996 rows per cycle).

## Checks performed on this repository
- `prepare_reunion.py` reproduces exactly (same hours, same values, same splits and labels) the table that produced the reported results.
- A clean run of steps 2–4 reproduces all keys of the result files; only the model timings in `extra_analyses.json` vary between machines.

## Integrity
SHA-256 of the derived inputs are in `docs/CHECKSUMS.txt`.
