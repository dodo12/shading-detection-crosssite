# Physics-normalised detection of PV partial shading across two tropical datasets

Supplementary material (code, derived data, results and figures) for a study on shading detection in photovoltaic (PV) systems. The study asks whether normalising electrical measurements by a **site-specific healthy reference** makes shading detectors transferable between two experimentally different tropical datasets:

| Dataset | Nature | Link |
|---|---|---|
| **PPGEN-SILVA2020** (Brazil) | Controlled experiment, 6-panel string, 100 kS/s windows, shading of each panel | [Mendeley Data, doi:10.17632/g6m3pbh5kw.1](https://data.mendeley.com/datasets/g6m3pbh5kw/1) |
| **La Réunion rooftop plant** | Real 4 kW plant, two lines, labelled shading on one line | [Data in Brief, doi:10.1016/j.dib.2022.108723](https://doi.org/10.1016/j.dib.2022.108723) · [Zenodo, doi:10.5281/zenodo.7157424](https://doi.org/10.5281/zenodo.7157424) |

Full descriptions, licences and the audit of both datasets are in [`docs/DATASETS.md`](docs/DATASETS.md).

## What is in this repository

```
├── src/                     analysis code (Python)
├── data/
│   ├── raw/                 (empty) place the original third-party files here if you want to regenerate the derived data
│   └── derived/             small derived tables used by the analysis (see docs/DATA_DICTIONARY.md)
├── results/                 JSON results, supplementary tables (CSV), logs
├── figures/                 all figures, PNG (300 dpi) and PDF
├── docs/                    datasets, methods, data dictionary, reproducibility, limitations, supplementary tables
├── tests/check_key_numbers.py   checks that the main numbers are reproduced
├── Makefile · requirements.txt · LICENSE · CITATION.cff
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
make all          # prepare data -> analyses -> figures -> tables -> check
```

Without `make`, run in this order: `src/prepare_reunion.py`, `src/feasibility_localisation.py`, `src/detect_transfer.py`, `src/extra_analyses.py`, `src/make_figures.py`, `src/make_supplementary_tables.py`, `tests/check_key_numbers.py`. Details and expected outputs: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Scripts

| Script | Purpose |
|---|---|
| `src/audit_extract_ppgen.py` | Run locally on the 20 raw Brazilian CSV files: audits row counts, cycles, ranges, and writes the window-level features (`ppgen_windows.csv`) and an audit report |
| `src/extract_ppgen_spectral.py` | Optional second pass on the raw Brazilian files: spectral features and a raw-waveform subsample for the localisation question (not used for the reported results) |
| `src/prepare_reunion.py` | Applies the quality control to the hourly La Réunion table and writes `reunion_hourly_qc.csv` |
| `src/common.py` | Data loading, site-specific healthy reference (normalisation), models and evaluation helpers |
| `src/feasibility_localisation.py` | Window-level classification and panel localisation (chronological split) |
| `src/detect_transfer.py` | Within-site detection, cross-site transfer in both directions, paired comparisons, SHAP, robustness |
| `src/extra_analyses.py` | Feature signatures by class, severity, Holm-corrected comparisons, domain gap, model cost |
| `src/make_figures.py` | All figures |
| `src/make_supplementary_tables.py` | Supplementary tables (`docs/SUPPLEMENTARY_TABLES.md`, `results/tables/*.csv`) |

## Main results reproduced by the code

| Quantity | Value |
|---|---|
| Within-site detection AUC (Brazil / La Réunion, all models and feature sets) | 0.935–0.986 / 0.952–0.989 |
| XGBoost on raw variables, Brazil → La Réunion / La Réunion → Brazil (AUC) | 0.497 / 0.472 |
| XGBoost on normalised features, same directions (AUC) | 0.974 / 0.915 |
| Paired AUC gain from normalisation (95% CI, Holm-adjusted p) | 0.476 [0.395, 0.575], 0.007 / 0.444 [0.406, 0.481], 0.007 |
| Full vs partial shading, relative loss alone (AUC) | 0.991 |
| Panel localisation from window-level features (balanced accuracy; chance = 0.167) | 0.14–0.19 |

The sample sizes behind these numbers are small at La Réunion (80 shaded hours on 17 days); see [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Data licences and attribution

The code is under the MIT licence; figures and documentation under CC BY 4.0. **The data are third-party**: the original raw files are not redistributed here. `data/derived/` contains only small summary tables computed from them, provided for reproducibility. Please cite the original datasets (see `docs/DATASETS.md`) whenever you use these derived tables, and check the licence of each source.

## Contact

Corresponding author: ssteyve@ltsa-univ-dla.cm — Laboratory of Technology and Applied Sciences, University of Douala, Cameroon.
