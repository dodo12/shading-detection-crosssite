# Scope and known limitations of this repository

- **Third-party data.** The raw datasets are not redistributed. `data/derived/` holds small summary tables (window statistics for the Brazilian data, hourly aggregates for La Réunion). Cite the original datasets and check their licences when reusing them.
- **Hourly La Réunion table.** The step that aggregates the 0.2 Hz and 1 Hz records into hourly means is not included in this repository; the analysis starts from the hourly table.
- **Brazilian equipment.** The panel model, the load or converter that sets the operating point, the tilt and the location are not described in the accessible metadata, which limits the physical interpretation of voltage and current signatures.
- **Labelled episodes at La Réunion are few** (80 shaded hours on 17 days, strong shading, one period of 2022, one line); cross-site conclusions that depend on this site are correspondingly fragile.
- **Different scales.** A 100 kS/s window of a six-panel string and an hourly mean of a twelve-module line are not the same object, even after normalisation.
- **Empirical reference.** The healthy reference is statistical, not a single-diode model, and needs healthy data from the target site.
- **Exploratory comparisons.** The six paired comparisons were selected after inspecting descriptive results; Holm correction is applied but the set is exploratory.
- **Waveform-level analysis not performed.** `src/extract_ppgen_spectral.py` (spectral features and a raw-waveform subsample, for the panel-localisation question) was tested on a synthetic file of the documented layout only; no result in this repository depends on it. No temporal deep-learning model was evaluated, because neither dataset offers the temporal structure that would justify one.
- **Robustness figures** are for models trained on clean data and not retrained under perturbation.
