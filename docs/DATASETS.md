# Datasets

## 1. PPGEN-SILVA2020 (Brazil)

- **Source:** Mendeley Data, version 1, <https://data.mendeley.com/datasets/g6m3pbh5kw/1>, doi:10.17632/g6m3pbh5kw.1.
- **Authors (as listed on the data page):** Wanderley Celeste, André Silva, Daniel José Custódio Coura, Hélder Roberto de Oliveira Rocha, Luís Otávio Rigo Jr., Leonardo José Silvestre. Published October 2020.
- **Licence:** see the Mendeley Data page of the dataset.
- **Content:** 20 CSV files, one per operating condition of a PV string of six panels: standard operation (1), full shading of panel 1–6 (6), partial shading of panel 1–6 (6), short circuit of panel 1–6 (6) and electric arc (1).
- **Columns (no header):** voltage, current, irradiance, ambient temperature.
- **Acquisition:** 100 kS/s, windows of 99.96 ms, repeated over 21 days.

### Audit results (this repository, `docs/SUPPLEMENTARY_TABLES.md`, table S1)

| Point | Finding |
|---|---|
| Window length | 9,996 samples (= 6 × 1,666), not 9,960 as in the description; each window is cut into six examples of 1,666 samples |
| Rows | 2,189,124 to 2,259,096 per file (45,011,988 in total; see table S1 for each file) |
| Cycles | 219 to 226 per file (the description gives 220) |
| Irradiance and temperature | one constant value per window (not time series) |
| Invalid values | 6 windows with G = 0 and T = 0 in `08_Partial_shading_Panel2` (removed) |
| Operating point | set by equipment that is not described in the accessible metadata (panel model, load or converter); tilt and site coordinates are not given |
| Time structure | consecutive windows of one condition are about two hours apart: no inter-window dynamics |

Analysis uses the three classes that bear on shading (standard, full shading, partial shading) and the examples with irradiance ≥ 300 W m⁻².

## 2. La Réunion rooftop plant

- **Article:** Graillet, A., Lebreton, C., Tang, C., Kbidi, F., Jegado, T., Damour, C., Benne, M. (2023). Electrical production data of a domestic grid-connected rooftop PV plant in normal and shading faults conditions associated with solar and meteorological data in a tropical climate. *Data in Brief* 46, 108723. <https://doi.org/10.1016/j.dib.2022.108723> (open access copy: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9800176/>).
- **Data:** Zenodo, <https://doi.org/10.5281/zenodo.7157424> (Creative Commons Attribution).
- **Related work on the same data:** Lebreton, C. et al. (2022), PV system failures diagnosis based on multiscale dispersion entropy, *Entropy* 24(9), 1311, <https://doi.org/10.3390/e24091311>.
- **Plant:** 4 kW rooftop, University of La Réunion (20.90126° S, 55.48422° E); two lines of twelve polycrystalline modules (line 1: TE1700 modules, shading experiments; line 2: TE1500 modules, healthy control), 26° tilt, facing north.
- **Records:** electrical quantities at 0.2 Hz from 1 October 2021; meteorological quantities at 1 Hz from 28 October 2021 (SPN1 pyranometer for global and diffuse tilted irradiance; type K thermocouple for ambient and type T thermocouple for module back-surface temperature). Time zone UTC+4.
- **Shading labels (line 1):** 0 healthy; 1 uniform shading; 2.1–2.3 constant shading of 1–3 whole modules; 3.1–3.2 constant shading of one or two thirds of a module; 4 intermittent partial shading.

### Hourly table used here

The analysis uses an hourly table (`data/derived/reunion_hourly_master.csv`) with, for each hour, means of the electrical and meteorological variables, the standard deviation, minimum and maximum of the 1 Hz irradiance, the record counts of each logger and the share of records carrying a shading label. It is a derived product of the open records; the aggregation step itself is not part of this repository (see `docs/LIMITATIONS.md`). After quality control (`src/prepare_reunion.py`), 2,712 hours with irradiance ≥ 300 W m⁻² remain, of which 80 shaded hours on 17 days (7 hours of whole-module shading, 10 of module-portion shading, 63 of intermittent shading).

## 3. How the two datasets differ (and why they are not merged)

| | Brazil | La Réunion |
|---|---|---|
| Nature | controlled experiment | real plant, months of operation |
| Configuration | one string of 6 panels | two lines of 12 modules |
| Resolution analysed | 1,666-sample window means | hourly means |
| Shaded element | known panel (1–6) | number of modules only |
| Labels | standard / full / partial (+ faults) | codes 0–4 on one line |

The two datasets are therefore used for **transfer evaluation in both directions**, not as one pooled training set.
