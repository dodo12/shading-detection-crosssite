# Methods (summary)

## Site-specific healthy reference and normalised features

For each site, a reference describes healthy behaviour as a function of irradiance `G` (W m⁻²) and ambient temperature `Ta` (°C). It is fitted by ordinary least squares on **healthy training data of that site only**:

```
P_ref = G (a_P + b_P Ta + c_P G)
I_ref = G (a_I + b_I Ta + c_I G)
V_ref = d0 + d1 ln G + d2 Ta + d3 (ln G)^2
```

Measurements are expressed relative to the reference:

```
loss  = 1 - P / P_ref        V_rel = V / V_ref        I_rel = I / I_ref
```

The reference is empirical (physically motivated forms), not a single-diode model: the panel parameters and the circuit that sets the operating point are unknown for the Brazilian string.

## Detectors and feature sets

Logistic regression (class-balanced, standardised inputs), XGBoost (250 trees, depth 3, learning rate 0.08, subsample 0.8, column sample 0.9, positive-class weight) and a two-layer MLP (32 + 32 units, early stopping), all with seed 42 and fixed a priori. Feature sets: raw (`G`, `Ta`, `V`, `I`, `P`), normalised (`loss`, `V_rel`, `I_rel`), relative loss alone, and normalised + `G`, `Ta`.

## Data used and splits

- Both sites: irradiance ≥ 300 W m⁻².
- Brazil: classes standard, full shading, partial shading; 12,660 examples; the first 70% of the cycles of each file train, the last 30% test (chronological, cycle-blocked).
- La Réunion: hourly means of line 1; positives are hours with more than half of their records labelled as shading; shaded hours are scored by leave-fault-day-out (the day and its neighbours excluded from training); negatives are the healthy hours of January–April 2023.
- Cross-site: train on the whole of one site, test on the other, each site with its own reference.

## Metrics and statistics

ROC-AUC (primary), balanced accuracy at a 0.5 threshold (secondary). Uncertainty by block bootstrap (blocks of 5 cycles for Brazil, days for La Réunion; 400 resamples, 800 for paired differences). Six paired comparisons were chosen after inspecting the descriptive results and are therefore exploratory; Holm correction is applied. Domain gap: `AUC_within − AUC_cross`.

## Further analyses

Severity (full vs partial shading) by AUC of the relative loss; panel localisation with window means, ripple statistics and all window features (chance = 1/6); robustness to multiplicative noise (1–20%) on `G` and `Ta` and to missing irradiance (median imputation), without retraining; interpretability by exact tree SHAP of the XGBoost detector; model size and cost.
