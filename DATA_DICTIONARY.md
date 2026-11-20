# Data dictionary and provenance

- UTF-8 CSV files use a comma delimiter; some include a UTF-8 BOM. Missing observations are blank/NaN, not zero.
- `reconstruction/data/*quality_checked_v2.csv`: original target concentrations and complete hydrometeorological predictors. `JDY=1` corresponds to 2023-01-01; the study starts at JDY 152. Preserve the supplied source units/conventions when using the original model code.
- `CIN_imputed_v2.csv`: `Date`, source `JDY`, measured targets, seven `*_Imputed` series and `*_WasImputed` flags (1 = originally missing). NH4, CODMn, DO, TP, TN, Chl_a are mg/L; Cond is µS/cm. Observed NH4 values of 0.025 are preserved in this release.
- `daily_reconstruction_error_ranges.csv`: daily point values, reference bounds, extreme-residual stress bounds, observed/imputed masks and event labels. These are exploratory marginal scenario envelopes, **not calibrated 95% PI/CI**.
- Validation prediction/mask files identify artificial masks, target, model, date, repeat, actual value and held-out prediction. Repeated mask pairs may overlap and are not independent sampling units.
- Water-quality observations are absent in August–October. Internal masks cannot validate those seasonal gaps.
- W2 CIN/CDT files retain the original W2 preamble/header and column order; parse them with the supplied builder. They are final concentration inputs, not a direct TN column. The TN-derived NOx conversion and fixed daily coefficients are given in `w2/protocol/`.
- `massbal.csv`: original W2 cumulative external fluxes and current inventories in kg N. Some data rows have an empty trailing field; the release parser handles it explicitly.
- `kflux_wb1.csv`: native interval-mean kinetic rates in kg N/day. NH4OM = aggregate water-column organic-N mineralization; NH4NITR = nitrification; NO3DEN = denitrification. NH4DOM and NH4POM are components of NH4OM.
- All analysis exports expressed in t N divide original kg N by 1000. `Net_external_tN = Fin_tN − Fout_tN`; `R_tN = Net_external_tN − DeltaM_tN`.
- `validation/data/W2_2025_observed_simulated.csv` supplies matched station observations and W2 concentrations; `results/paper_tables/W2_validation_2025_corrected_pairs.csv` is the long-format equivalent. TN validation is R² ≈ 0.742 (166 pairs), superseding the previous erroneous-data result.
- `provenance/source_map.csv` maps each copied scientific file to its source-relative location and original/released SHA-256. Portable path changes, output redirection, 61-day plot event definitions, and validation year-label corrections are recorded as adaptations. Scientific frozen data are not recalculated during packaging.
