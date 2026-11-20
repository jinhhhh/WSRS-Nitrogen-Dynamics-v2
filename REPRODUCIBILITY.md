# Reproduction guide

Run commands from the repository root. `results/` and the archived reconstruction directories are frozen reference outputs, not temporary working directories.

| Task | Command | Destination |
|---|---|---|
| Recompute 52 event budgets, native process statistics, validation and key plots | `python analysis/reproduce_results.py` | `generated/` |
| Check archive hashes and numerical agreement | `python analysis/verify_release.py` | console and `generated/verification.json` |
| Original S12 plotting layout | `python validation/code/plot_S12.py` | `generated/validation/` |
| Rerun water-quality nested masking and point reconstruction | `python reconstruction/code/paper_validation_v2.py` | `reconstruction/regenerated/paper_package/` |
| Plot water-quality diagnostics from archived tables | `python analysis/plot_reconstruction.py` | `reconstruction/regenerated/` |
| Recompute exploratory residual/method ranges around archived v2 points | `python reconstruction/code/build_error_ranges.py` | `reconstruction/regenerated/error_ranges/` |
| Regenerate paired boundary scenarios | `python w2/code/build_scenarios.py` | `generated_scenarios/` and ZIP |
| Rerun the historical RF/LSTM temperature protocol | `python temperature/code/reconstruct_temperature.py` | `generated/temperature/` |

Install the extra dependencies listed in `temperature/requirements.txt` before the last command. That script retains its original RF/LSTM comparison and multiple-imputation settings, requires substantially more time, and is not part of the quick verification. Its interval outputs are temperature-protocol diagnostics, not the water-quality scenario bounds or the 13 W2 scenarios.

Water-quality models are Linear, PCHIP, ExtraTrees, and ET_bridge. Seeds/settings are retained in the released code and protocol JSON. The random-point PCHIP comparator and nested continuous-block validation serve different masking designs; good random-point performance does not establish skill in unobserved seasonal gaps. Existing observed values are retained. The fixed water-temperature predictor is included in the source data; its uncertainty is not propagated through the water-quality/W2 boundary scenarios.

Recalculated machine-learning predictions may depend on dependency versions/platform. The exact archived daily point series used to create the W2 boundary is `reconstruction/paper_package/tables/CIN_imputed_v2.csv`. The exact simulation inputs are the paired `cin_br1.csv` and `cdt_br1.csv` under each `w2/scenarios/<scenario>/`; use these for comparison with the archived outputs.

For inclusive event days a–b, interpolate cumulative W2 output to endpoints a and b+1 and take endpoint differences. Kinetic outputs are interval-mean kg N/day; integrate rate × interval overlap before aggregating calendar days. Only 728 complete days are common to the returned records (61 WSRS and 667 non-WSRS); the input calendar has 731 days. NH4OM is used once and is not added again to NH4DOM + NH4POM.

The author will upload the complete final W2 project separately. Until then, rerunning CE-QUAL-W2 itself and regenerating spatial output from scratch are outside this package's verified scope. No W2 executable, old box model, editorial scripts, reviewer reports, or unpublished manuscript files are included.
