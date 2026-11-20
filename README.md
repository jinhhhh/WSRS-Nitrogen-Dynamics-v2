# WSRS-Nitrogen-Dynamics — revised analysis (v2.0.0)

Code and data supporting **Event-scale reservoir nitrogen balance during water-sediment regulation in the Xiaolangdi Reservoir**.

Study period: 1 June 2023–31 May 2025. Four inclusive WSRS events comprise **61 days**:

| Event | Model days | Dates | Days |
|---|---|---|---:|
| E1 | 21–41 | 2023-06-21–2023-07-11 | 21 |
| E2 | 389–398 | 2024-06-23–2024-07-02 | 10 |
| E3 | 413–429 | 2024-07-17–2024-08-02 | 17 |
| E4 | 436–448 | 2024-08-09–2024-08-21 | 13 |

## Contents

- `temperature/`: original water-temperature reconstruction code, source predictors, and archived validation/reconstruction results.
- `reconstruction/`: water-quality source data, nested masking code, frozen v2 daily reconstructions, empirical residual/method ranges, held-out predictions, and mask manifests.
- `w2/scenarios/`: paired CIN/CDT inputs and returned mass balances for the **13 completed main scenarios**; B00 additionally contains native kinetic, flow, and flux outputs.
- `w2/code/`: scenario generation from the corrected, frozen boundary. The original generator also offers four optional scenarios; these were **not** included in the reported 13-scenario analysis.
- `validation/`: corrected 2025 observed/simulated station data and S12 plotting code.
- `analysis/`: portable reproduction of event budgets, native process aggregation, validation metrics, and key figures.
- `results/`: archived data tables used for the revised manuscript and supplementary material.
- `provenance/`: SHA-256 source map, environment, and release checks.

## Quick start

Use Python 3.10 or later, from the repository root:

```bash
python -m pip install -r requirements.txt
python analysis/reproduce_results.py
python analysis/verify_release.py
```

The first command after installation regenerates all 52 scenario/event budgets, B00 native-process diagnostics, corrected station-validation metrics, and selected figures. Results are written to `generated/`; archived inputs and results remain untouched. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for slower reconstruction and scenario-generation commands.

## Scope of this release

This release supports reconstruction and **postprocessing of the supplied W2 outputs**. The complete final W2 runtime project (current control/parameter, bathymetry, initial conditions, meteorological, flow and temperature inputs) will be supplied separately by the author. These files and the CE-QUAL-W2 executable are **not** included, and this release does not claim to reproduce the 13 hydrodynamic simulations from scratch. Do not substitute old V1 model inputs for the final project without checking them.

The archived B00 process results are model diagnostics. No additional box model is fitted. The 13 boundary alternatives are deterministic conditional scenarios, not a probability distribution or confidence interval. R = input − output − combined inventory change is an aggregate budget residual, not a direct nitrogen-removal estimate. The inventory comprises the reported waterbody, sediment, and plant nitrogen stocks.

E1–E3 have output exceeding input and declining inventory in all 13 scenarios. E4 inventory increases in all 13, whereas input exceeds output in 12 of 13. The process comparison is B00 only. See the supplied direction-summary table for the complete result.

## Archive and citation

Repository: https://github.com/jinhhhh/WSRS-Nitrogen-Dynamics-v2

Zenodo record family: https://doi.org/10.5281/zenodo.19640420

Revised version DOI: https://doi.org/10.5281/zenodo.24391730

Previous V1 archive: https://doi.org/10.5281/zenodo.19640421 . Its fitted box model and old water-quality reconstructions have been superseded by this release. The MIT license continues the existing archive's license; third-party CE-QUAL-W2 software is not redistributed here.
