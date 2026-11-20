"""Check archive hashes, frozen observations, event totals and manuscript metrics."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'generated'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    checks={}
    manifest=ROOT/'provenance/SHA256SUMS.csv'
    if manifest.exists():
        d=pd.read_csv(manifest)
        bad=[r.File for r in d.itertuples() if not (ROOT/r.File).is_file() or sha(ROOT/r.File)!=r.SHA256]
        assert not bad, bad
        checks['archive_files_verified']=len(d)
    frozen=pd.read_csv(ROOT/'reconstruction/paper_package/tables/CIN_imputed_v2.csv')
    for target in ['NH4','CODMn','Cond','DO','TP','TN','Chl_a']:
        mask=frozen[target].notna()
        assert np.array_equal(frozen.loc[mask,target],frozen.loc[mask,target+'_Imputed'])
        assert frozen[target+'_Imputed'].notna().all()
    checks['observations_preserved']=True
    got=pd.read_csv(OUT/'budgets/all_13_event_budgets.csv').set_index(['Scenario','Event']).sort_index()
    ref=pd.read_csv(ROOT/'results/paper_tables/Supp_Table_all_13_event_budgets.csv').set_index(['Scenario','Event']).sort_index()
    assert got.index.equals(ref.index) and len(got)==52
    for key in ['Fin_tN','Fout_tN','DeltaM_tN','R_tN','R_over_Fin_percent']:
        # Archived publication tables round large event totals to 0.001 t N.
        assert np.allclose(got[key],ref[key],atol=5.1e-4,rtol=1e-9),key
    checks['all_52_event_budgets_match']=True
    checks['maximum_budget_table_rounding_difference_tN']=float((got[['Fin_tN','Fout_tN','DeltaM_tN','R_tN']]-ref[['Fin_tN','Fout_tN','DeltaM_tN','R_tN']]).abs().max().max())
    directions=pd.read_csv(OUT/'budgets/event_direction_summary.csv').set_index('Event')
    assert directions.loc[['E1','E2','E3'],'Output_exceeds_input'].eq(13).all()
    assert directions.loc[['E1','E2','E3'],'Inventory_decrease'].eq(13).all()
    assert directions.loc['E4','Input_exceeds_output']==12 and directions.loc['E4','Inventory_increase']==13
    checks['event_directions_match']=True
    stats=pd.read_csv(OUT/'native_processes/WSRS_nonWSRS_comparison.csv').set_index('Process')
    ref=pd.read_csv(ROOT/'results/paper_tables/Native_process_WSRS_nonWSRS_comparison.csv').set_index('Process')
    for key in ['Non_WSRS_mean_tN_day','WSRS_mean_tN_day','Mean_difference_percent']:
        assert np.allclose(stats[key],ref.loc[stats.index,key],atol=1e-6,rtol=1e-8),key
    checks['native_process_comparison_matches']=True
    native=pd.read_csv(OUT/'native_flux_audit/daily_native_N_diagnostics.csv')
    assert len(native)==728 and native.Event.ne('non_WSRS').sum()==61
    checks.update(complete_days=728,WSRS_days=61,non_WSRS_days=667)
    scores=pd.read_csv(OUT/'validation/corrected_2025_metrics.csv').set_index('Variable')
    ref=pd.read_csv(ROOT/'results/paper_tables/W2_validation_2025_corrected_metrics.csv').set_index('Variable')
    for key in ['N','R2','RMSE','MAE','PBIAS_percent_sim_minus_obs']:
        assert np.allclose(scores[key],ref.loc[scores.index,key],atol=1e-10),key
    assert scores.loc['TN','N']==166 and round(scores.loc['TN','R2'],3)==.742
    checks['corrected_validation_matches']=True
    (OUT/'verification.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
