"""Native W2 nitrogen postprocessing, replacing the separate fitted box model.

No fitted reaction parameters, regime-specific switches or surrogate outflow are used.
Run on the current/new B00 folder; missing process outputs stay missing, never zero.
Optional NOx inventory/outflow data use the schema supplied in tables/.
"""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parents[1]
DEFAULT=HERE/'w2/scenarios/B00_V2_Baseline'
EVENTS={'E1':(21,41),'E2':(389,398),'E3':(413,429),'E4':(436,448)}
I=['InflowTN(kg)','TributaryTN(kg)','DistributedTributaryTN(kg)','PrecipitationTN(kg)']
O=['OutflowTN(kg)','WithdrawalTN(kg)']
M=['TN-Waterbody(kg)','TN-Sediment(kg)','TN-Plants(kg)']
CORE=['NH4NITR','NO3DEN','NH4DOM','NH4POM','NH4OM']
def read(path):
    with path.open(encoding='utf-8-sig')as f:r=list(csv.reader(f))
    h=[x.strip()for x in r[0]]
    while h and not h[-1]:h.pop()
    rows=[]
    for row in r[1:]:
        if not any(x.strip()for x in row):continue
        if len(row)<len(h)or any(x.strip()for x in row[len(h):]):raise ValueError('Unexpected CSV row width: '+str(path))
        rows.append(row[:len(h)])
    d=pd.DataFrame(rows,columns=h).apply(pd.to_numeric)
    if not np.isfinite(d.to_numpy()).all()or not(np.diff(d.JDAY)>0).all():raise ValueError('Invalid numeric/time records: '+str(path))
    return d
def write(d,path):d.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.10g')
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline-dir',type=Path,default=DEFAULT)
    ap.add_argument('--output-dir',type=Path,default=HERE/'generated/native_flux_audit')
    ap.add_argument('--nox-diagnostics',type=Path)
    args=ap.parse_args();src=args.baseline_dir;out=args.output_dir
    if out.resolve()==src.resolve():raise ValueError('Output directory must differ from original model folder')
    out.mkdir(parents=True,exist_ok=True)
    mass=read(src/'massbal.csv');kin=read(src/'kflux_wb1.csv')
    if mass.WB.nunique()!=1:raise ValueError('This analysis requires a single identified waterbody')
    start=int(np.ceil(max(mass.JDAY.min(),kin.JDAY.min())))
    end=int(np.floor(min(mass.JDAY.max(),kin.JDAY.max())))
    bounds=np.arange(start,end+1,dtype=float);days=bounds[:-1].astype(int)
    if not len(days):raise ValueError('No complete common calendar days')
    d=pd.DataFrame({'Model_JDAY':days,'Date':pd.Timestamp('2023-06-01')+pd.to_timedelta(days-1,unit='D')})
    diff=lambda col:np.diff(np.interp(bounds,mass.JDAY,mass[col]))
    d['Fin_TN_kgd']=sum(diff(c)for c in I);d['Fout_TN_kgd']=sum(diff(c)for c in O)
    d['Net_external_TN_kgd']=d.Fin_TN_kgd-d.Fout_TN_kgd
    d['Delta_inventory_TN_kgd']=sum(diff(c)for c in M)
    d['R_TN_kgd']=d.Net_external_TN_kgd-d.Delta_inventory_TN_kgd
    ends=kin.JDAY.to_numpy();first=ends[0]-kin.ELTM.iloc[0];t=np.r_[first,ends];dt=np.diff(t)
    fields={}
    for col in kin.columns:
        if col in ['JDAY','ELTM']:continue
        if 'kg/d'not in col:raise ValueError('Unrecognized kinetic flux unit: '+col)
        key=col.split()[0]
        fields[key]=col
        cumul=np.r_[0,np.cumsum(kin[col].to_numpy()*dt)]
        d[key+'_kgd']=np.diff(np.interp(bounds,t,cumul))
    d['Event']=[next((e for e,(a,b)in EVENTS.items()if a<=day<=b),'non_WSRS')for day in days]
    report=[]
    for event,(a,b)in EVENTS.items():
        part=d[d.Event==event]
        if len(part)!=b-a+1:raise ValueError('Incomplete event '+event)
        row={'Event':event,'Days':len(part),'Start_JDAY':a,'End_JDAY_inclusive':b}
        row.update({c.replace('_kgd','_tN'):part[c].sum()/1000 for c in d.columns if c.endswith('_kgd')})
        row['R_over_Fin_percent']=100*part.R_TN_kgd.sum()/part.Fin_TN_kgd.sum();report.append(row)
    report=pd.DataFrame(report)
    missing=[key for key in CORE if key not in fields]
    optional_status='Not supplied; inventory-normalized and NOx-export-normalized rates were not computed.'
    if args.nox_diagnostics:
        n=pd.read_csv(args.nox_diagnostics)
        required=['Model_JDAY','Whole_reservoir_NOx_inventory_kg','W2_NOx_outflow_kg_per_day']
        if not set(required).issubset(n):raise ValueError('NOx diagnostics do not match the supplied schema')
        if n.Model_JDAY.duplicated().any():raise ValueError('Duplicate NOx diagnostic dates')
        d=d.merge(n[required],on='Model_JDAY',how='left',validate='one_to_one')
        if d[required].isna().any().any():raise ValueError('Incomplete optional NOx diagnostics')
        if 'NO3DEN_kgd'not in d:raise ValueError('NO3DEN output is required for normalized diagnostics')
        if (d.Whole_reservoir_NOx_inventory_kg<=0).any()or(d.W2_NOx_outflow_kg_per_day<0).any():raise ValueError('Invalid NOx inventory/outflow')
        d['Denitrification_per_NOx_inventory_day_inverse']=d.NO3DEN_kgd/d.Whole_reservoir_NOx_inventory_kg
        d['Denitrification_over_NOx_outflow']=d.NO3DEN_kgd/d.W2_NOx_outflow_kg_per_day.replace(0,np.nan)
        for idx,row in report.iterrows():
            part=d[d.Event==row.Event]
            report.loc[idx,'Denitrification_per_NOx_inventory_day_inverse']=part.NO3DEN_kgd.sum()/part.Whole_reservoir_NOx_inventory_kg.sum()
            denom=part.W2_NOx_outflow_kg_per_day.sum()
            report.loc[idx,'Denitrification_over_event_NOx_outflow']=part.NO3DEN_kgd.sum()/denom if denom>0 else np.nan
        optional_status='Computed from supplied whole-reservoir inventory and actual outflow; provenance must be retained.'
    write(d,out/'daily_native_N_diagnostics.csv');write(report,out/'event_native_N_diagnostics.csv')
    state={'Core_process_outputs_complete':not missing,'Missing_core_output_fields':missing,'Available_native_flux_fields':list(fields),
           'Complete_daily_records':len(d),'WSRS_days':int((d.Event!='non_WSRS').sum()),
           'Separate_box_model_used':False,'Additional_postprocessing_fitted_parameters':0,'Additional_regime_specific_reaction_parameters':False,
           'Outflow':'Actual cumulative W2 outflow and withdrawals from massbal.csv, not Q times box-average concentration.',
           'Interpretation':'Native process diagnostics conditional on W2. No independent process validation or causal identification is implied.',
           'Aggregation':'All available process terms are kept separate. NH4OM and NH4DOM/POM are not added together.',
           'Optional_NOx_diagnostics':optional_status,
           'Source_SHA256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in [src/'massbal.csv',src/'kflux_wb1.csv']}}
    (out/'native_analysis_status.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(state,ensure_ascii=True,indent=2))
if __name__=='__main__':main()
