"""Reproduce event budgets and native-process summaries from the archived W2 files."""
from pathlib import Path
import csv, json, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'generated'
EVENTS={'E1':(21,41),'E2':(389,398),'E3':(413,429),'E4':(436,448)}
START=pd.Timestamp('2023-06-01')
BASE='B00_V2_Baseline'
I=['InflowTN(kg)','TributaryTN(kg)','DistributedTributaryTN(kg)','PrecipitationTN(kg)']
O=['OutflowTN(kg)','WithdrawalTN(kg)']
M=['TN-Waterbody(kg)','TN-Sediment(kg)','TN-Plants(kg)']
P={'NH4OM':('OM-N mineralization','#ad7436'),'NH4NITR':('Nitrification','#287c91'),'NO3DEN':('Denitrification','#776394')}

def read_w2(path):
    with path.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.reader(f))
    header=[s.strip() for s in rows[0]]
    while header and not header[-1]: header.pop()
    data=[]
    for row in rows[1:]:
        if not any(s.strip() for s in row): continue
        if len(row)<len(header) or any(s.strip() for s in row[len(header):]):
            raise ValueError(f'Unexpected W2 row width: {path}')
        data.append(row[:len(header)])
    d=pd.DataFrame(data,columns=header).apply(pd.to_numeric)
    if not np.isfinite(d.to_numpy()).all() or not (np.diff(d.JDAY)>0).all():
        raise ValueError(f'Invalid numeric/time records: {path}')
    return d

def save(d,name):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    d.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.12g')

def budgets():
    names=pd.read_csv(ROOT/'w2/protocol/scenario_manifest.csv').Scenario.iloc[:13].tolist()
    raw={name:read_w2(ROOT/'w2/scenarios'/name/'massbal.csv') for name in names}
    a=int(np.ceil(max(d.JDAY.min() for d in raw.values())))
    b=int(np.floor(min(d.JDAY.max() for d in raw.values())))
    bounds=np.arange(a,b+1,dtype=float);days=bounds[:-1].astype(int)
    rows=[];daily=[];periods=[]
    for name,d in raw.items():
        if d.WB.nunique()!=1: raise ValueError('One waterbody required')
        delta={c:np.diff(np.interp(bounds,d.JDAY,d[c])) for c in I+O+M}
        frame=pd.DataFrame({'Scenario':name,'Model_JDAY':days,'Date':START+pd.to_timedelta(days-1,unit='D')})
        frame['Event']=[next((e for e,(lo,hi) in EVENTS.items() if lo<=day<=hi),'non_WSRS') for day in days]
        frame['Fin_kg']=sum(delta[c] for c in I)
        frame['Fout_kg']=sum(delta[c] for c in O)
        frame['DeltaM_kg']=sum(delta[c] for c in M)
        frame['Net_external_kg']=frame.Fin_kg-frame.Fout_kg
        frame['R_kg']=frame.Net_external_kg-frame.DeltaM_kg
        daily.append(frame)
        def total(part):
            r={'Scenario':name,'Days':len(part)}
            r.update({c.replace('_kg','_tN'):part[c].sum()/1000 for c in ['Fin_kg','Fout_kg','Net_external_kg','DeltaM_kg','R_kg']})
            r['R_over_Fin_percent']=100*r['R_tN']/r['Fin_tN']
            return r
        for event,(lo,hi) in EVENTS.items():
            part=frame[frame.Event.eq(event)]
            assert len(part)==hi-lo+1
            r=total(part);r['Event']=event
            # Independent check against original cumulative endpoint differences.
            direct=lambda c:np.interp(hi+1,d.JDAY,d[c])-np.interp(lo,d.JDAY,d[c])
            expected=sum(direct(c) for c in I)-sum(direct(c) for c in O)-sum(direct(c) for c in M)
            assert abs(expected-r['R_tN']*1000)<1e-5
            rows.append(r)
        for period,mask in [('All_fully_covered_days',np.ones(len(frame),bool)),('WSRS',frame.Event.ne('non_WSRS')),('non_WSRS',frame.Event.eq('non_WSRS'))]:
            r=total(frame[mask]);r['Period']=period;periods.append(r)
    events=pd.DataFrame(rows)
    save(events,'budgets/all_13_event_budgets.csv')
    save(pd.concat(daily,ignore_index=True),'budgets/daily_calendar_aligned_budgets.csv')
    save(pd.DataFrame(periods),'budgets/period_budgets.csv')
    directions=[]
    for event,g in events.groupby('Event',sort=False):
        baseline=g[g.Scenario.eq(BASE)].iloc[0]
        directions.append({'Event':event,'Scenarios':len(g),
            'Output_exceeds_input':int((g.Fout_tN>g.Fin_tN).sum()),
            'Input_exceeds_output':int((g.Fin_tN>g.Fout_tN).sum()),
            'Inventory_decrease':int((g.DeltaM_tN<0).sum()),'Inventory_increase':int((g.DeltaM_tN>0).sum()),
            'Positive_R':int((g.R_tN>0).sum()),'Negative_R':int((g.R_tN<0).sum()),
            'Baseline_R_tN':baseline.R_tN,'R_min_tN':g.R_tN.min(),'R_max_tN':g.R_tN.max()})
    save(pd.DataFrame(directions),'budgets/event_direction_summary.csv')
    return events

def validation():
    d=pd.read_csv(ROOT/'results/paper_tables/W2_validation_2025_corrected_pairs.csv')
    rows=[]
    for var,g in d.groupby('Variable',sort=False):
        y=g.Observed.to_numpy();p=g.Simulated.to_numpy();res=p-y
        rows.append({'Variable':var,'N':len(g),'R2':1-(res**2).sum()/((y-y.mean())**2).sum(),
                     'RMSE':np.sqrt((res**2).mean()),'MAE':abs(res).mean(),
                     'PBIAS_percent_sim_minus_obs':100*res.sum()/y.sum()})
    save(pd.DataFrame(rows),'validation/corrected_2025_metrics.csv')

def native():
    subprocess.run([sys.executable,str(ROOT/'analysis/analyze_native_processes.py')],check=True)
    daily=pd.read_csv(OUT/'native_flux_audit/daily_native_N_diagnostics.csv',parse_dates=['Date'])
    rows=[];events=[]
    groups={'All_complete_days':np.ones(len(daily),bool),'Non_WSRS':daily.Event.eq('non_WSRS'),'WSRS':daily.Event.ne('non_WSRS')}
    groups.update({event:daily.Event.eq(event) for event in EVENTS})
    for period,mask in groups.items():
        for key,(label,_) in P.items():
            v=daily.loc[mask,key+'_kgd']/1000
            rows.append({'Period':period,'Process':key,'Days':len(v),'Mean_tN_day':v.mean(),'Median_tN_day':v.median(),
                         'Q25_tN_day':v.quantile(.25),'Q75_tN_day':v.quantile(.75),'Cumulative_tN':v.sum()})
    stats=pd.DataFrame(rows);save(stats,'native_processes/period_event_statistics.csv')
    lookup=stats.set_index(['Period','Process'])
    comparison=[]
    for key,(label,_) in P.items():
        n=lookup.loc[('Non_WSRS',key),'Mean_tN_day'];w=lookup.loc[('WSRS',key),'Mean_tN_day']
        comparison.append({'Process':key,'Non_WSRS_mean_tN_day':n,'WSRS_mean_tN_day':w,'Mean_difference_percent':100*(w/n-1)})
    save(pd.DataFrame(comparison),'native_processes/WSRS_nonWSRS_comparison.csv')
    for event in EVENTS:
        part=daily[daily.Event.eq(event)];r={'Event':event,'Days':len(part)}
        for key in P:
            r[key+'_mean_tN_day']=part[key+'_kgd'].mean()/1000
            r[key+'_cumulative_tN']=part[key+'_kgd'].sum()/1000
        events.append(r)
    save(pd.DataFrame(events),'native_processes/event_totals_and_means.csv')
    return daily,lookup

def plots(events,daily,lookup):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    names=list(EVENTS);x=np.arange(4);base=events[events.Scenario.eq(BASE)].set_index('Event').loc[names]
    ax=axs[0]
    ax.bar(x-.18,(base.Fout_tN-base.Fin_tN)/1000,width=.36,label='Output minus input',color='#277f89')
    ax.bar(x+.18,-base.DeltaM_tN/1000,width=.36,label='Inventory depletion',color='#d58b49')
    ax.axhline(0,color='.35',lw=.7);ax.set_xticks(x,names);ax.set_ylabel('10³ t N');ax.legend(frameon=False,fontsize=8)
    ax.set_title('(a) B00 event budgets',loc='left')
    ax=axs[1]
    for j,event in enumerate(names):
        g=events[events.Event.eq(event)]
        ax.plot(np.full(len(g),j),g.R_tN,'o',color='#a2bbc1',ms=4)
        ax.plot(j,base.loc[event,'R_tN'],'D',color='#173b50',ms=5)
    ax.axhline(0,color='.35',lw=.7);ax.set_xticks(x,names);ax.set_ylabel('Residual R (t N)');ax.set_title('(b) All 13 boundary scenarios',loc='left')
    savefig(fig,'event_budget_summary')
    fig,axs=plt.subplots(3,2,figsize=(8.8,6.9),gridspec_kw={'width_ratios':[2.6,1.15],'hspace':.43,'wspace':.33})
    for i,(key,(label,color)) in enumerate(P.items()):
        ax=axs[i,0];ax.plot(daily.Date,daily[key+'_kgd']/1000,color=color,lw=.95)
        ax.set_ylabel('t N/day');ax.set_title(f'({chr(97+2*i)}) {label}',loc='left')
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6,9,12,3]));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.set_xlim(START,pd.Timestamp('2025-05-31'))
        for lo,hi in EVENTS.values(): ax.axvspan(START+pd.Timedelta(days=lo-1),START+pd.Timedelta(days=hi),color='#db9b68',alpha=.14)
        if i<2:ax.tick_params(labelbottom=False)
        else:ax.set_xlabel('Date')
        ax=axs[i,1];vals=[lookup.loc[(event,key),'Mean_tN_day'] for event in names]
        ref=lookup.loc[('Non_WSRS',key),'Mean_tN_day']
        ax.bar(x,vals,color=color,width=.58);ax.axhline(ref,color='#535c64',ls='--',lw=.9)
        ax.set_xticks(x,[f'{event}\n{hi-lo+1} d' for event,(lo,hi) in EVENTS.items()])
        ax.set_title(f'({chr(98+2*i)}) Event mean flux',loc='left');ax.set_ylim(0,max(max(vals),ref)*1.23)
        for a in axs[i]:a.grid(axis='y',alpha=.14);a.set_axisbelow(True)
    fig.subplots_adjust(left=.085,right=.98,top=.95,bottom=.10)
    savefig(fig,'native_nitrogen_processes')

def savefig(fig,name):
    out=OUT/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/(name+'.jpg'),dpi=300,bbox_inches='tight')
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True)
    events=budgets();validation();daily,lookup=native();plots(events,daily,lookup)
    print(json.dumps({'event_budget_rows':len(events),'scenarios':events.Scenario.nunique(),
                      'complete_days':len(daily),'WSRS_days':int(daily.Event.ne('non_WSRS').sum()),
                      'output':str(OUT)},indent=2))

if __name__=='__main__': main()
