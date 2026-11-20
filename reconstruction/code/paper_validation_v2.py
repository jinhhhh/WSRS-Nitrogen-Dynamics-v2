"""Repeated nested masking for offline reconstruction; never use hidden target values.

Run with D:/python/python.exe. Outputs are independent of the first-run results.
Outer repeats may overlap: they are masking scenarios, NOT independent samples.
"""
from __future__ import annotations
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

warnings.filterwarnings('ignore', category=UserWarning)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'regenerated/paper_package'
TARGETS = ['NH4', 'CODMn', 'Cond', 'DO', 'TP', 'TN', 'Chl_a']
LENGTHS = [1, 3, 7, 15, 30, 60, 115, 150]
MODELS = ['Linear', 'PCHIP', 'ExtraTrees', 'ET_bridge']
N_OUTER = 5
N_INNER = 2
SEED = 20261116

def data():
    d = pd.read_csv(ROOT / 'data/CIN_2023-2024-2025_quality_checked_v2.csv')
    d['Date'] = pd.Timestamp('2023-01-01') + pd.to_timedelta(d.JDY-1, unit='D')
    return d[d.Date.between('2023-06-01', '2025-05-31')].reset_index(drop=True)

def features(d):
    # Only the complete hydrometeorological inputs and calendar enter the tree.
    # They are treated as fixed inputs; existing WT reconstruction uncertainty is not propagated.
    day = d.Date.dt.dayofyear.to_numpy()
    x = {'time': np.arange(len(d)), 'sin': np.sin(2*np.pi*day/365.25),
         'cos': np.cos(2*np.pi*day/365.25),
         'wind_sin': np.sin(d.WD), 'wind_cos': np.cos(d.WD)}
    for c in ['WT','QIN','TAIR','TDEW','WS','CLOUD']:
        s = d[c]
        x[c] = s
        x[c+'_mean7'] = s.rolling(7, min_periods=1, center=True).mean()
        x[c+'_mean21'] = s.rolling(21, min_periods=1, center=True).mean()
        x[c+'_diff'] = s.diff()
    return pd.DataFrame(x)

def intervals(mask):
    v = np.r_[False, np.asarray(mask, bool), False].astype(int)
    return list(zip(np.where(np.diff(v)==1)[0], np.where(np.diff(v)==-1)[0]))

def metrics(y, p, q=None):
    y, p = np.asarray(y,float), np.asarray(p,float)
    ok = np.isfinite(y)&np.isfinite(p)
    y,p = y[ok],p[ok]
    if not len(y):
        return dict(N=0,R2=np.nan,RMSE=np.nan,MAE=np.nan,Bias=np.nan,FlowWeightedMAE=np.nan)
    sst = np.sum((y-y.mean())**2)
    r2 = 1-np.sum((y-p)**2)/sst if sst>1e-20 else np.nan
    wmae = np.nan
    if q is not None:
        w = np.asarray(q,float)[ok]
        wmae = np.sum(np.abs(p-y)*w)/w.sum() if w.sum()>0 else np.nan
    return dict(N=len(y),R2=r2,RMSE=float(np.sqrt(np.mean((p-y)**2))),
                MAE=float(np.mean(np.abs(p-y))),Bias=float(np.mean(p-y)), FlowWeightedMAE=wmae)

def candidates(y, length, excluded=None):
    n=len(y); seen=np.isfinite(y)
    forbidden=np.zeros(n,bool) if excluded is None else excluded
    cs=[]
    frac=1.0 if length<=7 else (0.75 if length<=30 else 0.50)
    for a in range(1,n-length):
        b=a+length
        if forbidden[a:b].any() or seen[a:b].sum()<max(1,int(np.ceil(frac*length))):
            continue
        # Require nearby observed anchors, while allowing actual holes within a long mask.
        if not seen[max(0,a-7):a].any() or not seen[b:min(n,b+7)].any():
            continue
        if seen.sum()-seen[a:b].sum()<80:
            continue
        cs.append(a)
    return cs

def draw_blocks(cs,length,n,rng):
    # Prefer disjoint blocks; if impossible keep distinct starts, with overlap declared.
    order=list(rng.permutation(cs))
    selected=[]
    for a in order:
        if all(a+length<=b or a>=b+length for b in selected):
            selected.append(int(a))
        if len(selected)==n:
            return selected
    for a in order:
        if int(a) not in selected:
            selected.append(int(a))
        if len(selected)==n:
            break
    return selected

def all_predictions(y,x,hidden):
    train=np.isfinite(y)&~hidden
    values=y.copy(); values[~train]=np.nan
    xp=np.where(train)[0]
    full=np.arange(len(y))
    if len(xp)<10:
        raise ValueError('Insufficient measured training data')
    linear=np.interp(full,xp,y[train])
    pchip=PchipInterpolator(xp,y[train],extrapolate=False)(full)
    pchip=np.where(np.isfinite(pchip),pchip,linear)
    et=make_pipeline(SimpleImputer(strategy='median'),ExtraTreesRegressor(
        n_estimators=70,min_samples_leaf=4,max_features=0.85,random_state=SEED,n_jobs=1))
    et.fit(x.loc[train],y[train])
    pred=et.predict(x)
    bridge=pred.copy()
    # Interpolate residual corrections from at most 5 measured anchors on either side.
    # These are available observations, and are never within hidden blocks.
    resid=y-pred
    for a,b in intervals(hidden | ~np.isfinite(y)):
        l=xp[xp<a][-5:]; r=xp[xp>=b][:5]
        if len(l) and len(r):
            left=float(np.median(resid[l])); right=float(np.median(resid[r]))
            w=(full[a:b]-l[-1])/(r[0]-l[-1])
            bridge[a:b]+=left*(1-w)+right*w
        elif len(l):
            bridge[a:b]+=np.median(resid[l])
        elif len(r):
            bridge[a:b]+=np.median(resid[r])
    return {k:np.maximum(v,0) for k,v in zip(MODELS,[linear,pchip,pred,bridge])}

def write_csv(frame,name):
    frame.to_csv(OUT/'tables'/name,index=False,encoding='utf-8-sig')

def main():
    for folder in ['tables','figures','notes','qa']:
        (OUT/folder).mkdir(parents=True,exist_ok=True)
    d=data(); x=features(d)
    observations=[]; gaps=[]; predictions=[]; selections=[]; manifest=[]; final_rows=[]
    final=d.copy()
    for ti,t in enumerate(TARGETS):
        y=d[t].to_numpy(float)
        print('Validating',t,flush=True)
        gs=intervals(~np.isfinite(y))
        observations.append(dict(Target=t,Study_days=len(d),Observed_days=int(np.isfinite(y).sum()),
            Missing_days=int((~np.isfinite(y)).sum()),Missing_percent=100*np.mean(~np.isfinite(y)),
            Gap_count=len(gs),Max_gap_days=max(b-a for a,b in gs),
            Long_gap_missing_days=sum(b-a for a,b in gs if b-a>30),
            Long_gap_share_percent=100*sum(b-a for a,b in gs if b-a>30)/(~np.isfinite(y)).sum(),
            Observed_sd=float(np.nanstd(y,ddof=1)),Near_0_025_days=int(np.isclose(y,0.025,atol=1e-6).sum()) if t=='NH4' else np.nan))
        for a,b in gs:
            gaps.append(dict(Target=t,Start=d.Date[a],End=d.Date[b-1],Length=b-a))
        for length in LENGTHS:
            rng=np.random.default_rng(SEED+ti*1000+length)
            cs=candidates(y,length)
            blocks=draw_blocks(cs,length,N_OUTER,rng)
            print(' ',length,'d:',len(blocks),'masks',flush=True)
            for rep,a in enumerate(blocks):
                b=a+length; hidden=np.zeros(len(d),bool);hidden[a:b]=True
                yp=all_predictions(y,x,hidden)
                inner_y=y.copy();inner_y[hidden]=np.nan
                inner_cs=candidates(inner_y,length,hidden)
                inners=draw_blocks(inner_cs,length,N_INNER,rng)
                scores={m:[] for m in MODELS}
                for ia in inners:
                    ih=hidden.copy();ih[ia:ia+length]=True
                    ip=all_predictions(y,x,ih)
                    ii=np.arange(ia,ia+length);ii=ii[np.isfinite(y[ii])]
                    for m in MODELS:
                        scores[m].extend((ip[m][ii]-y[ii])**2)
                if not inners:
                    chosen='Linear'
                else:
                    chosen=min(MODELS,key=lambda m:np.mean(scores[m]))
                selections.append(dict(Target=t,Length=length,Repeat=rep+1,Start=d.Date[a],
                    End=d.Date[b-1],Selected=chosen,Inner_blocks=len(inners),
                    Inner_RMSE=float(np.sqrt(np.mean(scores[chosen]))) if inners else np.nan))
                manifest.append(dict(Target=t,Length=length,Repeat=rep+1,Start=d.Date[a],End=d.Date[b-1],
                    Eval_days=int(np.isfinite(y[a:b]).sum()),Candidate_starts=len(cs)))
                idx=np.arange(a,b);idx=idx[np.isfinite(y[idx])]
                for m in MODELS+['Nested_selected']:
                    p=yp[chosen if m=='Nested_selected' else m]
                    for j in idx:
                        predictions.append(dict(Target=t,Design='Blocked',Length=length,Repeat=rep+1,
                            Date=d.Date[j],Position=int(j),Model=m,Actual=y[j],Predicted=p[j],QIN=d.QIN[j],
                            Selected=chosen if m=='Nested_selected' else m))

        # Random-point baseline is fixed PCHIP, rather than chosen using random-test scores.
        for rep in range(N_OUTER):
            rng=np.random.default_rng(SEED+ti*1000+900+rep)
            seen=np.where(np.isfinite(y))[0]
            idx=np.sort(rng.choice(seen,int(round(0.2*len(seen))),replace=False))
            hidden=np.zeros(len(d),bool);hidden[idx]=True
            train=np.isfinite(y)&~hidden;xp=np.where(train)[0]
            pred=PchipInterpolator(xp,y[train],extrapolate=False)(np.arange(len(d)))
            pred=np.maximum(np.where(np.isfinite(pred),pred,np.interp(np.arange(len(d)),xp,y[train])),0)
            for j in idx:
                predictions.append(dict(Target=t,Design='Random20pct',Length=0,Repeat=rep+1,
                    Date=d.Date[j],Position=int(j),Model='PCHIP',Actual=y[j],Predicted=pred[j],QIN=d.QIN[j],Selected='PCHIP'))

        # Full-data model selection is internal reconstruction CV, reported separately from outer scores.
        methods={}
        for length in LENGTHS:
            rng=np.random.default_rng(SEED+ti*1000+3000+length)
            cs=candidates(y,length)
            blocks=draw_blocks(cs,length,3,rng)
            loss={m:[] for m in MODELS}
            for a in blocks:
                h=np.zeros(len(d),bool);h[a:a+length]=True
                pp=all_predictions(y,x,h)
                idx=np.arange(a,a+length);idx=idx[np.isfinite(y[idx])]
                for m in MODELS:
                    loss[m].extend((pp[m][idx]-y[idx])**2)
            methods[length]=min(MODELS,key=lambda m:np.mean(loss[m])) if blocks else 'Linear'
        full_pred=all_predictions(y,x,~np.isfinite(y))
        values=y.copy(); flags=np.zeros(len(y),int)
        for a,b in gs:
            length=b-a
            representative=min(LENGTHS,key=lambda z:abs(z-length))
            method=methods[representative]
            values[a:b]=full_pred[method][a:b]; flags[a:b]=1
            final_rows.append(dict(Target=t,Start=d.Date[a],End=d.Date[b-1],Length=length,
                Selection_length=representative,Method=method,Support='limited_long_gap' if length>30 else 'short_or_medium_gap'))
        final[t+'_Imputed']=values
        final[t+'_WasImputed']=flags
        assert np.isfinite(values).all(),t
        assert np.array_equal(values[np.isfinite(y)],y[np.isfinite(y)]),t

    pred=pd.DataFrame(predictions)
    write_csv(pred,'validation_predictions_v2.csv')
    rows=[]; repeat_rows=[]
    for keys,g in pred.groupby(['Target','Design','Length','Model'],sort=False):
        row=dict(zip(['Target','Design','Length','Model'],keys))
        row.update(metrics(g.Actual,g.Predicted,g.QIN));row['Unique_dates']=g.Date.nunique()
        row['Masks']=g.Repeat.nunique();row['Observed_sd']=float(np.nanstd(d[row['Target']],ddof=1))
        row['NRMSE_global_sd']=row['RMSE']/row['Observed_sd']
        rows.append(row)
    for keys,g in pred.groupby(['Target','Design','Length','Model','Repeat'],sort=False):
        row=dict(zip(['Target','Design','Length','Model','Repeat'],keys))
        row.update(metrics(g.Actual,g.Predicted,g.QIN));repeat_rows.append(row)
    met=pd.DataFrame(rows)
    write_csv(met,'Table_S2_all_methods_metrics.csv')
    write_csv(pd.DataFrame(repeat_rows),'Table_S3_mask_level_metrics.csv')
    write_csv(pd.DataFrame(observations),'Table_1_data_availability.csv')
    write_csv(pd.DataFrame(gaps),'Table_S1_actual_gaps.csv')
    write_csv(pd.DataFrame(manifest),'Table_S4_validation_masks.csv')
    write_csv(pd.DataFrame(selections),'Table_S5_nested_selections.csv')
    write_csv(pd.DataFrame(final_rows),'Table_S6_final_gap_methods.csv')
    write_csv(final,'CIN_imputed_v2.csv')
    main_metrics=met[((met.Design=='Blocked')&(met.Model=='Nested_selected'))|(met.Design=='Random20pct')].copy()
    write_csv(main_metrics,'Table_2_main_performance.csv')
    (OUT/'notes/protocol.json').write_text(json.dumps(dict(seed=SEED,outer_repeats=N_OUTER,inner_masks=N_INNER,
        lengths=LENGTHS,models=MODELS,source=str(ROOT/'data/CIN_2023-2024-2025_quality_checked_v2.csv'),
        design='Nested repeated offline masking. Inner masks do not intersect outer masks. Outer labels are removed from all inner fits.',
        caveats=['Outer masks may overlap; prediction pairs are not independent replicates.',
        'Repeated evaluation informed this revised protocol; results are internal validation, not a new external independent dataset.',
        'No observations in the long missing summer/autumn intervals: unsupported seasonal domain is not validated.',
        'WT is a fixed existing reconstructed input; its uncertainty is not propagated.',
        'No probabilistic intervals or significance tests are produced.'],
        metrics='R2 is pooled over masking prediction pairs; report Unique_dates as well as N. FlowWeightedMAE weights absolute concentration error by daily QIN.'),ensure_ascii=False,indent=2),encoding='utf-8')
    print(main_metrics[['Target','Length','R2','RMSE','Unique_dates']].to_string(index=False),flush=True)

if __name__=='__main__':
    main()
