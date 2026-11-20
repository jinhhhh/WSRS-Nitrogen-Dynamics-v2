"""Exploratory residual and method envelopes for frozen v2 reconstructions.

These are marginal scenario ranges, not calibrated 90/95% prediction intervals.
Nested audits remove the outer block from every inner calibration fit.
"""
from __future__ import annotations
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['LOKY_MAX_CPU_COUNT']='1'
os.environ['MKL_NUM_THREADS']='1'
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import paper_validation_v2 as model

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'regenerated/error_ranges'
SOURCE=ROOT/'paper_package/tables/CIN_imputed_v2.csv'
TARGETS=model.TARGETS
METHODS=model.MODELS
LENGTHS=[1,3,7,15,30,60,115,150,162]
N_REFERENCE=16
N_AUDIT=6
N_INNER=10
SEED=2026111602
WINDOWS=[(21,41),(389,398),(413,429),(436,448)]

def save(d,name):
    d.to_csv(OUT/'data'/name,index=False,encoding='utf-8-sig')

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def one_prediction(y,x,hidden,method):
    if method in ['ExtraTrees','ET_bridge']:
        return model.all_predictions(y,x,hidden)[method]
    seen=np.isfinite(y)&~hidden
    pos=np.where(seen)[0];full=np.arange(len(y))
    linear=np.interp(full,pos,y[seen])
    if method=='Linear':return np.maximum(linear,0)
    p=PchipInterpolator(pos,y[seen],extrapolate=False)(full)
    return np.maximum(np.where(np.isfinite(p),p,linear),0)

def weighted_quantile(values,quantiles,dates):
    values=np.asarray(values,float);dates=np.asarray(dates,int)
    good=np.isfinite(values);values=values[good];dates=dates[good]
    if not len(values):return np.full(len(quantiles),np.nan)
    _,inv,n=np.unique(dates,return_inverse=True,return_counts=True)
    weights=1/n[inv]
    order=np.argsort(values);values=values[order];weights=weights[order]
    centers=(np.cumsum(weights)-.5*weights)/weights.sum()
    return np.interp(quantiles,centers,values,left=values[0],right=values[-1])

def residual_bounds(errors,dates):
    low,high=weighted_quantile(errors,[.05,.95],dates)
    # Keep the frozen point reconstruction within the reference range.
    return min(float(low),0),max(float(high),0)

def max_nonoverlap(starts,length):
    n=0;end=-1
    for a in sorted(starts):
        if a>=end:n+=1;end=a+length
    return n

def select_starts(y,length,n,rng,excluded=None):
    cs=model.candidates(y,length,excluded)
    return model.draw_blocks(cs,length,n,rng),len(cs)

def frozen_method(gaps,target,length):
    g=gaps[gaps.Target==target].copy()
    levels=np.sort(g.Selection_length.unique())
    nearest=min(levels,key=lambda a:abs(a-length))
    methods=g[g.Selection_length==nearest].Method.unique()
    assert len(methods)==1,(target,length,methods)
    return str(methods[0])

def main():
    for sub in ['data','figures','notes','qa']:(OUT/sub).mkdir(parents=True,exist_ok=True)
    source_hash=digest(SOURCE)
    d=pd.read_csv(SOURCE,parse_dates=['Date'])
    gaps=pd.read_csv(ROOT/'paper_package/tables/Table_S6_final_gap_methods.csv',parse_dates=['Start','End'])
    x=model.features(d)
    daily=pd.DataFrame({'Date':d.Date,'Model_JDAY':np.arange(1,len(d)+1),'Original_JDY':d.JDY})
    event=np.full(len(d),'Non_WSRS',dtype=object)
    for k,(a,b) in enumerate(WINDOWS,1):event[a-1:b]=f'E{k}'
    daily['Event']=event
    assert (daily.Event!='Non_WSRS').sum()==61
    reference_rows=[];reference_masks=[];audit_rows=[];audit_masks=[];gap_rows=[];checks=[]
    alternatives=pd.DataFrame({'Date':d.Date})
    for ti,t in enumerate(TARGETS):
        print('Error ranges:',t,flush=True)
        y=d[t].to_numpy(float);base=d[t+'_Imputed'].to_numpy(float)
        observed=np.isfinite(y);month_fraction=d.groupby(d.Date.dt.month)[t].agg(lambda a:a.notna().mean())
        full=model.all_predictions(y,x,~observed)
        for method in METHODS:
            a=y.copy();a[~observed]=full[method][~observed];alternatives[t+'_'+method]=a
        refs={}
        for length in LENGTHS:
            n_reference=40 if length==1 else (24 if length==3 else N_REFERENCE)
            n_audit=12 if length==1 else (8 if length==3 else N_AUDIT)
            n_inner=24 if length==1 else (16 if length==3 else N_INNER)
            rng=np.random.default_rng(SEED+ti*10000+length)
            starts,n_candidates=select_starts(y,length,n_reference,rng)
            residuals={m:[] for m in METHODS};positions=[]
            for rep,a in enumerate(starts,1):
                hidden=np.zeros(len(d),bool);hidden[a:a+length]=True
                pp=model.all_predictions(y,x,hidden)
                idx=np.arange(a,a+length);idx=idx[observed[idx]]
                positions.extend(idx.tolist())
                reference_masks.append(dict(Target=t,Length=length,Reference_mask=rep,
                    Start=d.Date[a],End=d.Date[a+length-1],Measured_days=len(idx),Candidate_starts=n_candidates))
                for method in METHODS:residuals[method].extend((y[idx]-pp[method][idx]).tolist())
            if not starts:raise RuntimeError(f'No reference masks: {t}/{length}')
            for method in METHODS:
                err=np.array(residuals[method])
                low,high=residual_bounds(err,positions)
                q90=float(weighted_quantile(np.abs(err),[.90],positions)[0])
                row=dict(Target=t,Length=length,Method=method,Residual_q05=low,Residual_q95=high,
                    Residual_min=float(min(err.min(),0)),Residual_max=float(max(err.max(),0)),
                    Abs_error_q90=q90,Reference_masks=len(starts),Max_nonoverlap_masks=max_nonoverlap(starts,length),
                    Prediction_pairs=len(err),Unique_dates=len(set(positions)),
                    Weighted_signed_median=float(weighted_quantile(err,[.5],positions)[0]))
                refs[length,method]=row;reference_rows.append(row)

            # Nested audit: outer truth is absent from all internal calibration training sets.
            outer_rng=np.random.default_rng(SEED+ti*10000+4000+length)
            outer_starts,_=select_starts(y,length,n_audit,outer_rng)
            method=frozen_method(gaps,t,length)
            for rep,a in enumerate(outer_starts,1):
                b=a+length;hidden=np.zeros(len(d),bool);hidden[a:b]=True
                pp=model.all_predictions(y,x,hidden)
                outer_idx=np.arange(a,b);outer_idx=outer_idx[observed[outer_idx]]
                inner_y=y.copy();inner_y[hidden]=np.nan
                inner_starts,_=select_starts(inner_y,length,n_inner,outer_rng,hidden)
                errs=[];dates=[];training_counts=[]
                for ia in inner_starts:
                    ih=hidden.copy();ih[ia:ia+length]=True
                    assert not hidden[ia:ia+length].any()
                    pred=one_prediction(y,x,ih,method)
                    ii=np.arange(ia,ia+length);ii=ii[observed[ii]]
                    assert not hidden[ii].any()
                    errs.extend((y[ii]-pred[ii]).tolist());dates.extend(ii.tolist())
                    training_counts.append(int((observed&~ih).sum()))
                if not errs:
                    audit_masks.append(dict(Target=t,Length=length,Audit_mask=rep,Start=d.Date[a],End=d.Date[b-1],
                        Method=method,Status='No_disjoint_calibration_support',Inner_masks=0))
                    continue
                lo,hi=residual_bounds(errs,dates)
                point=pp[method][outer_idx]
                residual_lo=np.maximum(0,point+lo);residual_hi=np.maximum(0,point+hi)
                spread=np.vstack([pp[m][outer_idx] for m in METHODS])
                envelope_lo=np.minimum(residual_lo,spread.min(axis=0))
                envelope_hi=np.maximum(residual_hi,spread.max(axis=0))
                stress_lo=np.minimum(envelope_lo,np.maximum(0,point+min(min(errs),0)))
                stress_hi=np.maximum(envelope_hi,np.maximum(0,point+max(max(errs),0)))
                audit_masks.append(dict(Target=t,Length=length,Audit_mask=rep,Start=d.Date[a],End=d.Date[b-1],
                    Method=method,Status='Audited',Inner_masks=len(inner_starts),
                    Max_nonoverlap_inner_masks=max_nonoverlap(inner_starts,length),
                    Inner_unique_dates=len(set(dates)),Inner_train_min=min(training_counts),
                    Inner_residual_q05=lo,Inner_residual_q95=hi,
                    Inner_residual_min=min(min(errs),0),Inner_residual_max=max(max(errs),0),
                    Outer_residual_labels_used=0))
                for j,pos in enumerate(outer_idx):
                    audit_rows.append(dict(Target=t,Length=length,Audit_mask=rep,Position=int(pos),Date=d.Date[pos],
                        Method=method,Actual=y[pos],Prediction=point[j],
                        Residual_Lower=residual_lo[j],Residual_Upper=residual_hi[j],
                        Envelope_Lower=envelope_lo[j],Envelope_Upper=envelope_hi[j],
                        Stress_Lower=stress_lo[j],Stress_Upper=stress_hi[j],
                        Residual_Covered=int(residual_lo[j]<=y[pos]<=residual_hi[j]),
                        Envelope_Covered=int(envelope_lo[j]<=y[pos]<=envelope_hi[j]),
                        Stress_Covered=int(stress_lo[j]<=y[pos]<=stress_hi[j])))
            print(' ',length,'d references/audits:',len(starts),len(outer_starts),flush=True)

        # The ranges accompany, and never replace, the frozen v2 point reconstructions.
        rl=base.copy();ru=base.copy();ml=base.copy();mu=base.copy();sl=base.copy();su=base.copy()
        gap_lengths=np.zeros(len(d),int);ref_lengths=np.zeros(len(d),int)
        method_col=np.full(len(d),'Measured',object);flags=np.full(len(d),'Measured_fixed',object)
        for row in gaps[gaps.Target==t].itertuples():
            mask=d.Date.between(row.Start,row.End).to_numpy();idx=np.where(mask)[0]
            reference_length=next((z for z in LENGTHS if z>=row.Length),LENGTHS[-1])
            reference=refs[reference_length,row.Method]
            expected=full[row.Method][idx]
            assert np.allclose(base[idx],expected,rtol=1e-10,atol=1e-10),(t,row.Start)
            rl[idx]=np.maximum(0,base[idx]+reference['Residual_q05'])
            ru[idx]=np.maximum(0,base[idx]+reference['Residual_q95'])
            spread=np.vstack([full[m][idx] for m in METHODS])
            ml[idx]=spread.min(axis=0);mu[idx]=spread.max(axis=0)
            sl[idx]=np.maximum(0,base[idx]+reference['Residual_min'])
            su[idx]=np.maximum(0,base[idx]+reference['Residual_max'])
            gap_lengths[idx]=row.Length;ref_lengths[idx]=reference_length;method_col[idx]=row.Method
            support=month_fraction.reindex(d.Date.iloc[idx].dt.month).to_numpy()
            unsupported=int((support<.2).sum())
            flag='Long_gap_extrapolation' if row.Length>30 else 'Internal_empirical_reference'
            if unsupported:flag='Seasonal_extrapolation'
            flags[idx]=flag
            gap_rows.append(dict(Target=t,Start=row.Start,End=row.End,Gap_days=row.Length,Method=row.Method,
                Reference_length=reference_length,Residual_q05=reference['Residual_q05'],Residual_q95=reference['Residual_q95'],
                Residual_min=reference['Residual_min'],Residual_max=reference['Residual_max'],
                Reference_masks=reference['Reference_masks'],Max_nonoverlap_masks=reference['Max_nonoverlap_masks'],
                Unique_reference_dates=reference['Unique_dates'],Low_season_support_days=unsupported,Support=flag))
        lower=np.minimum(rl,ml);upper=np.maximum(ru,mu)
        stress_lower=np.minimum(lower,sl);stress_upper=np.maximum(upper,su)
        assert np.isfinite(lower).all() and np.isfinite(upper).all()
        assert (lower<=base+1e-12).all() and (upper>=base-1e-12).all()
        assert np.array_equal(lower[observed],y[observed]) and np.array_equal(upper[observed],y[observed])
        assert (stress_lower<=lower+1e-12).all() and (stress_upper>=upper-1e-12).all()
        assert np.array_equal(stress_lower[observed],y[observed]) and np.array_equal(stress_upper[observed],y[observed])
        daily[t+'_Base']=base;daily[t+'_WasImputed']=(~observed).astype(int)
        daily[t+'_Residual_Lower']=rl;daily[t+'_Residual_Upper']=ru
        daily[t+'_Method_Min']=ml;daily[t+'_Method_Max']=mu
        daily[t+'_Range_Lower']=lower;daily[t+'_Range_Upper']=upper
        daily[t+'_Stress_Lower']=stress_lower;daily[t+'_Stress_Upper']=stress_upper
        daily[t+'_Gap_Days']=gap_lengths;daily[t+'_Reference_Length']=ref_lengths
        daily[t+'_Method']=method_col;daily[t+'_Support']=flags
        checks.append(dict(Target=t,Base_unchanged=True,Measured_bounds_fixed=True,
            Finite_bounds=True,Base_inside_range=True,Stress_contains_reference=True,
            Stress_measured_bounds_fixed=True,Nonnegative_lower=bool((stress_lower>=0).all())))

    audit=pd.DataFrame(audit_rows);summaries=[]
    for keys,g in audit.groupby(['Target','Length'],sort=False):
        frequency=g.groupby('Position').Position.transform('size').to_numpy()
        weights=1/frequency
        wavg=lambda v:float(np.average(np.asarray(v,float),weights=weights))
        summaries.append(dict(Target=keys[0],Length=keys[1],Audit_masks=g.Audit_mask.nunique(),
            Prediction_pairs=len(g),Unique_dates=g.Position.nunique(),
            Residual_coverage_pct=100*wavg(g.Residual_Covered),Envelope_coverage_pct=100*wavg(g.Envelope_Covered),
            Stress_coverage_pct=100*wavg(g.Stress_Covered),
            Mean_residual_width=wavg(g.Residual_Upper-g.Residual_Lower),
            Mean_envelope_width=wavg(g.Envelope_Upper-g.Envelope_Lower),
            Mean_stress_width=wavg(g.Stress_Upper-g.Stress_Lower),
            Point_RMSE=float(np.sqrt(wavg((g.Actual-g.Prediction)**2)))))
    event_rows=[]
    for k,(a,b) in enumerate(WINDOWS,1):
        e=daily.iloc[a-1:b];q=d.QIN.iloc[a-1:b].to_numpy(float)
        for t in TARGETS:
            # Q-weighted concentrations need no constituent-to-W2 conversion.
            event_rows.append(dict(Event=f'E{k}',Start=e.Date.min(),End=e.Date.max(),Days=len(e),Target=t,
                Observed_days=int((e[t+'_WasImputed']==0).sum()),Imputed_days=int(e[t+'_WasImputed'].sum()),
                Flow_weighted_Base=float(np.average(e[t+'_Base'],weights=q)),
                Flow_weighted_Lower_scenario=float(np.average(e[t+'_Range_Lower'],weights=q)),
                Flow_weighted_Upper_scenario=float(np.average(e[t+'_Range_Upper'],weights=q)),
                Flow_weighted_Stress_Lower=float(np.average(e[t+'_Stress_Lower'],weights=q)),
                Flow_weighted_Stress_Upper=float(np.average(e[t+'_Stress_Upper'],weights=q)),
                Range_type='coherent_all_lower_all_upper_concentration_stress_not_probability'))
    monthly=[]
    for t in TARGETS:
        for month,grp in d.groupby(d.Date.dt.month):
            monthly.append(dict(Target=t,Month=int(month),Calendar_days=len(grp),Measured_days=int(grp[t].notna().sum()),
                Measured_fraction=float(grp[t].notna().mean())))
    save(daily,'daily_reconstruction_error_ranges.csv')
    save(alternatives,'alternative_method_reconstructions.csv')
    save(pd.DataFrame(reference_rows),'residual_reference_by_length_method.csv')
    save(pd.DataFrame(reference_masks),'reference_mask_manifest.csv')
    save(audit,'nested_range_audit_predictions.csv')
    save(pd.DataFrame(audit_masks),'nested_range_audit_manifest.csv')
    save(pd.DataFrame(summaries),'nested_range_audit_summary.csv')
    save(pd.DataFrame(gap_rows),'actual_gap_range_sources.csv')
    save(pd.DataFrame(event_rows),'event_concentration_ranges_61days.csv')
    save(pd.DataFrame(monthly),'monthly_observation_support.csv')
    save(pd.DataFrame(checks),'integrity_checks.csv')
    assert source_hash==digest(SOURCE)
    protocol=dict(source=str(SOURCE),source_SHA256=source_hash,seed=SEED,lengths=LENGTHS,
        residual_quantiles=[.05,.95],reference_masks_per_length={'1':40,'3':24,'other':N_REFERENCE},
        outer_audits_per_length={'1':12,'3':8,'other':N_AUDIT},
        inner_masks_per_audit={'1':24,'3':16,'other':N_INNER},methods=METHODS,event_windows_1_based_inclusive=WINDOWS,event_days=61,
        reference_residual='actual minus prediction; weight each pair by inverse multiplicity of its calendar date',
        residual_range='base + empirical residual q05/q95, expanded to include base, truncated at zero',
        final_range='union of residual reference and the four full-data method reconstructions on missing dates only',
        stress_range='expand the reference envelope using minimum/maximum observed held-out residuals; no probability interpretation',
        audit='Each outer audit block is removed from all inner calibration training labels and inner residual labels. Fixed v2 method, no outer retuning.',
        seasonal_flag='Month-level measured fraction <0.20 is a descriptive coverage flag, not a model validity threshold.',
        limitations=['Marginal exploratory scenario ranges: not calibrated 90/95% PI or CI.',
            'Masks overlap and dependent series are not exchangeable; no independent or distribution-free coverage guarantee.',
            'This protocol was developed after examining data and earlier scores; all diagnostics remain internal.',
            'The extreme-residual stress layer was added after inspecting preliminary internal coverage; its audit is exploratory, not independent confirmation.',
            'No water-quality observations in August-October; coverage in observed seasons cannot validate those gaps.',
            'All measured dates are held fixed; measurement uncertainty is omitted.',
            'Water temperature, hydrology, conversion coefficients and W2 parameters remain fixed.',
            'Seven marginal ranges are not a joint multivariate ensemble. Nitrogen closure must be enforced downstream.',
            'Event low/high values aggregate all-lower/all-upper concentration scenarios, not probability intervals or worst-case W2 budgets.'])
    (OUT/'notes/range_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Completed ranges; original v2 file unchanged.',flush=True)

if __name__=='__main__':main()
