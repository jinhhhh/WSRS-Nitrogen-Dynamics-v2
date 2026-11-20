"""Build paired W2 concentration files from the user-corrected v2 boundary.

No W2 execution. Original inputs and reconstruction data are never edited.
Outputs are deterministic conditional stress experiments, not probability bounds.
"""
from __future__ import annotations
from pathlib import Path
import csv
import hashlib
import json
import shutil
import zipfile
import numpy as np
import pandas as pd

PACKAGE=Path(__file__).resolve().parents[2]
ROOT=PACKAGE/'generated_scenarios'
WORKSPACE=PACKAGE
INPUT=WORKSPACE/'w2/source_baseline'
RECON=WORKSPACE/'reconstruction'
POINT=RECON/'paper_package/tables/CIN_imputed_v2.csv'
RANGES=RECON/'error_ranges/data/daily_reconstruction_error_ranges.csv'
METHODS=RECON/'error_ranges/data/alternative_method_reconstructions.csv'
COLS=['JDAY','TDS','CODMn','WaterAge','NH4','NOx','LDOM','LPOM','ALG1','DO']
TARGETS=['NH4','CODMn','Cond','DO','TN','Chl_a']
CONVERSION={'NH4':['NH4'],'CODMn':['CODMn','LDOM','LPOM'],'Cond':['TDS'],
            'DO':['DO'],'TN':['NOx'],'Chl_a':['ALG1']}
BRANCH_FACTOR=48.101/51.899
WINDOWS=[(21,41),(389,398),(413,429),(436,448)]
CASES=[
 ('B00_V2_Baseline','主情景','新v2基准',{},None),
 ('M01_Linear','主情景','缺测值统一线性插值对照',{},'Linear'),
 ('M02_PCHIP','主情景','缺测值统一PCHIP对照',{},'PCHIP'),
 ('N01_NOx_TNderived_StressLow','主情景','TN误差驱动的NOx输入低值',{'TN':'Lower'},None),
 ('N02_NOx_TNderived_StressHigh','主情景','TN误差驱动的NOx输入高值',{'TN':'Upper'},None),
 ('A01_NH4_StressLow','主情景','NH4输入低值',{'NH4':'Lower'},None),
 ('A02_NH4_StressHigh','主情景','NH4输入高值',{'NH4':'Upper'},None),
 ('O01_OM_StressLow','主情景','CODMn及有机物输入低值',{'CODMn':'Lower'},None),
 ('O02_OM_StressHigh','主情景','CODMn及有机物输入高值',{'CODMn':'Upper'},None),
 ('D01_DO_StressLow','主情景','DO输入低值',{'DO':'Lower'},None),
 ('D02_DO_StressHigh','主情景','DO输入高值',{'DO':'Upper'},None),
 ('J01_Cross_NOxLow_OMHigh_DOHigh','主情景','交叉A：NOx低、有机物高、DO高',{'TN':'Lower','CODMn':'Upper','DO':'Upper'},None),
 ('J02_Cross_NOxHigh_OMLow_DOLow','主情景','交叉B：NOx高、有机物低、DO低',{'TN':'Upper','CODMn':'Lower','DO':'Lower'},None),
 ('S01_TDS_StressLow','可选扩展','电导率误差驱动的TDS低值',{'Cond':'Lower'},None),
 ('S02_TDS_StressHigh','可选扩展','电导率误差驱动的TDS高值',{'Cond':'Upper'},None),
 ('G01_Algae_StressLow','可选扩展','叶绿素误差驱动的ALG1低值',{'Chl_a':'Lower'},None),
 ('G02_Algae_StressHigh','可选扩展','叶绿素误差驱动的ALG1高值',{'Chl_a':'Upper'},None),
]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(frame,name):frame.to_csv(ROOT/'_audit'/name,index=False,encoding='utf-8-sig')
def read_input(p):
    lines=p.read_text(encoding='utf-8-sig').splitlines()
    if len(lines)<4:raise ValueError(f'Incomplete W2 file: {p}')
    df=pd.read_csv(p,skiprows=1).dropna(subset=['JDAY']).reset_index(drop=True)
    if list(df.columns)!=COLS:raise ValueError(f'Unexpected column order: {p}')
    df=df.apply(pd.to_numeric,errors='raise')
    if len(df)!=731 or not np.array_equal(df.JDAY.to_numpy(),np.arange(1,732)):
        raise ValueError(f'JDAY must be 1..731: {p}')
    if not np.isfinite(df.to_numpy()).all() or (df.to_numpy()<0).any():
        raise ValueError(f'Nonfinite or negative concentrations: {p}')
    return df,lines[:3]

def write_input(frame,prefix,path):
    with path.open('w',encoding='utf-8',newline='') as f:
        f.write('\r\n'.join(prefix)+'\r\n')
        writer=csv.writer(f,lineterminator='\r\n')
        for row in frame.itertuples(index=False,name=None):
            writer.writerow([str(int(row[0]))]+[(f'{float(x):.12f}'.rstrip('0').rstrip('.') or '0') for x in row[1:]])

def ratio(numerator,denominator):
    a=np.asarray(numerator,float);b=np.asarray(denominator,float)
    good=np.abs(b)>1e-14
    fallback=float(np.median(a[good]/b[good]))
    return np.divide(a,b,out=np.full(len(a),fallback),where=good)

def main():
    for directory in ['_audit','_source_original','_return_results']:(ROOT/directory).mkdir(parents=True,exist_ok=True)
    # A rerun must not overwrite inputs for scenarios that already have model results.
    for sid,*_ in CASES:
        if (ROOT/sid/'massbal.csv').exists() or (ROOT/'_return_results'/sid/'massbal.csv').exists():
            raise RuntimeError('Model results exist; use a new output folder to rebuild scenario inputs.')
    source_hash={str(INPUT/f):sha(INPUT/f) for f in ['cin_br1.csv','cdt_br1.csv']}
    c,cp=read_input(INPUT/'cin_br1.csv');d,dp=read_input(INPUT/'cdt_br1.csv')
    s=pd.read_csv(POINT,parse_dates=['Date'],float_precision='round_trip')
    u=pd.read_csv(RANGES,parse_dates=['Date'],float_precision='round_trip')
    m=pd.read_csv(METHODS,parse_dates=['Date'],float_precision='round_trip')
    expected=pd.date_range('2023-06-01','2025-05-31')
    for frame in [s,u,m]:
        if not np.array_equal(frame.Date.to_numpy(),expected.to_numpy()):raise ValueError('Date alignment failed')
    alpha_raw=c.NOx.to_numpy()/s.TN_Imputed.to_numpy()
    allowed=np.array([.89,.82,.79,.714])
    alpha=allowed[np.argmin(abs(alpha_raw[:,None]-allowed),axis=1)]
    nox_error=abs(c.NOx.to_numpy()-alpha*s.TN_Imputed.to_numpy())
    if nox_error.max()>1e-7:raise ValueError('NOx must reference TN with the user-approved period factors')
    direct_checks=[]
    for field,t in [('NH4','NH4'),('CODMn','CODMn'),('DO','DO')]:
        delta=abs(c[field].to_numpy()-s[t+'_Imputed'].to_numpy())
        if delta.max()>1e-7:raise ValueError(f'{field} does not match frozen v2')
        direct_checks.append(dict(Field=field,Max_abs_difference=delta.max(),Matches_v2=True))
    if ((c.NH4+c.NOx)-s.TN_Imputed>1e-7).any():raise ValueError('Baseline inorganic N exceeds reference TN')
    for field in COLS[1:]:
        if abs(d[field]-c[field]*BRANCH_FACTOR).max()>1e-7:
            raise ValueError(f'CDT/CIN coefficient changed unexpectedly: {field}')
    for f in ['cin_br1.csv','cdt_br1.csv']:
        destination=ROOT/'_source_original'/f
        if destination.exists() and sha(destination)!=sha(INPUT/f):
            raise RuntimeError('Frozen source copy differs from current inputs; retain provenance and use a new folder.')
        shutil.copy2(INPUT/f,destination)
    factors={
        'TDS':ratio(c.TDS,s.Cond_Imputed),
        'LDOM':ratio(c.LDOM,s.CODMn_Imputed),
        'LPOM':ratio(c.LPOM,s.CODMn_Imputed),
        'ALG1':ratio(c.ALG1,s.Chl_a_Imputed),
    }
    daily=pd.DataFrame({'Date':s.Date,'Model_JDAY':np.arange(1,732),'Original_JDY':s.JDY,
                        'NOx_per_TN':alpha,'CDT_CIN_factor':BRANCH_FACTOR})
    for key,value in factors.items():daily[key+'_conversion_factor']=value
    daily['Event']='Non_WSRS'
    for k,(a,b) in enumerate(WINDOWS,1):daily.loc[a-1:b-1,'Event']=f'E{k}'
    assert daily.Event.ne('Non_WSRS').sum()==61
    save(daily,'daily_conversion_factors_and_events.csv')
    periods=[];start=0
    for j in range(1,len(s)+1):
        if j==len(s) or alpha[j]!=alpha[start]:
            periods.append(dict(Start=s.Date.iloc[start],End=s.Date.iloc[j-1],Days=j-start,NOx_per_TN=alpha[start]))
            start=j
    save(pd.DataFrame(periods),'NOx_reference_periods.csv')
    save(pd.DataFrame(direct_checks),'baseline_v2_comparison.csv')
    manifests=[];quality=[];changes=[];clips=[];hashes=[];nitrogen=[]
    for order,(sid,priority,label,perturb,method) in enumerate(CASES,1):
        folder=ROOT/sid;folder.mkdir(exist_ok=True)
        cc=c.copy();dd=d.copy()
        values={t:s[t+'_Imputed'].to_numpy().copy() for t in TARGETS}
        masks={t:s[t+'_WasImputed'].eq(1).to_numpy() for t in TARGETS}
        active=set(TARGETS) if method else set(perturb)
        field_masks={field:np.zeros(len(s),bool) for field in COLS[1:]}
        for t in active:
            mask=masks[t]
            proposed=m[t+'_'+method].to_numpy() if method else u[t+'_Stress_'+perturb[t]].to_numpy()
            values[t][mask]=proposed[mask]
            if t=='NH4':replacements={'NH4':values[t]}
            elif t=='DO':replacements={'DO':values[t]}
            elif t=='TN':replacements={'NOx':alpha*values[t]}
            elif t=='Cond':replacements={'TDS':factors['TDS']*values[t]}
            elif t=='Chl_a':replacements={'ALG1':factors['ALG1']*values[t]}
            elif t=='CODMn':replacements={'CODMn':values[t],'LDOM':factors['LDOM']*values[t],'LPOM':factors['LPOM']*values[t]}
            else:raise ValueError(t)
            for field,vec in replacements.items():
                cc.loc[mask,field]=vec[mask];field_masks[field]|=mask
        # Necessary inorganic feasibility guard only. It does NOT establish full TN closure.
        # N cases change a NOx proxy using a TN driver, rather than setting model total TN.
        if active:
            headroom=np.maximum(values['TN']-cc.NH4.to_numpy(),0)
            clipped=(cc.NOx.to_numpy()>headroom+1e-10)
            if clipped.any():
                allowed_clip=field_masks['NOx']|field_masks['NH4']
                if (clipped&~allowed_clip).any():raise ValueError('Cannot modify an unrelated NOx source date')
                for pos in np.where(clipped)[0]:
                    clips.append(dict(Scenario=sid,Date=s.Date.iloc[pos],JDAY=int(pos+1),Field='NOx',
                        Proposed=cc.NOx.iloc[pos],Realized=headroom[pos],TN_driver=values['TN'][pos],
                        NH4=cc.NH4.iloc[pos],Reason='Necessary NH4+NOx <= TN driver guard; organic/algal N not included'))
                cc.loc[clipped,'NOx']=headroom[clipped];field_masks['NOx']|=clipped
            if (cc.NH4.to_numpy()>values['TN']+1e-7).any():raise ValueError('NH4 alone exceeds TN driver')
        for field in COLS[1:]:
            mask=field_masks[field]
            dd.loc[mask,field]=cc.loc[mask,field]*BRANCH_FACTOR
        if sid=='B00_V2_Baseline':
            shutil.copy2(INPUT/'cin_br1.csv',folder/'cin_br1.csv')
            shutil.copy2(INPUT/'cdt_br1.csv',folder/'cdt_br1.csv')
        else:
            write_input(cc,cp,folder/'cin_br1.csv');write_input(dd,dp,folder/'cdt_br1.csv')
        rereads=[]
        changed_dates=set();n_changed=0
        for filename,proposed,baseline,prefix in [('cin_br1.csv',cc,c,cp),('cdt_br1.csv',dd,d,dp)]:
            check,actual_prefix=read_input(folder/filename)
            assert actual_prefix==prefix
            assert np.allclose(check.to_numpy(),proposed.to_numpy(),rtol=0,atol=1e-9)
            for field in COLS[1:]:
                diff=np.abs(check[field].to_numpy()-baseline[field].to_numpy())
                changed=diff>1e-9
                assert not (changed&~field_masks[field]).any(),(sid,field,'outside source mask')
                assert np.allclose(check.loc[~field_masks[field],field],baseline.loc[~field_masks[field],field],rtol=0,atol=1e-9)
                for pos in np.where(changed)[0]:
                    changes.append(dict(Scenario=sid,File=filename,Date=s.Date.iloc[pos],JDAY=int(pos+1),Field=field,
                        Baseline=baseline[field].iloc[pos],Scenario_value=check[field].iloc[pos],Delta=check[field].iloc[pos]-baseline[field].iloc[pos]))
                changed_dates.update(np.where(changed)[0].tolist());n_changed+=int(changed.sum())
            hashes.append(dict(Scenario=sid,File=filename,Bytes=(folder/filename).stat().st_size,SHA256=sha(folder/filename)))
            rereads.append(check)
        cin_check,cdt_check=rereads
        proportional=True
        for field in COLS[1:]:
            proportional &= bool(np.allclose(cdt_check[field],cin_check[field]*BRANCH_FACTOR,rtol=0,atol=1e-7))
        assert proportional
        quality.append(dict(Scenario=sid,Rows_each_file=731,Column_order_correct=True,Dates_1_to_731=True,
            Finite_and_nonnegative=True,Paired_branch_ratio_preserved=True,Changes_within_dependency_masks=True,
            Direct_measured_records_preserved=True,Baseline_files_byte_identical=bool(sid!='B00_V2_Baseline' or all(sha(folder/f)==sha(INPUT/f) for f in ['cin_br1.csv','cdt_br1.csv'])),
            Inorganic_feasibility_guard_cells=int(sum(v['Scenario']==sid for v in clips)),Complete_model_TN_closure_checked=False))
        changed_fields=[field for field in COLS[1:] if field_masks[field].any()]
        manifests.append(dict(Run_order=order,Scenario=sid,Priority=priority,Description=label,
            Changed_fields='+'.join(changed_fields) or 'None',Changed_dates=len(changed_dates),
            Changed_cells_both_files=n_changed,Uncertainty_basis='Alternative method' if method else ('Extreme-residual stress envelope' if active else 'Frozen corrected user baseline'),
            NOx_interpretation='TN-derived NOx driver; not measured NOx and not total model TN bound',
            Other_W2_inputs_and_parameters='Fixed to common baseline',Probability_interpretation='None'))
        for pos in range(len(s)):
            nitrogen.append(dict(Scenario=sid,Date=s.Date.iloc[pos],JDAY=int(pos+1),Reference_TN_driver=values['TN'][pos],
                NH4=cc.NH4.iloc[pos],NOx=cc.NOx.iloc[pos],Available_N_after_inorganic=values['TN'][pos]-cc.NH4.iloc[pos]-cc.NOx.iloc[pos],
                LDOM=cc.LDOM.iloc[pos],LPOM=cc.LPOM.iloc[pos],ALG1=cc.ALG1.iloc[pos],
                Note='Remaining amount is a diagnostic, not imposed organic N. Model N content coefficients not provided.'))
        instructions=f'''{sid}\n{label}\n优先级：{priority}\n\n将本文件夹的cin_br1.csv与cdt_br1.csv成对复制到独立模型运行目录。其他输入、参数、计算设置保持同一基准。完整模拟期，不只运行61个事件日。\n\n本组不是95%概率上下界。NOx按用户已确认的TN分时期系数推导；N组仅改变该NOx输入，并不保证模型总TN等于TN驱动值。\n\n运行后将原始massbal.csv、警告/错误日志保存在_return_results/{sid}/。不要把别组输出混入本组，也不要对原始massbal四舍五入后覆盖。\n'''
        (folder/'情景说明.txt').write_text(instructions,encoding='utf-8-sig')
        result=ROOT/'_return_results'/sid;result.mkdir(exist_ok=True)
    save(pd.DataFrame(manifests),'scenario_manifest.csv')
    save(pd.DataFrame(quality),'scenario_integrity_checks.csv')
    save(pd.DataFrame(changes),'changed_cells.csv')
    save(pd.DataFrame(clips,columns=['Scenario','Date','JDAY','Field','Proposed','Realized','TN_driver','NH4','Reason']),'inorganic_guard_adjustments.csv')
    save(pd.DataFrame(hashes),'scenario_file_checksums.csv')
    save(pd.DataFrame(nitrogen),'nitrogen_component_diagnostics.csv')
    config=dict(created_date='2026-11-16',study_start='2023-06-01',study_end='2025-05-31',days=731,
        event_windows_inclusive=WINDOWS,event_days=61,main_runs=13,optional_runs=4,
        input_files={str(INPUT/f):sha(INPUT/f) for f in ['cin_br1.csv','cdt_br1.csv']},
        source_reconstruction=str(POINT),source_reconstruction_SHA256=sha(POINT),
        ranges=str(RANGES),ranges_SHA256=sha(RANGES),methods=str(METHODS),methods_SHA256=sha(METHODS),
        NOx_rule='Use TN and preserve the user-confirmed original period factors (0.89,0.82,0.79,0.714).',
        branch_factor_expression='48.101/51.899',branch_factor=BRANCH_FACTOR,
        interpretation='Deterministic conditional concentration-input stress scenarios, not calibrated prediction intervals.',
        assumptions=['Daily TDS/Cond, ALG1/Chl-a, LDOM/CODMn and LPOM/CODMn conversion coefficients frozen to supplied baseline.',
            'NOx reconstructed mask follows TN because NOx is explicitly derived from TN here, not measured independently.',
            'Water temperature, hydrology, other boundary inputs, parameters and model structure held fixed.',
            'TP is not an active column in the supplied CIN/CDT; no TP perturbation files generated.',
            'No TIN file supplied and no temperature scenarios can be encoded in CIN/CDT.',
            'Complete TN closure requires actual model organic/algal nitrogen content coefficients; not checked or claimed.',
            'An inorganic feasibility guard clips NOx if NH4+NOx exceeds its TN driver; all adjustments are recorded.',
            'Cross scenarios are prespecified opposing input combinations, not proven W2 budget extrema.',
            'No W2 model has been executed by this builder.'])
    (ROOT/'_audit/protocol.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    for f,oldhash in source_hash.items():assert sha(Path(f))==oldhash
    for f in [POINT,RANGES,METHODS]:assert f.exists()
    mainlines='\n'.join(f'{i:02d}. {sid}：{label}' for i,(sid,priority,label,_,_) in enumerate(CASES[:13],1))
    readme=f'''新重建边界敏感性分析：请先阅读
版本：2026-11-16。研究期731天；四个事件共61天。

一、先运行哪个
先跑B00_V2_Baseline，确认新基准能正常运行及结果输出完整，再运行其余主情景。主情景共13组，含基准；S01/S02/G01/G02是4组可选扩展，不必第一轮全部运行。

{mainlines}

可选扩展：S01/S02为电导率误差通过固定系数映射的TDS变化；G01/G02为叶绿素误差通过固定系数映射的藻类输入变化。

二、怎样运行
每组文件夹中的cin_br1.csv和cdt_br1.csv必须成对替换。建议每组使用独立W2模型目录。其他输入（含水温、流量、气象、出流等）、参数及计算设置保持同一基准，不要每个情景分别重新率定。运行完整研究时段，保留事件前期的输移与库存效应。

基准B00逐字节保留用户修正后的两份文件。原始输入另存于_source_original/，不作为另一个待跑情景。没有删除或改写旧十组分析，也未执行W2。

三、这些情景代表什么
采用新的v2重建及error_ranges中极值残差压力范围。单变量情景仅改变相应原始缺测日期：NH4自身、DO自身、CODMn及其派生LDOM/LPOM、TN派生NOx、Cond派生TDS、Chl-a派生ALG1。直接实测记录保留。M01/M02使用两种完整重建方法作为对照。

NOx的2023年引用错误已由用户自行修正；本包不再修正或重写基准。731天均核对为TN乘原分时期系数。该系数日历来自给定输入，与61天事件分组分别保存，不自动把分时期系数改为按61天事件切换。

N01/N02的名称是“TN误差驱动的NOx变化”，不是“模型总TN严格上下界”。完整模型TN还包括有机物和藻类所含氮，需要实际W2的氮含量系数才能对账。本次只做必要的NH4+NOx不超过TN驱动值的检查，个别越界NOx被限制并记录于_audit/inorganic_guard_adjustments.csv；这不代表有机/藻类氮已闭合。

O组增加有机物时同时增加其潜在氮输入，G组增加藻类时也可能增加藻类氮；这些组是条件单因素压力试验，不是固定总TN的组分替换试验。不要在论文中误写为“所有情景TN都保持不变”或“已经保证总TN完全复原”。

参考和压力范围都不是95%PI/CI。极值范围本身也不保证涵盖未观测季节的所有真实极端。J01/J02是交叉方向压力对照，不是氮预算的全局最小/最大，也没有各50%的发生概率。

主入流与分布式入流沿用用户原有浓度比48.101/51.899。该比例是给定假设，不构成独立支流水质验证。换算系数和水温在本轮固定，故本轮不是完整模型不确定性分析。

四、建议返回哪些结果
每组至少返回原始完整massbal.csv，及模型警告/错误日志。按原情景名放入_return_results/相应子文件夹，或直接保留每个独立运行目录后打包。不要把只剩几列或四舍五入后的表替换原始输出。

massbal中至少需要JDAY、TN-Waterbody(kg)、TN-Sediment(kg)、InflowTN(kg)、DistributedTributaryTN(kg)、OutflowTN(kg)、WithdrawalTN(kg)。若实际版本字段名不同，请保留原文件完整列。

B00额外保留W2控制/参数文件（如w2_con.csv或npt）和用于校准/验证的时序输出，以便核对有机/藻类氮系数、激活组分、输出时段及新基准表现。换算假设与模型组分是否匹配，不能仅凭输入CSV的非负检查确认。

五、返回后怎样分析
按四事件逐组比较累计Fin、Fout、库存变化、R=Fin-Fout-DeltaM及其占累计入流氮的比例，同时展示非WSRS对照。负R不自动等同Fout>Fin。所有统计使用21—41、389—398、413—429、436—448天，含首尾共61天。

六、审计资料
_audit/scenario_manifest.csv：运行清单及每组改动范围。
_audit/scenario_integrity_checks.csv：731天、列序、非负、缺测掩膜和成对比例检查。
_audit/NOx_reference_periods.csv：各段NOx/TN系数及日期。
_audit/changed_cells.csv：每个改变单元的基准值、新值和日期。
_audit/nitrogen_component_diagnostics.csv：无机氮与剩余TN驱动的诊断，不能代替实际模型总TN核算。
_audit/scenario_file_checksums.csv及protocol.json：文件哈希、来源和分析定义。

若以后有模型结果已放入情景或_return_results文件夹，构建脚本会拒绝覆盖输入；应另建版本目录。
'''
    (ROOT/'请先阅读_运行与返回结果说明.txt').write_text(readme,encoding='utf-8-sig')
    (ROOT/'_return_results/返回结果放这里.txt').write_text('按情景子文件夹放原始massbal.csv和运行日志。B00同时保留控制参数文件与验证用时序输出。\n',encoding='utf-8-sig')
    record=pd.DataFrame([dict(Scenario=sid,Priority=priority,Run_status='Not_run',Model_run_directory='',Massbal_saved='',Notes='') for sid,priority,*_ in CASES])
    record.to_csv(ROOT/'运行记录_可填写.csv',index=False,encoding='utf-8-sig')
    archive=ROOT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(ROOT.rglob('*')):
            if path.is_file() and path.suffix!='.pyc':z.write(path,path.relative_to(ROOT.parent))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert sum(i.filename.endswith('/cin_br1.csv') for i in z.infolist())==18
        assert sum(i.filename.endswith('/cdt_br1.csv') for i in z.infolist())==18
    print('Created 13 main + 4 optional paired scenarios; all integrity checks passed.')
    print('NOx inorganic guard adjustments:',len(clips))
    print('Folder:',ROOT)
    print('Archive:',archive)

if __name__=='__main__':main()
