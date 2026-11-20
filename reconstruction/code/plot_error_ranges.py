"""Plot exploratory reconstruction ranges as JPG and PDF; write manuscript notes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image,ImageDraw
from build_error_ranges import TARGETS,LENGTHS,METHODS,frozen_method

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'regenerated/error_ranges';DATA=OUT/'data';FIG=OUT/'figures'
LABELS={'NH4':r'NH$_4^+$-N','CODMn':r'COD$_{Mn}$','Cond':'Conductivity','DO':'DO','TP':'TP','TN':'TN','Chl_a':'Chl-a'}
UNITS={t:r'mg L$^{-1}$' for t in TARGETS};UNITS['Cond']=r'$\mu$S cm$^{-1}$'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
                     'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white','figure.facecolor':'white'})

def save(fig,name,pages):
    fig.savefig(FIG/(name+'.jpg'),dpi=600,pil_kwargs={'quality':95})
    fig.savefig(FIG/(name+'.pdf'));pages.savefig(fig);plt.close(fig)

def dates(ax,d):
    ax.set_xlim(d.Date.min(),d.Date.max())
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.grid(axis='y',alpha=.15)

def fig1(ref,source,gaps,pages):
    fig,axs=plt.subplots(4,2,figsize=(11.5,10.4));xx=np.arange(len(LENGTHS))
    for i,t in enumerate(TARGETS):
        ax=axs.flat[i];sd=source[t].std()
        matrix=[];selected=[]
        for length in LENGTHS:
            g=ref[(ref.Target==t)&(ref.Length==length)]
            matrix.append([g[g.Method==m].Abs_error_q90.iloc[0]/sd for m in METHODS])
            method=frozen_method(gaps,t,length)
            selected.append(g[g.Method==method].Abs_error_q90.iloc[0]/sd)
        a=np.array(matrix)
        ax.fill_between(xx,a.min(axis=1),a.max(axis=1),color='#c4dce5',alpha=.8,label='Range across four methods')
        ax.plot(xx,selected,'o-',ms=3.5,lw=1.5,color='#126c8e',label='Frozen v2 method for this scale')
        ax.set_xticks(xx,[str(z) for z in LENGTHS]);ax.set_xlim(-.2,len(xx)-.8)
        ax.set_ylabel(r'$Q_{0.90}(|y-\hat{y}|)$ / observed SD')
        ax.set_title(f'({chr(97+i)}) '+LABELS[t],loc='left',fontweight='bold');ax.grid(alpha=.15)
        if i>=5:ax.set_xlabel('Calendar-block length (days; ordered categories)')
    ax=axs.flat[7];ax.axis('off');h,l=axs.flat[0].get_legend_handles_labels()
    ax.legend(h,l,loc='upper left',frameon=False,bbox_to_anchor=(0,1.0),fontsize=9)
    ax.text(.03,.1,'Residuals come from artificial missing blocks.\nRepeated dates receive inverse-frequency weights.\nThe 90th error percentile is descriptive;\nit is not a guaranteed prediction bound.',transform=ax.transAxes,fontsize=8.5,linespacing=1.7)
    fig.subplots_adjust(left=.10,right=.98,top=.91,bottom=.075,wspace=.26,hspace=.48)
    fig.suptitle('Held-out reconstruction error and method dependence',x=.10,ha='left',fontsize=12,fontweight='bold')
    fig.text(.10,.02,'Global observed SD is used only for scale normalization. Longer masks can be placed only where measured reference values exist.',fontsize=8)
    save(fig,'U01_residual_error_by_gap',pages)

def fig2(d,source,pages):
    fig,axs=plt.subplots(7,1,figsize=(11.5,11.8),sharex=True)
    for i,t in enumerate(TARGETS):
        ax=axs[i];miss=d[t+'_WasImputed'].eq(1)
        ax.fill_between(d.Date,d[t+'_Stress_Lower'],d[t+'_Stress_Upper'],where=miss,
                        color='#d5dce1',alpha=.8,interpolate=True,lw=0)
        ax.fill_between(d.Date,d[t+'_Range_Lower'],d[t+'_Range_Upper'],where=miss,
                        color='#73a9bd',alpha=.65,interpolate=True,lw=0)
        ax.plot(d.Date,d[t+'_Base'],color='#ce7b30',lw=.75)
        obs=source[t].notna();ax.scatter(source.loc[obs,'Date'],source.loc[obs,t],s=5,color='#233c4a',zorder=4)
        ax.text(.008,.85,f'({chr(97+i)})',transform=ax.transAxes,fontweight='bold',fontsize=8)
        ax.set_ylabel(LABELS[t]+'\n'+UNITS[t],fontsize=8);dates(ax,d)
    fig.legend(handles=[Line2D([],[],color='#233c4a',marker='o',lw=0,ms=3,label='Measured'),
        Line2D([],[],color='#ce7b30',lw=1,label='Frozen v2 point series'),
        Patch(facecolor='#73a9bd',alpha=.65,label='Residual + method reference envelope'),
        Patch(facecolor='#d5dce1',alpha=.8,label='Extreme-residual stress range')],
        loc='upper center',bbox_to_anchor=(.55,.98),ncol=2,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.10,right=.98,top=.925,bottom=.075,hspace=.21)
    fig.text(.10,.033,'Ranges are shown only on originally missing dates; measured values are fixed. They are exploratory marginal scenarios, not 95% PI/CI.',fontsize=8)
    fig.text(.10,.015,'No water-quality observations exist in August-October. Ranges in those seasons have no direct empirical validation.',fontsize=8)
    save(fig,'U02_daily_reconstruction_ranges',pages)

def fig3(d,source,events,pages):
    fig=plt.figure(figsize=(11.5,8.7));gs=fig.add_gridspec(2,2,height_ratios=[1.35,1],wspace=.36,hspace=.40)
    ax=fig.add_subplot(gs[0,:]);miss=d.TN_WasImputed.eq(1)
    ax.fill_between(d.Date,d.TN_Stress_Lower,d.TN_Stress_Upper,where=miss,color='#d5dce1',alpha=.8,interpolate=True,lw=0)
    ax.fill_between(d.Date,d.TN_Range_Lower,d.TN_Range_Upper,where=miss,color='#98c0ce',alpha=.7,interpolate=True,lw=0)
    ax.plot(d.Date,d.TN_Base,color='#cc7628',lw=1.1)
    obs=source.TN.notna();ax.scatter(source.loc[obs,'Date'],source.loc[obs,'TN'],s=9,color='#243d4a',zorder=4)
    tn=events[events.Target=='TN'].reset_index(drop=True)
    for r in tn.itertuples():
        ax.axvspan(r.Start,r.End,color='#be8056',alpha=.12)
        mid=r.Start+(r.End-r.Start)/2
        ax.text(mid,1.01,r.Event,transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
    dates(ax,d);ax.set_ylabel(r'TN (mg L$^{-1}$)');ax.set_title('(a) TN reconstruction and event windows',loc='left',fontweight='bold',pad=21)
    ax.legend(handles=[Patch(facecolor='#98c0ce',alpha=.7,label='Reference envelope'),Patch(facecolor='#d5dce1',alpha=.8,label='Stress range')],loc='upper right',frameon=False,ncol=2,fontsize=8)
    ax=fig.add_subplot(gs[1,0]);yy=np.arange(4)
    b=tn.Flow_weighted_Base.to_numpy();lo=tn.Flow_weighted_Lower_scenario.to_numpy();hi=tn.Flow_weighted_Upper_scenario.to_numpy()
    slo=tn.Flow_weighted_Stress_Lower.to_numpy();shi=tn.Flow_weighted_Stress_Upper.to_numpy()
    ax.errorbar(b,yy,xerr=np.vstack([b-slo,shi-b]),fmt='none',color='#acb7bf',capsize=4,lw=3)
    ax.errorbar(b,yy,xerr=np.vstack([b-lo,hi-b]),fmt='o',color='#226f8b',capsize=5,lw=1.8,ms=5)
    ax.set_yticks(yy,[f'{r.Event}\n{r.Observed_days}/{r.Days} measured' for r in tn.itertuples()]);ax.invert_yaxis()
    ax.tick_params(axis='y',labelsize=8)
    ax.set_xlabel(r'Flow-weighted TN (mg L$^{-1}$)');ax.set_xlim(left=0);ax.grid(axis='x',alpha=.2)
    ax.set_title('(b) Conditional boundary-concentration scenarios',loc='left',fontweight='bold',fontsize=9)
    ax=fig.add_subplot(gs[1,1]);frac=tn.Imputed_days/tn.Days*100
    ax.bar(np.arange(4),100-frac,color='#277f8d',label='Measured dates')
    ax.bar(np.arange(4),frac,bottom=100-frac,color='#e7a967',label='Reconstructed dates')
    for i,r in enumerate(tn.itertuples()):ax.text(i,50,f'{r.Imputed_days}/{r.Days}\nreconstructed',ha='center',va='center',fontsize=8)
    ax.set_xticks(range(4),tn.Event);ax.set_ylim(0,100);ax.set_ylabel('Event days (%)')
    ax.set_title('(c) Actual observation coverage: 61 event days',loc='left',fontweight='bold',fontsize=9)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),frameon=False,ncol=2,fontsize=8)
    fig.subplots_adjust(left=.10,right=.98,top=.91,bottom=.16)
    fig.text(.10,.07,'E1: 21 Jun-11 Jul 2023; E2: 23 Jun-2 Jul 2024; E3: 17 Jul-2 Aug 2024; E4: 9-21 Aug 2024.',fontsize=8)
    fig.text(.10,.045,'Panel (b): blue = reference; gray = stress range. QIN-weighted all-lower/all-upper trajectories are scenarios, not probability intervals.',fontsize=8)
    fig.text(.10,.022,'No W2 simulation, NOx conversion or nitrogen-species closure is included in these boundary-concentration diagnostics.',fontsize=8)
    save(fig,'U03_TN_event_ranges_61days',pages)

def fig4(audit,monthly,pages):
    fig,axs=plt.subplots(3,1,figsize=(11.5,10.4),gridspec_kw={'height_ratios':[1,1,.95]})
    cmap=LinearSegmentedColormap.from_list('cov',['#e9c0a4','#fbf4e9','#267e8f'])
    for k,col in enumerate(['Envelope_coverage_pct','Stress_coverage_pct']):
        ax=axs[k];mat=np.full((7,len(LENGTHS)),np.nan)
        for i,t in enumerate(TARGETS):
            for j,length in enumerate(LENGTHS):
                g=audit[(audit.Target==t)&(audit.Length==length)]
                if len(g):mat[i,j]=g.iloc[0][col]
        ax.imshow(mat,vmin=0,vmax=100,cmap=cmap,aspect='auto')
        for i,t in enumerate(TARGETS):
            for j,length in enumerate(LENGTHS):
                g=audit[(audit.Target==t)&(audit.Length==length)]
                txt='NA' if g.empty else f'{mat[i,j]:.0f}%\n(n={int(g.Unique_dates.iloc[0])})'
                ax.text(j,i,txt,ha='center',va='center',fontsize=7.5,color='white' if mat[i,j]>75 else '#293f4a')
        ax.set_xticks(range(len(LENGTHS)),[str(z)+' d' for z in LENGTHS]);ax.set_yticks(range(7),[LABELS[t] for t in TARGETS])
        ax.set_title('(a) Internal audit: residual + method reference coverage' if k==0 else '(b) Exploratory audit: extreme-residual stress-range coverage',loc='left',fontweight='bold')
    ax=axs[2];mat=np.array([[monthly[(monthly.Target==t)&(monthly.Month==m)].Measured_fraction.iloc[0]*100 for m in range(1,13)] for t in TARGETS])
    ax.imshow(mat,vmin=0,vmax=100,cmap=cmap,aspect='auto')
    for i in range(7):
        for j in range(12):ax.text(j,i,f'{mat[i,j]:.0f}%',ha='center',va='center',fontsize=8,color='white' if mat[i,j]>75 else '#293f4a')
    ax.set_xticks(range(12),['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_yticks(range(7),[LABELS[t] for t in TARGETS]);ax.set_title('(c) Original measured-day fraction by calendar month',loc='left',fontweight='bold')
    for ax in axs:
        ax.set_xticks(np.arange(-.5,len(ax.get_xticks()),1),minor=True)
        ax.set_yticks(np.arange(-.5,7,1),minor=True)
        ax.grid(which='minor',color='white',lw=1);ax.tick_params(which='minor',bottom=False,left=False)
    fig.subplots_adjust(left=.12,right=.98,top=.94,bottom=.13,hspace=.40)
    fig.text(.10,.073,'n = unique reference dates evaluated; repeated dates have inverse-frequency weights. Audit masks may overlap.',fontsize=8)
    fig.text(.10,.049,'Outer labels are removed from inner fits. The stress layer was added after preliminary audits; this is not independent confirmation.',fontsize=8)
    fig.text(.10,.025,'Neither range has a coverage guarantee. Observed-season coverage cannot validate the absent August-October records.',fontsize=8)
    save(fig,'U04_range_audit_and_season_support',pages)

def write_notes(d,ref,audit,gap,events):
    tn=events[events.Target=='TN'];audit_tn=audit[(audit.Target=='TN')&audit.Length.isin([7,15,30,115,150,162])]
    event_text='\n'.join(f"{r.Event}: {r.Days}天，TN实测{r.Observed_days}天、重建{r.Imputed_days}天；流量加权TN基准{r.Flow_weighted_Base:.3f} mg/L，参考范围{r.Flow_weighted_Lower_scenario:.3f}—{r.Flow_weighted_Upper_scenario:.3f} mg/L，压力范围{r.Flow_weighted_Stress_Lower:.3f}—{r.Flow_weighted_Stress_Upper:.3f} mg/L。" for r in tn.itertuples())
    audit_text='\n'.join(f"{r.Length}天：参考包络覆盖率{r.Envelope_coverage_pct:.1f}%，压力范围{r.Stress_coverage_pct:.1f}%；{r.Unique_dates}个不重复实测日期。" for r in audit_tn.itertuples())
    notes=f'''重建误差范围：使用说明与论文解释
版本：2026-11-16。统一61个事件日。原v2重建点值保持不变。

一、本次完成什么
已为7项水质、731个研究日建立逐日边际误差参考与方法情景包络，并进行额外嵌套遮挡的内部覆盖诊断。已生成4张英文科学图，每张含600 dpi JPG和矢量PDF。没有运行W2，没有转换NOx、LDOM等派生组分，没有构造完整联合概率不确定性。

二、范围怎样建立
1. 固定已有CIN_imputed_v2.csv的点值和真实缺测掩膜，不重新挑选最有利模型，不修改实测值。
2. 人工遮挡1、3、7、15、30、60、115、150、162天。1天使用40个残差参考块，3天24个，其余长度16个；块放置受真实观测密度限制，不足时用可用数量并记录。
3. 对四种方法分别计算留出残差e=实测-重建。每个日期在残差池中的总权重相同，避免重叠遮挡让某些日期被重复加权。残差的加权5%和95%分位数构成描述性参考；方向不跨0时扩到0，使原基准包含于范围内。下限仅按浓度非负截断。
4. 每个真实缺口使用与实际v2方法对应的残差池，参考长度向上取到最近测试长度，避免以更短遮挡直接代表更长缺口。这个长度迁移规则是透明的近似，并不意味着误差一定随长度单调。
5. 以原始实测值拟合四种方法，计算真实缺失日期的四方法最小—最大值。参考包络下限取“残差参考下限、方法最小值”的较小者；上限取“残差参考上限、方法最大值”的较大者。实测日期的上下限均固定为原始实测值。
6. 初步审计发现分位数参考在某些长缺口上覆盖不足，另加“压力范围”：用同类遮挡中已观察到的残差最小值和最大值扩展参考包络。这是对尾部敏感的压力定义，不通过反复调整倍率把审计覆盖率调到95%。因该层是在看过初步审计后增加，其后续覆盖诊断只属探索性，不能作为独立确认。

这种范围称为“残差与方法情景包络”或“探索性重建误差范围”。q05/q95来自经验分位数，绝不能因此就写成已校准的90%或95%预测区间。把不同方法的范围取并集，也不能获得概率保证。

建议后续W2试验分别比较参考包络和更宽的压力范围，明确使用哪一层，不把压力上限当作有特定发生概率的浓度。极值范围也可能漏掉未观测季节的新极端；不存在实测数据时，不能据此宣称“最坏情况已经覆盖”。

三、怎样检查范围是否有实际覆盖能力
1天设置12个外层审计块，3天8个，其余长度各6个。每个外层块用额外的24/16/10个内层遮挡块构造残差参考。外层全部标签从内层模型训练和残差校准中排除，再评价外层已知真值的覆盖率。固定v2尺度方法，不用外层覆盖率再调宽度。

内层与外层日期不交叉；不同内层块、不同外层场景之间仍可重叠。故结果只作为内部诊断，不是独立外部验证。现有数据和初步分数已经参与协议制定，也没有时间序列可交换性或跨季节稳定性的证明。

审计块的方法按最接近该长度的原v2 Selection_length及其既定方法确定，不在审计块上重新选择方法。这个诊断评价的是误差范围，不替代原v2选模流程的R2评价。真实缺口仍严格使用它原来记录的Method。

TN内部诊断：
{audit_text}

以上覆盖率描述可遮挡的实测日期，不是原始缺测日期的真实覆盖率；重复场景不是独立样本。部分长度上方法加入后覆盖率仍可能偏低，应披露并在后续压力试验考虑扩大有依据的范围；不能根据同一审计集反复调到目标比例后再称独立验证。

四、最大限制
七项水质在8、9、10月均没有原始实测值，7月实测覆盖也很低。方法彼此接近时可能同时存在系统偏差，因此方法包络即使较窄，也不证明真实峰值被覆盖。

NH4原始数值大量集中在0.025附近；是否为检出限替代需另核对检测元数据。其尾部误差由少数较高观测决定，极值压力范围会明显受这些记录影响，不能把这种敏感性当作已确定的真实事件峰值。

逐日Support标记为Seasonal_extrapolation，表示该缺口包含“原始月度实测占比低于20%”的日期；20%只是说明覆盖不足的标记阈值，不是统计有效性界线。Long_gap_extrapolation表示长期缺口。两者都应在论文中标为外推，不声称范围已在这些季节得到验证。

五、四次事件的TN条件范围
研究期第1天为2023-06-01。事件为21—41、389—398、413—429、436—448天，共61天，TN实测20天、重建41天。
{event_text}

事件范围通过相同QIN权重，对整条TN下限、基准、上限浓度轨迹分别求流量加权平均。它相当于所有缺失日期同时处在下限或上限的浓度情景，不是事件平均浓度的置信区间，也不是W2氮收支的最小—最大值。

六、后续用哪些文件和列
点值仍使用：paper_package/tables/CIN_imputed_v2.csv中的七个 *_Imputed 列。
配套误差范围使用：error_ranges/data/daily_reconstruction_error_ranges.csv。

新表的各变量列：
*_Base：与原v2完全一致的完整点值。
*_Residual_Lower / *_Residual_Upper：仅由留出残差形成的参考上下限。
*_Method_Min / *_Method_Max：四种方法在真实缺测日期的最小—最大值。
*_Range_Lower / *_Range_Upper：上述两部分的并集，称为参考包络。
*_Stress_Lower / *_Stress_Upper：进一步纳入已观察到的留出残差极值的压力范围，包含前述参考包络。
*_WasImputed：1为原始缺失，0为原始实测。
*_Gap_Days / *_Reference_Length / *_Method：真实缺口长度、参照长度和原v2方法。
*_Support：实测固定、内部经验参考、长缺口或季节外推标记。

Date为真实日期；Model_JDAY为研究期1—731；Original_JDY保留原2023-01-01起点的152—882。Event包含Non_WSRS、E1—E4。

不能独立给七个水质变量随机抽上下限并称为有效联合样本。TN、NH4及后续NOx、有机物与藻类氮需要先满足质量约束。原始直接测量的日期不扰动；派生组分的不确定性和掩膜来自其全部依赖变量。当前表不是W2可直接读取的CIN文件，也不是最终氮组分闭合的联合情景。

水温、流量、测量误差、转换系数和W2参数不确定性未包括在这些水质边际范围内。

七、图和表的论文用法
U01：按缺口长度的留出误差及方法差异，适合补充材料的误差参考依据。
U02：七变量完整点序列、参考包络及极值压力范围，替换不能再沿用的旧95%PI图；图注明确非概率范围。
U03：TN逐日包络、61天事件覆盖、各事件流量加权浓度情景。它是边界不确定性图，不是W2模拟结果图。
U04：参考包络与极值压力范围的内部覆盖诊断、月度实测覆盖，展示范围有效性的实证限制。

论文可写：
“在保持原始观测和既定点重建不变的条件下，按缺口长度汇集留出残差，构造经验残差参考，并与多方法重建范围取并集，形成探索性边界浓度情景包络。通过额外的嵌套连续块遮挡检查其内部覆盖表现。由于原始夏秋缺测域没有真值、遮挡场景可能重叠且存在时间依赖，该包络不解释为已校准的概率预测区间；后续用于条件边界敏感性试验。”

英文方法段和图注见range_methods_and_captions_EN.txt。

八、复现与完整性检查
运行code/build_error_ranges.py后运行code/plot_error_ranges.py。
data/integrity_checks.csv核对七项变量的基准未改、实测上下限固定、上下限有限、非负下限及基准落在范围中。
notes/range_protocol.json记录源文件SHA256、随机种子、所有规则及局限。
data/reference_mask_manifest.csv和nested_range_audit_manifest.csv给出具体参考/审计日期、可用数量和内层拟合规模。
本次没有使用旧十组W2情景的95%倍率，也没有把随机点高R2换算为误差区间。

参考资料：
Bikše et al. (2023), Journal of Hydrology 620, 129424. https://doi.org/10.1016/j.jhydrol.2023.129424
Oliveira et al. (2024), Split Conformal Prediction and Non-Exchangeable Data, JMLR 25(225). https://jmlr.org/papers/v25/23-1553.html
后者说明时间相关和不可交换数据中的覆盖保证需要附加理论条件；本分析未验证这些条件，不宣称符合其定理。
'''
    (OUT/'notes/误差范围_使用说明与论文写法.txt').write_text(notes,encoding='utf-8-sig')
    english='''Methods for exploratory reconstruction-error ranges

The v2 point reconstructions and original measured values were held fixed. Artificial continuous masks of 1, 3, 7, 15, 30, 60, 115, 150 and 162 days were introduced where measured reference values were available. Residual references used 40 masks for 1-day gaps, 24 for 3-day gaps and 16 for other durations, subject to placement availability. For each method and duration, actual-minus-predicted residuals were pooled with inverse multiplicity weights so that each evaluated calendar date had equal total weight. The weighted 5th and 95th residual percentiles were added to the frozen point reconstruction, expanded to include that point value and truncated at zero. Each real gap used the next tested duration at least as long as the gap. The resulting residual reference was combined with the minimum-maximum range of four full-data reconstruction methods, producing an exploratory marginal concentration envelope on missing dates only. Following undercoverage in preliminary internal audits, an additional stress range was defined using the minimum and maximum observed held-out residuals, together with the reference envelope. This tail-sensitive extension was not tuned to attain a specified coverage percentage, and its subsequent coverage diagnostics remain exploratory rather than independent confirmation.

Additional nested audits used 12 outer masks at 1 day, eight at 3 days and six at other durations. Every outer block was removed from all corresponding inner model fits and residual-calibration labels. Inner residual references used up to 24, 16 and ten masks, respectively. Fixed v2 methods were evaluated without retuning on outer coverage. Empirical coverage was summarized with inverse date-multiplicity weights, and unique reference-date counts were reported. Overlapping masks, temporal dependence, protocol development after preliminary analyses and missing summer/autumn observations preclude an independent or distribution-free coverage guarantee. The empirical percentile construction and method union are therefore not interpreted as calibrated 90% or 95% prediction intervals. The seven marginal envelopes are not a jointly constrained nitrogen-species ensemble.

For each audit duration, the fixed method associated with the nearest available selection-length level in the frozen v2 actual-gap records was used. This is a range diagnostic rather than a replacement evaluation of the original nested model-selection pipeline. Each original missing episode retained its recorded v2 method when forming the final daily ranges.

U01. Held-out residual magnitude by calendar-block duration. The line gives the weighted 90th absolute-error percentile for the frozen v2 method associated with each scale, divided by the constituent's global observed standard deviation. Shading spans the corresponding values across four reconstruction methods. Neither the percentile nor the method range is a probability guarantee for original missing dates.

U02. Frozen daily v2 reconstructions and exploratory error ranges. Blue shading denotes the empirical residual-plus-method reference envelope; gray shading denotes the additional extreme-residual stress range. Shading is restricted to originally missing dates, and all measured values are fixed. The original August-October absence of measurements prevents direct validation of those seasonal reconstructions. Neither range is a calibrated prediction interval or a proven worst-case bound.

U03. TN concentration scenarios and observed-data availability for 61 WSRS-associated days. (a) Daily point reconstruction, ranges and event windows. (b) QIN-weighted mean TN under baseline, all-lower and all-upper missing-date trajectories for the reference (blue) and stress (gray) layers. (c) Measured and reconstructed event-day fractions. Panel (b) gives conditional boundary-concentration scenarios, not confidence intervals, jointly constrained nitrogen-species scenarios or W2 budget bounds.

U04. Internal coverage diagnostics and original seasonal observation support. Panels (a) and (b) give inverse-date-multiplicity-weighted point coverage of the reference envelope and extreme-residual stress range; n denotes unique measured dates evaluated. Each outer block was excluded from its inner calibration fits. The stress layer was added after preliminary internal audits and its diagnostics do not constitute independent confirmation. Panel (c) gives measured-day fractions by calendar month. These diagnostics do not validate the original unobserved summer/autumn intervals.
'''
    (OUT/'notes/range_methods_and_captions_EN.txt').write_text(english,encoding='utf-8')

def main():
    source=pd.read_csv(ROOT/'paper_package/tables/CIN_imputed_v2.csv',parse_dates=['Date'])
    d=pd.read_csv(DATA/'daily_reconstruction_error_ranges.csv',parse_dates=['Date'])
    ref=pd.read_csv(DATA/'residual_reference_by_length_method.csv')
    audit=pd.read_csv(DATA/'nested_range_audit_summary.csv')
    monthly=pd.read_csv(DATA/'monthly_observation_support.csv')
    gaps=pd.read_csv(ROOT/'paper_package/tables/Table_S6_final_gap_methods.csv')
    actual_gap=pd.read_csv(DATA/'actual_gap_range_sources.csv')
    events=pd.read_csv(DATA/'event_concentration_ranges_61days.csv',parse_dates=['Start','End'])
    with PdfPages(OUT/'All_error_range_figures.pdf') as pdf:
        fig1(ref,source,gaps,pdf);fig2(d,source,pdf);fig3(d,source,events,pdf);fig4(audit,monthly,pdf)
    write_notes(d,ref,audit,actual_gap,events)
    thumbs=[]
    for p in sorted(FIG.glob('*.jpg')):
        im=Image.open(p).convert('RGB');im.thumbnail((1000,1100))
        tile=Image.new('RGB',(1050,1150),'#e8edef');tile.paste(im,((1050-im.width)//2,(1120-im.height)//2+25))
        ImageDraw.Draw(tile).text((15,8),p.stem,fill='black');thumbs.append(tile)
    sheet=Image.new('RGB',(2100,2300),'white')
    for i,im in enumerate(thumbs):sheet.paste(im,((i%2)*1050,(i//2)*1150))
    sheet.save(OUT/'qa/figure_contact_sheet.jpg',quality=95)
    print('Created 4 JPG/PDF figures, a combined PDF, and manuscript notes.')

if __name__=='__main__':main()
