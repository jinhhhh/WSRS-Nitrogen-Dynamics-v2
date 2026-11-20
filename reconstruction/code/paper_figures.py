"""Generate six English journal figures, each as 600-dpi JPG and vector PDF."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap, TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageOps, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'regenerated/paper_package'; FIG=OUT/'figures'; TAB=OUT/'tables'; NOTE=OUT/'notes'
TARGETS=['NH4','CODMn','Cond','DO','TP','TN','Chl_a']
LABELS={'NH4':r'NH$_4^+$-N','CODMn':r'COD$_{Mn}$','Cond':'Conductivity','DO':'DO','TP':'TP','TN':'TN','Chl_a':'Chl-a'}
UNITS={t:r'mg L$^{-1}$' for t in TARGETS};UNITS['Cond']=r'$\mu$S cm$^{-1}$'
COL={'Linear':'#74828f','PCHIP':'#db8b27','ExtraTrees':'#7774ad','ET_bridge':'#438779','Nested_selected':'#176d9a'}
EVENT_WINDOWS=[(21,41),(389,398),(413,429),(436,448)]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
                     'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white','figure.facecolor':'white'})

def intervals(mask):
    a=np.r_[False,np.asarray(mask,bool),False].astype(int)
    return list(zip(np.where(np.diff(a)==1)[0],np.where(np.diff(a)==-1)[0]))

def fmt(v):
    if not np.isfinite(v):return 'NA'
    if abs(v)>=100:return f'{v:.1e}'
    return f'{v:.2f}'

def setup_time(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.grid(axis='y',color='#e4e8eb',lw=.5)

def save(fig,stem,pages):
    fig.savefig(FIG/(stem+'.jpg'),dpi=600,pil_kwargs={'quality':95})
    fig.savefig(FIG/(stem+'.pdf'))
    pages.savefig(fig)
    plt.close(fig)

def figure1(d,availability,pages):
    fig=plt.figure(figsize=(11.4,7.2));gs=fig.add_gridspec(2,2,height_ratios=[1,1.05],width_ratios=[1,1],hspace=.42,wspace=.28)
    ax=fig.add_subplot(gs[0,:]); arr=d[TARGETS].isna().to_numpy().T.astype(int)
    first=mdates.date2num(d.Date.min()); last=mdates.date2num(d.Date.max()+pd.Timedelta(days=1))
    ax.imshow(arr,aspect='auto',interpolation='nearest',cmap=ListedColormap(['#248797','#eeae68']),
              extent=(first,last,6.5,-.5))
    ax.set_yticks(range(7),[LABELS[t] for t in TARGETS]);setup_time(ax)
    ax.set_xlim(d.Date.min(),d.Date.max())
    ax.set_title('(a) Daily availability of measured boundary concentrations',loc='left',fontweight='bold')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#248797',label='Observed'),Patch(color='#eeae68',label='Missing')],
              loc='upper right',ncol=2,bbox_to_anchor=(1,1.23),frameon=False)
    for idx,weight in enumerate(['count','days']):
        ax=fig.add_subplot(gs[1,idx]); bottoms=np.zeros(7)
        for cat,color in zip(['1-3 d','4-30 d','>30 d'],['#76b9b2','#b9d9c7','#e79a54']):
            vals=[]
            for t in TARGETS:
                lens=[b-a for a,b in intervals(d[t].isna())]
                which=lambda l:cat==('1-3 d' if l<=3 else '4-30 d' if l<=30 else '>30 d')
                vals.append(100*sum((1 if weight=='count' else l) for l in lens if which(l))/sum(1 if weight=='count' else l for l in lens))
            bars=ax.bar(np.arange(7),vals,bottom=bottoms,color=color,label=cat,width=.72)
            for i,v in enumerate(vals):
                if v>=12:ax.text(i,bottoms[i]+v/2,f'{v:.0f}%',ha='center',va='center',fontsize=8,color='#23313a')
            bottoms+=vals
        ax.set_ylim(0,100);ax.set_ylabel('Share of gap episodes (%)' if weight=='count' else 'Share of missing days (%)')
        ax.set_xticks(range(7),[LABELS[t] for t in TARGETS],rotation=35,ha='right')
        ax.set_title('(b) Weighting by gap episodes' if weight=='count' else '(c) Weighting by missing days',loc='left',fontweight='bold')
        ax.grid(axis='y',alpha=.15)
        if idx==1:ax.legend(loc='upper center',bbox_to_anchor=(.5,-.29),ncol=3,frameon=False)
    fig.subplots_adjust(left=.09,right=.98,top=.91,bottom=.16)
    fig.text(.09,.025,'Study period: 1 Jun 2023 - 31 May 2025 (731 days). Long gaps dominate missing-day totals.',fontsize=9)
    save(fig,'Fig01_data_availability',pages)

def figure2(met,pages):
    lengths=[0,1,3,7,15,30,60,115,150]
    fig,axs=plt.subplots(2,1,figsize=(11.5,7.3),sharex=True)
    cmap=LinearSegmentedColormap.from_list('skill',['#b95649','#fbf7f0','#1b858c'])
    norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1)
    for k,method in enumerate(['PCHIP','Nested_selected']):
        ax=axs[k];mat=np.full((7,9),np.nan)
        for i,t in enumerate(TARGETS):
            for j,l in enumerate(lengths):
                row=met[(met.Target==t)&(met.Length==l)&(met.Model==('PCHIP' if l==0 else method))]
                if len(row):mat[i,j]=row.R2.iloc[0]
        im=ax.imshow(mat,cmap=cmap,norm=norm,aspect='auto')
        for i in range(7):
            for j in range(9):
                val=mat[i,j];color='white' if np.isfinite(val) and abs(val)>.65 else '#23313a'
                ax.text(j,i,fmt(val),ha='center',va='center',color=color,fontsize=8.5)
        ax.set_yticks(range(7),[LABELS[t] for t in TARGETS])
        ax.set_title('(a) Fixed PCHIP baseline' if k==0 else '(b) Methods selected by inner masks for each outer block',loc='left',fontweight='bold')
        ax.set_xticks(range(9),['Random*','1 d','3 d','7 d','15 d','30 d','60 d','115 d','150 d'])
        ax.set_xticks(np.arange(-.5,9,1),minor=True);ax.set_yticks(np.arange(-.5,7,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=1);ax.tick_params(which='minor',bottom=False,left=False)
    axs[1].set_xlabel('Artificial masking design / continuous calendar-block length')
    fig.subplots_adjust(left=.12,right=.98,top=.94,bottom=.28,hspace=.25)
    cb=fig.colorbar(im,cax=fig.add_axes([.35,.155,.37,.025]),orientation='horizontal',extend='min')
    cb.set_label(r'Pooled $R^2$ (colors saturate below -1; annotations retain values)')
    fig.text(.08,.055,'* Random: fixed PCHIP, 20% of measured dates hidden; 5 repeats. Blocked: 5 masks per length.',fontsize=8)
    fig.text(.08,.028,'Repeated masks may overlap. Scores assess observed calendar domains, not the unobserved summer/autumn gaps.',fontsize=8)
    save(fig,'Fig02_validation_R2',pages)

def figure3(met,rep,availability,pages):
    fig,axs=plt.subplots(4,2,figsize=(11.5,10.5));axes=axs.flatten()
    for i,t in enumerate(TARGETS):
        ax=axes[i];sd=float(availability.loc[availability.Target==t,'Observed_sd'].iloc[0])
        xpos={v:j for j,v in enumerate([1,3,7,15,30,60,115,150])}
        for method in ['PCHIP','ExtraTrees','ET_bridge','Nested_selected']:
            g=met[(met.Target==t)&(met.Design=='Blocked')&(met.Model==method)].sort_values('Length')
            ax.plot(g.Length.map(xpos),g.RMSE/sd,'o-',ms=3,lw=1.2,label=method.replace('_',' '),color=COL[method])
        r=rep[(rep.Target==t)&(rep.Design=='Blocked')&(rep.Model=='Nested_selected')]
        ax.scatter(r.Length.map(xpos),r.RMSE/sd,s=8,alpha=.2,color=COL['Nested_selected'],zorder=1)
        ax.axhline(1,color='#8c9399',ls=':',lw=.8)
        ax.set_xlim(-.25,7.25)
        ax.set_xticks(range(8),['1','3','7','15','30','60','115','150']);ax.grid(alpha=.15)
        ax.set_title(f'({chr(97+i)}) '+LABELS[t],loc='left',fontweight='bold')
        ax.set_ylabel('RMSE / observed SD')
        if i>=5:ax.set_xlabel('Mask length (days; ordered categories)')
    axes[7].axis('off');handles,labels=axes[0].get_legend_handles_labels()
    axes[7].legend(handles,labels,loc='upper left',frameon=False,fontsize=9,bbox_to_anchor=(.0,1.08))
    axes[7].text(.03,.02,'Faint dots: individual outer masks.\nThey are overlapping scenarios,\nnot independent replicates or a 95% CI.',transform=axes[7].transAxes,fontsize=8,linespacing=1.5)
    fig.subplots_adjust(left=.08,right=.98,top=.90,bottom=.09,hspace=.50,wspace=.25)
    fig.suptitle('Reconstruction error increases with the length and placement of missing episodes',x=.08,ha='left',fontsize=12,fontweight='bold')
    fig.text(.08,.022,'SD is computed from all available measured values of each variable and used only to normalize error magnitudes.',fontsize=8)
    save(fig,'Fig03_error_vs_gap_length',pages)

def figure4(pred,pages):
    fig,axs=plt.subplots(2,2,figsize=(9.5,8.2));tn=pred[pred.Target=='TN']
    for i,length in enumerate([0,7,30,150]):
        ax=axs.flat[i];method='PCHIP' if length==0 else 'Nested_selected'
        g=tn[(tn.Length==length)&(tn.Model==method)]
        y=g.Actual.to_numpy();p=g.Predicted.to_numpy();v=1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)
        rmse=np.sqrt(np.mean((y-p)**2));limit=max(y.max(),p.max())*1.06
        ax.scatter(y,p,s=17,alpha=.45,facecolor='#247c98',edgecolors='none')
        ax.plot([0,limit],[0,limit],ls='--',color='#66727a',lw=1)
        ax.set(xlim=(0,limit),ylim=(0,limit),xlabel=r'Observed TN (mg L$^{-1}$)',ylabel=r'Reconstructed TN (mg L$^{-1}$)')
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
        ax.set_title(f'({chr(97+i)}) '+('Random 20% / fixed PCHIP' if length==0 else f'{length}-day blocks / inner-selected'),loc='left',fontweight='bold')
        ax.text(.04,.96,f'$R^2$ = {v:.3f}\nRMSE = {rmse:.3f}\nPairs = {len(g)}; dates = {g.Date.nunique()}',
                transform=ax.transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    fig.subplots_adjust(left=.09,right=.98,top=.94,bottom=.10,hspace=.3,wspace=.28)
    fig.text(.09,.025,'Each point is a held-out prediction pair; repeated masks can evaluate the same measured date more than once.',fontsize=8)
    save(fig,'Fig04_TN_validation_scatter',pages)

def figure5(d,pages):
    fig,axs=plt.subplots(7,1,figsize=(11.5,11.3),sharex=True)
    for i,t in enumerate(TARGETS):
        ax=axs[i];obs=d[t].notna()
        ax.plot(d.Date,d[t],color='#273c49',lw=.6,alpha=.6)
        ax.scatter(d.loc[obs,'Date'],d.loc[obs,t],s=5,color='#273c49',label='Measured',zorder=3)
        for a,b in intervals(~obs):
            left=max(0,a-1);right=min(len(d),b+1)
            ax.plot(d.Date.iloc[left:right],d[t+'_Imputed'].iloc[left:right],color='#d97c2a',lw=.95)
            if b-a>30:ax.axvspan(d.Date[a],d.Date[b-1],color='#c4cbd0',alpha=.22,zorder=0)
        ax.set_ylabel(LABELS[t]+'\n'+UNITS[t],fontsize=8)
        ax.text(.01,.83,f'({chr(97+i)})',transform=ax.transAxes,fontsize=9,fontweight='bold')
        setup_time(ax)
        ax.set_xlim(d.Date.min(),d.Date.max())
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(handles=[Line2D([],[],color='#273c49',marker='o',ms=3,lw=.6,label='Measured records'),
                        Line2D([],[],color='#d97c2a',lw=1,label='Reconstruction on missing dates'),
                        Patch(facecolor='#c4cbd0',alpha=.45,label='Actual gaps >30 days')],
               loc='upper center',bbox_to_anchor=(.55,.978),ncol=3,frameon=False)
    fig.subplots_adjust(left=.1,right=.98,top=.94,bottom=.07,hspace=.20)
    fig.text(.10,.018,'Completed boundary input series (v2). Shading marks data absence, not an uncertainty interval; long-gap dynamics remain unverified.',fontsize=8)
    save(fig,'Fig05_completed_boundary_series',pages)

def figure6(d,pred,manifest,pages):
    fig,axs=plt.subplots(3,1,figsize=(11.5,8.1))
    for i,length in enumerate([15,115,150]):
        ax=axs[i];g=manifest[(manifest.Target=='TN')&(manifest.Length==length)].sort_values('Start').iloc[0]
        start=pd.Timestamp(g.Start);end=pd.Timestamp(g.End)
        show=d.Date.between(start-pd.Timedelta(days=12),end+pd.Timedelta(days=12))
        truth=d[show];ax.plot(truth.Date,truth.TN,'o-',color='#344a55',ms=3,lw=.7,label='Available reference observations')
        for method,color,ls in [('PCHIP',COL['PCHIP'],'--'),('Nested_selected',COL['Nested_selected'],'-')]:
            p=pred[(pred.Target=='TN')&(pred.Length==length)&(pred.Repeat==g.Repeat)&(pred.Model==method)]
            p=p.sort_values('Date');ax.plot(p.Date,p.Predicted,ls=ls,color=color,lw=1.5,label='PCHIP' if method=='PCHIP' else 'Inner-selected reconstruction')
        ax.axvspan(start,end,color='#d9e5eb',alpha=.45)
        selected=pred[(pred.Target=='TN')&(pred.Length==length)&(pred.Repeat==g.Repeat)&(pred.Model=='Nested_selected')].Selected.iloc[0]
        ax.set_title(f'({chr(97+i)}) {length}-day mask: {start:%Y-%m-%d} to {end:%Y-%m-%d}; selected: {selected}',loc='left',fontsize=10)
        ax.set_ylabel(r'TN (mg L$^{-1}$)');ax.grid(axis='y',alpha=.2)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4,maxticks=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.985),ncol=3,frameon=False,fontsize=8.5)
    fig.subplots_adjust(left=.09,right=.98,top=.91,bottom=.10,hspace=.48)
    fig.text(.09,.028,'Examples are the earliest-starting outer mask at each length, chosen by date rather than by performance. Shading: hidden block.',fontsize=8)
    save(fig,'Fig06_TN_mask_examples',pages)

def event_tables(d):
    rows=[];all_idx=[]
    for k,(a,b) in enumerate(EVENT_WINDOWS,1):
        all_idx.extend(range(a-1,b))
        e=d.iloc[a-1:b]
        for t in TARGETS:
            missing=e[t].isna()
            conditional_mass=e.QIN*e[t+'_Imputed']
            rows.append(dict(Event=f'E{k}',Study_day_start=a,Study_day_end=b,
                Start=e.Date.min(),End=e.Date.max(),Duration_days=len(e),Target=t,
                Observed_days=int((~missing).sum()),Missing_days=int(missing.sum()),
                Observed_percent=100*(~missing).mean(),
                Missing_flow_weighted_percent=100*e.loc[missing,'QIN'].sum()/e.QIN.sum(),
                Estimated_load_share_on_imputed_dates_percent=100*conditional_mass[missing].sum()/conditional_mass.sum()))
    et=pd.DataFrame(rows)
    et.to_csv(TAB/'Table_3_event_observation_coverage.csv',index=False,encoding='utf-8-sig')
    et[et.Target=='TN'].to_csv(TAB/'Table_3_TN_event_coverage.csv',index=False,encoding='utf-8-sig')
    return et

def figure7(d,events,pages):
    fig,axs=plt.subplots(1,2,figsize=(11.5,5.5),gridspec_kw={'width_ratios':[1.1,1]})
    ax=axs[0];mat=np.array([[events[(events.Target==t)&(events.Event==f'E{k}')].Observed_percent.iloc[0] for k in range(1,5)] for t in TARGETS])
    ax.imshow(mat,vmin=0,vmax=100,aspect='auto',cmap=LinearSegmentedColormap.from_list('availability',['#f4e3d3','#2e8790']))
    for i,t in enumerate(TARGETS):
        for j in range(4):
            r=events[(events.Target==t)&(events.Event==f'E{j+1}')].iloc[0]
            ax.text(j,i,f'{r.Observed_days}/{r.Duration_days}\n({r.Observed_percent:.0f}%)',ha='center',va='center',fontsize=9,color='white' if r.Observed_percent>65 else '#34434b')
    labels=[f'E{k}' for k in range(1,5)]
    ax.set_xticks(range(4),labels);ax.set_yticks(range(7),[LABELS[t] for t in TARGETS])
    ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,7,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=1);ax.tick_params(which='minor',bottom=False,left=False)
    ax.set_title('(a) Measured days / event duration',loc='left',fontweight='bold')
    ax=axs[1];tn=events[events.Target=='TN'];xx=np.arange(4)
    ax.bar(xx-.18,100-tn.Observed_percent,width=.36,color='#ecb16d',label='Share of event dates imputed')
    ax.bar(xx+.18,tn.Estimated_load_share_on_imputed_dates_percent,width=.36,color='#257b90',label='Estimated TN load on imputed dates')
    for i,(p,q) in enumerate(zip(100-tn.Observed_percent,tn.Estimated_load_share_on_imputed_dates_percent)):
        ax.text(i-.18,p+2,f'{p:.0f}',ha='center',fontsize=8)
        ax.text(i+.18,q+2,f'{q:.0f}',ha='center',fontsize=8)
    ax.set_xticks(xx,labels);ax.set_ylim(0,115);ax.set_ylabel('Share (%)');ax.grid(axis='y',alpha=.15)
    ax.set_title('(b) Dependence of TN boundary input on reconstruction',loc='left',fontweight='bold',fontsize=9)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),frameon=False,fontsize=8)
    fig.subplots_adjust(left=.10,right=.98,top=.9,bottom=.28,wspace=.30)
    dates='; '.join(f'{r.Event}: {r.Start:%Y-%m-%d} to {r.End:%Y-%m-%d}' for r in tn.itertuples())
    fig.text(.10,.09,dates[:dates.index('; E3')],fontsize=8)
    fig.text(.10,.06,dates[dates.index('E3'):],fontsize=8)
    fig.text(.10,.025,'58 inclusive event days supplied by the author. Load shares use v2 concentrations and daily QIN; they are conditional estimates, not validation.',fontsize=7.5)
    save(fig,'Fig07_event_observation_coverage',pages)

def captions():
    entries=[
    ('Fig01','Daily availability and duration of missing boundary-concentration records. (a) Observed and missing daily records. (b) Proportions of gap episodes by duration. (c) Proportions of missing days by duration. Gaps exceeding 30 days dominate the missing-day totals despite accounting for few episodes.'),
    ('Fig02','Internal reconstruction performance under repeated random-point and continuous-block masking. Each cell gives pooled R2 over five masks; random-point masking hides 20% of measured dates. PCHIP is fixed before comparison. In panel (b), the blocked-test method is selected using two inner masks after removing the outer-block labels. Colors saturate below R2=-1 while annotations retain the scores. Overlapping masks are not independent observations.'),
    ('Fig03','Reconstruction error as a function of mask duration. Lines denote pooled RMSE divided by the standard deviation of available observations for each constituent. Faint dots show individual outer-mask scores for the inner-selected protocol, not confidence intervals. The availability and season of measurements constrain which long gaps can be tested.'),
    ('Fig04','Observed versus reconstructed total nitrogen (TN) under random-point masking and 7-, 30-, and 150-day blocked masking. The dashed line denotes 1:1 agreement. Pair counts include repeated evaluation of the same date; unique-date counts are reported separately. Blocked results use inner-selected methods and constitute internal reconstruction validation.'),
    ('Fig05','Observed and reconstructed daily boundary-concentration series. Original measured values are preserved; orange segments are reconstructions on missing dates. Gray shading indicates actual gaps longer than 30 days and does not represent a prediction interval. The reconstructions do not validate dynamics within the original unobserved intervals.'),
    ('Fig06','TN reconstruction examples for 15-, 115-, and 150-day continuous masks. The earliest-starting outer mask at each duration was selected by calendar date, not by its score. Reference observations inside the shaded interval were hidden from all corresponding model fits and internal method-selection steps. PCHIP and the inner-selected reconstruction are shown for comparison.'),
    ('Fig07','Event-specific availability of measured water-quality boundary records. Event intervals were supplied by the author as study days 21-41, 389-398, 413-429, and 436-445 (day 1 = 1 June 2023; inclusive boundaries, totaling 58 days). Panel (a) gives measured-day counts and percentages. Panel (b) compares the share of TN dates reconstructed with the share of estimated incoming TN load assigned to those dates, computed as sum(QIN x reconstructed TN) on missing dates divided by the event total. These load shares are conditional on the completed v2 series and are not independent validation or uncertainty estimates.')]
    (NOTE/'figure_captions_EN.txt').write_text('\n\n'.join(k+'. '+v for k,v in entries),encoding='utf-8')

def main():
    FIG.mkdir(exist_ok=True,parents=True)
    d=pd.read_csv(TAB/'CIN_imputed_v2.csv',parse_dates=['Date'])
    av=pd.read_csv(TAB/'Table_1_data_availability.csv')
    met=pd.read_csv(TAB/'Table_S2_all_methods_metrics.csv')
    rep=pd.read_csv(TAB/'Table_S3_mask_level_metrics.csv')
    pred=pd.read_csv(TAB/'validation_predictions_v2.csv',parse_dates=['Date'])
    manifest=pd.read_csv(TAB/'Table_S4_validation_masks.csv',parse_dates=['Start','End'])
    events=event_tables(d)
    with PdfPages(OUT/'All_figures.pdf') as pdf:
        figure1(d,av,pdf);figure2(met,pdf);figure3(met,rep,av,pdf)
        figure4(pred,pdf);figure5(d,pdf);figure6(d,pred,manifest,pdf)
        figure7(d,events,pdf)
    captions()
    # Contact sheet for quick visual QA, not a manuscript figure.
    thumbs=[]
    for p in sorted(FIG.glob('*.jpg')):
        im=Image.open(p).convert('RGB');im.thumbnail((1000,1000))
        tile=Image.new('RGB',(1050,1050),'#e7ecef')
        tile.paste(im,((1050-im.width)//2,(1020-im.height)//2+25))
        ImageDraw.Draw(tile).text((15,10),p.stem,fill='black')
        thumbs.append(tile)
    sheet=Image.new('RGB',(2100,4200),'white')
    for i,im in enumerate(thumbs):sheet.paste(im,((i%2)*1050,(i//2)*1050))
    sheet.save(OUT/'qa/contact_sheet.jpg',quality=95)
    print('Created 7 JPGs, 7 individual PDFs, and All_figures.pdf')

if __name__=='__main__':main()
