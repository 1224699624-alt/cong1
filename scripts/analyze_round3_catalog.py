"""Generate evidence-first Round3 comparison figures from saved per-case data."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'outputs/v3_round3';F=D/'analysis/figures';F.mkdir(parents=True,exist_ok=True)

def read(name):return json.loads((D/name).read_text())

def tsrs():
    a={r['case']:r for r in read('TSRS_A_R201.json')['per_case']};b={r['case']:r for r in read('TSRS_B_R201.json')['per_case']}
    metrics=[('Dice','dice',1),('Recall','recall',1),('Boundary F1','boundary_f1',1),('Surface Dice 2px','surface_dice_2px',1),('Gap FP','gap_region_fp_rate',-1)]
    rng=np.random.default_rng(3);fig,axes=plt.subplots(1,2,figsize=(7,3.2),gridspec_kw={'width_ratios':[2,1]})
    vals=[]
    for _,key,direction in metrics:vals.append(np.array([(b[c][key]-a[c][key])*direction*1000 for c in sorted(a)]))
    axes[0].boxplot(vals,tick_labels=[m[0] for m in metrics],showfliers=False,medianprops={'color':'black'})
    for i,v in enumerate(vals,1):axes[0].scatter(i+rng.normal(0,.045,len(v)),v,s=5,alpha=.25,color='#0072B2')
    axes[0].axhline(0,color='black',lw=.8);axes[0].set_ylabel('Favorable B − A delta (×10⁻³)');axes[0].tick_params(axis='x',rotation=35,labelsize=8)
    dist=[np.array([(a[c][k]-b[c][k]) for c in sorted(a)]) for k in ['hd95_px','assd_px']]
    axes[1].boxplot(dist,tick_labels=['HD95','ASSD'],showfliers=False,medianprops={'color':'black'})
    for i,v in enumerate(dist,1):axes[1].scatter(i+rng.normal(0,.035,len(v)),v,s=5,alpha=.25,color='#E69F00')
    axes[1].axhline(0,color='black',lw=.8);axes[1].set_ylabel('Favorable A − B distance (px)')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(F/'figure-01-tsrs-b-vs-a-paired.pdf',bbox_inches='tight');plt.close(fig)

def ram():
    x=read('PILOT_RESULTS.json')['RAM'];arms=['B','C','D'];spec=[('Dice','dice',1),('IoU','iou',1),('NSD','nsd',1),('MSD','msd',-1),('O-DSC','overlap_dsc',1),('O-NSD','overlap_nsd',1),('O-MSD','overlap_msd',-1),('P-DSC','pair_dsc',1),('P-NSD','pair_nsd',1),('P-MSD','pair_msd',-1)]
    values=np.array([[(x[a]['metrics'][key]-x['A']['metrics'][key])*direction*1000 for _,key,direction in spec] for a in arms])
    fig,ax=plt.subplots(figsize=(7,2.5));mx=max(abs(values.min()),abs(values.max()));im=ax.imshow(values,cmap='RdBu',vmin=-mx,vmax=mx,aspect='auto')
    ax.set_xticks(range(len(spec)),[s[0] for s in spec],rotation=35,ha='right');ax.set_yticks(range(3),arms)
    for i in range(3):
        for j in range(len(spec)):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=6,color='black')
    cb=fig.colorbar(im,ax=ax,pad=.02);cb.set_label('Favorable delta vs A (×10⁻³)')
    fig.tight_layout();fig.savefig(F/'figure-02-ram-arms-vs-a-heatmap.pdf',bbox_inches='tight');plt.close(fig)

if __name__=='__main__':tsrs();ram();print(F)
