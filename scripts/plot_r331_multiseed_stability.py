#!/usr/bin/env python3
"""Publication-style R331 five-seed stability plot from result.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--result",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    data=json.loads(args.result.read_text(encoding="utf-8"));agg=data["aggregate"];seeds=[str(s) for s in data["seeds"]]
    specs=[("overall_instance_metrics","dsc","Overall DSC",True),
           ("overall_instance_metrics","nsd_2px","Overall NSD@2px",True),
           ("overlap_region_metrics","nsd_2px","Overlap NSD@2px",True),
           ("overlap_pair_intersection_metrics","msd_px","Pair MSD (px)",False)]
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],"font.size":9,
                         "axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":300})
    fig,axes=plt.subplots(1,4,figsize=(10.8,2.65));color="#0072B2";base_color="#D55E00"
    for ax,(section,metric,label,higher) in zip(axes,specs):
        item=agg[section][metric];values=np.asarray([item["values"][s] for s in seeds]);x=np.arange(len(seeds))
        ax.plot(x,values,"o-",color=color,lw=1.15,ms=4,label="R330")
        ax.axhline(item["baseline"],color=base_color,ls="--",lw=1.05,label="R325")
        ax.fill_between([-0.3,len(seeds)-0.7],item["mean"]-item["std"],item["mean"]+item["std"],color=color,alpha=0.10)
        ax.set_xticks(x);ax.set_xticklabels(seeds,rotation=35,ha="right");ax.set_xlabel("Seed");ax.set_ylabel(label)
        ax.ticklabel_format(axis="y",style="plain",useOffset=False);ax.margins(x=0.06)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="upper center",ncol=2,frameon=False,bbox_to_anchor=(0.5,1.03))
    fig.tight_layout(pad=0.7,w_pad=1.15);fig.savefig(args.output/"r331_multiseed_stability.pdf",bbox_inches="tight")
    fig.savefig(args.output/"r331_multiseed_stability.png",bbox_inches="tight");plt.close(fig)


if __name__=="__main__":main()
