#!/usr/bin/env python3
"""Bounded exact-Q low-FPR evaluation with streamed AUC/AP histograms.

Only used if the original exact-sort evaluator's measured cost threatens the
two-GPU-hour lock.  It never trains a model or changes the frozen C0/probes.
Exact Q denominators, TP, FP and H0 counts are retained; ROC-AUC/AP use a
fixed 16,384-bin score histogram and are explicitly approximate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ram_highres_capture import load_frozen_c0
import run_v31_ram_mature as legacy
import audit_ram_source_correctability_response as old_audit
from run_ram_highres_readout import (
    FeatureTransform, Probe, feature_at, load_case_and_capture, probe_scores,
    sha256_file, write_json, ensure_disk_floor, load_samples,
    classification_metrics,
)


NBINS=16384


class Running:
    def __init__(self)->None:
        self.pos=np.zeros(NBINS,dtype=np.int64)
        self.neg=np.zeros(NBINS,dtype=np.int64)
        self.tp=self.fp=self.h0_hit=0
        self.h0_total=self.h0_q=self.q_total=0

    def add(self,score:np.ndarray,label:np.ndarray,h0:np.ndarray,
            threshold:float)->None:
        s=np.asarray(score,dtype=np.float32)
        y=np.asarray(label,dtype=bool)
        h=np.asarray(h0,dtype=bool)
        self.q_total+=len(s)
        bin_index=np.minimum((s*NBINS).astype(np.int64),NBINS-1)
        self.pos+=np.bincount(bin_index[y],minlength=NBINS)
        self.neg+=np.bincount(bin_index[~y],minlength=NBINS)
        chosen=s>=threshold
        self.tp+=int((chosen&y).sum())
        self.fp+=int((chosen&~y).sum())
        self.h0_hit+=int((chosen&h).sum())

    def merge_scored(self,other:'Running')->None:
        """Merge scored rows; H0 denominators are managed per case separately."""
        self.pos+=other.pos
        self.neg+=other.neg
        self.tp+=other.tp
        self.fp+=other.fp
        self.h0_hit+=other.h0_hit
        self.q_total+=other.q_total

    def metrics(self)->dict:
        p=int(self.pos.sum());n=int(self.neg.sum())
        if not p or not n:
            auc=ap=None
        else:
            lower=np.cumsum(self.neg)-self.neg
            auc=float((self.pos*(lower+.5*self.neg)).sum(dtype=np.float64)/(p*n))
            cp=np.cumsum(self.pos[::-1],dtype=np.float64)
            cn=np.cumsum(self.neg[::-1],dtype=np.float64)
            ap=float(((self.pos[::-1]/p)*(cp/np.maximum(cp+cn,1))).sum())
        return {'q_exact':self.q_total,'positive_exact':p,'negative_exact':n,
                'auc_hist_approx':auc,'ap_hist_approx':ap,
                'threshold_tp_exact':self.tp,'threshold_fp_exact':self.fp,
                'threshold_precision_exact':self.tp/max(self.tp+self.fp,1),
                'threshold_recall_exact':self.tp/max(p,1),
                'threshold_fpr_exact':self.fp/max(n,1),
                'h0_exact':self.h0_total,'h0_in_q_exact':self.h0_q,
                'h0_selected_exact':self.h0_hit,
                'h0_full_recall_exact':self.h0_hit/max(self.h0_total,1),
                'histogram_bins':NBINS}


def add_member_scores(running:list[Running],score:np.ndarray,label:np.ndarray,
                      h0:np.ndarray,member:np.ndarray,threshold:float)->None:
    """Update all 14 per-member histograms in two vectorized bincounts."""
    s=np.asarray(score,dtype=np.float32)
    y=np.asarray(label,dtype=bool)
    h=np.asarray(h0,dtype=bool)
    k=np.asarray(member,dtype=np.int64)
    bins=np.minimum((s*NBINS).astype(np.int64),NBINS-1)
    joined=k*NBINS+bins
    pos=np.bincount(joined[y],minlength=14*NBINS).reshape(14,NBINS)
    neg=np.bincount(joined[~y],minlength=14*NBINS).reshape(14,NBINS)
    selected=s>=threshold
    tp=np.bincount(k[selected&y],minlength=14)
    fp=np.bincount(k[selected&~y],minlength=14)
    hit=np.bincount(k[selected&h],minlength=14)
    total=np.bincount(k,minlength=14)
    for member_id,row in enumerate(running):
        row.pos+=pos[member_id]
        row.neg+=neg[member_id]
        row.tp+=int(tp[member_id]);row.fp+=int(fp[member_id])
        row.h0_hit+=int(hit[member_id]);row.q_total+=int(total[member_id])


def load_transform(path:Path)->FeatureTransform:
    obj=FeatureTransform.__new__(FeatureTransform)
    with np.load(path,allow_pickle=False) as z:
        for name in ('b_mean','b_std','h_mean','h_std','a_b','a_h'):
            setattr(obj,name,z[name])
    obj.input_dim=len(obj.b_mean)+32+14
    return obj


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    lock=json.loads(a.lock.read_text(encoding='utf-8'))
    if lock['status']!='LOCKED_STREAM_REPAIR_BEFORE_HOLDOUT_VAL':
        raise ValueError('stream repair not locked')
    for file,expected in lock['file_hashes'].items():
        if sha256_file(Path(file))!=expected:
            raise ValueError(f'hash mismatch: {file}')
    origin=Path(lock['original_run'])
    cal=json.loads((origin/'NATURAL_Q_probe_calibration.json').read_text())
    thresholds=cal['thresholds_from_calibration']
    if thresholds!=lock['thresholds']:
        raise ValueError('calibration thresholds changed')
    split=json.loads(Path(lock['split_manifest']).read_text())
    out=a.out
    out.mkdir(parents=True)
    torch.set_num_threads(int(lock['cpu_threads']))
    device=torch.device('cuda')
    started=time.monotonic();deadline=started+float(lock['max_seconds'])
    model,refiner,_=load_frozen_c0(Path(lock['c0']),device)
    frozen=(old_audit._module_state_digest(model),old_audit._module_state_digest(refiner))
    transform=load_transform(origin/'NORMALIZATION_AND_PROJECTION.npz')
    probes={}
    for arm in ('P0','P1'):
        payload=torch.load(origin/f'{arm}_PROBE.pth',map_location='cpu',weights_only=False)
        probe=Probe(payload['input_dim'],payload['hidden_width']).to(device)
        probe.load_state_dict(payload['model'],strict=True)
        probes[arm]=probe.eval().requires_grad_(False)
    results={}
    all_rows=[]
    for subset in ('probe_holdout','val'):
        source_split='val' if subset=='val' else 'train'
        cases=split['val_cases'] if subset=='val' else split['groups'][subset]
        dataset=legacy.NativeWristDataset(Path(lock['dataset']),source_split,False)
        idx={x.stem:i for i,x in enumerate(dataset.mask_files)}
        if not set(cases).issubset(idx):raise ValueError('missing case')
        aggregates={arm:Running() for arm in ('P0','P1')}
        member_aggr={arm:[Running() for _ in range(14)] for arm in ('P0','P1')}
        case_start=time.monotonic()
        for cno,case in enumerate(cases,1):
            if time.monotonic()>deadline:raise TimeoutError('stream repair wall budget')
            if cno%20==0:ensure_disk_floor(out,int(lock['disk_free_floor_bytes']))
            record,target,h0,q,_=load_case_and_capture(dataset,idx[case],model,refiner,
                device,Path(lock['oracle_h0_root']),source_split,parity=False)
            flat=np.flatnonzero(q.reshape(-1))
            k,y,x=np.unravel_index(flat,q.shape)
            case_aggr={arm:Running() for arm in ('P0','P1')}
            h_total=int(h0.sum());h_q=int((h0&q).sum())
            for arm in ('P0','P1'):
                aggregates[arm].h0_total+=h_total
                aggregates[arm].h0_q+=h_q
                case_aggr[arm].h0_total=h_total
                case_aggr[arm].h0_q=h_q
                for member in range(14):
                    member_aggr[arm][member].h0_total+=int(h0[member].sum())
                    member_aggr[arm][member].h0_q+=int((h0[member]&q[member]).sum())
            chunk=int(lock['score_chunk'])
            for start in range(0,len(flat),chunk):
                stop=start+chunk
                kk,yy,xx=k[start:stop],y[start:stop],x[start:stop]
                B,H=feature_at(record,kk,yy,xx)
                labels=target[kk,yy,xx]
                hflag=h0[kk,yy,xx]
                for arm in ('P0','P1'):
                    X=transform.apply(B,H,kk,arm)
                    score=probe_scores(probes[arm],X,device,int(lock['score_batch']))
                    case_aggr[arm].add(score,labels,hflag,thresholds[arm])
                    add_member_scores(member_aggr[arm],score,labels,hflag,kk,thresholds[arm])
            for arm in ('P0','P1'):
                aggregates[arm].merge_scored(case_aggr[arm])
                row={'subset':subset,'case':case,'member':-1,'arm':arm,
                     **case_aggr[arm].metrics()}
                all_rows.append(row)
            if cno%20==0:
                print(json.dumps({'phase':subset,'cases':cno,
                                  'elapsed':time.monotonic()-case_start}),flush=True)
        result={'status':'COMPLETE','subset':subset,'calibrated_thresholds':thresholds,
                'population':'ALL_Q_EXACT','AUC_AP':'fixed_16384_bin_approximation',
                'arms':{arm:aggregates[arm].metrics() for arm in ('P0','P1')},
                'elapsed_seconds':time.monotonic()-case_start,
                'test_used':False,'C0_updates':0}
        results[subset]=result
        write_json(out/f'NATURAL_Q_{subset}_STREAM.json',result)
        for arm in ('P0','P1'):
            for member in range(14):
                all_rows.append({'subset':subset,'case':'ALL','member':member,'arm':arm,
                                 **member_aggr[arm][member].metrics()})
        if subset=='probe_holdout':
            write_json(out/'HOLDOUT_COMPLETE_BEFORE_VAL.json',{
                'status':'HOLDOUT_COMPLETE_BEFORE_VAL','thresholds':thresholds,
                'P0_checkpoint_sha256':sha256_file(origin/'P0_PROBE.pth'),
                'P1_checkpoint_sha256':sha256_file(origin/'P1_PROBE.pth'),
                'transform_sha256':sha256_file(origin/'NORMALIZATION_AND_PROJECTION.npz'),
                'val_used_for_choice':False})
            val_samples=load_samples(origin/'SAMPLES_val.npz')
            select=val_samples['kind']<2
            matched_y=val_samples['label'][select]
            matched_k=val_samples['member'][select]
            matched={}
            for arm in ('P0','P1'):
                X=transform.apply(val_samples['B'][select],val_samples['H'][select],matched_k,arm)
                score=probe_scores(probes[arm],X,device,int(lock['score_batch']))
                matched[arm]=classification_metrics(matched_y,score)
            write_json(out/'MATCHED_val_STREAM.json',{
                'status':'COMPLETE','subset':'val','view':'balanced_matched_H0_controls',
                'arms':matched,'val_used_for_choice':False,'test_used':False})
    fields=list(dict.fromkeys(key for row in all_rows for key in row))
    with (out/'NATURAL_Q_PER_CASE_MEMBER_STREAM.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(all_rows)
    if (old_audit._module_state_digest(model),old_audit._module_state_digest(refiner))!=frozen:
        raise RuntimeError('C0 state changed')
    summary={'status':'COMPLETE','results':results,
             'C0_frozen_state_preserved':True,'C0_updates':0,
             'P0_P1_additional_updates':0,
             'elapsed_seconds':time.monotonic()-started,
             'test_used':False,'TSRS_used':False}
    write_json(out/'STREAM_RESULT.json',summary)
    print(json.dumps({'status':'COMPLETE','seconds':summary['elapsed_seconds'],
                      'holdout':results['probe_holdout']['arms'],
                      'val':results['val']['arms']}),flush=True)


if __name__=='__main__':main()
