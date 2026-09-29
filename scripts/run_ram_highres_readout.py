#!/usr/bin/env python3
"""Frozen C0/R332 high-resolution feature readout, P0 versus P1.

This experiment trains only two small probe MLPs.  C0 and its refiner are
never passed to an optimizer.  GT/H0 select and score examples only; all
readout inputs come from the image, frozen predictions/features, coordinates,
and the known member channel index.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

import audit_ram_source_correctability_response as old_audit
import run_v31_ram_mature as legacy
from ram_highres_capture import capture_case, load_frozen_c0
from train_r327_ram_native_instance_completion_prior import image_gradient


N_MEMBERS = 14
Q_SOURCE_CUTOFF = -5.00001
Q_FINAL_THRESHOLD = 0.5
H_DIM = 32
PROJECTION_DIM = 32
B_NAMES = (
    'image', 'image_gradient', 'step1_own', 'step1_other_max',
    'step1_other_sum_div2', 'step1_uncertainty',
    'source_logit', 'step1_logit', 'step2_logit',
    'source_probability', 'step2_probability',
    'source_other_max', 'source_other_sum_div2',
    'step2_other_max', 'step2_other_sum_div2',
    *(f'refiner_dec1_c{i}' for i in range(12)),
    *(f'source_p_member_{i}' for i in range(14)),
    *(f'step2_p_member_{i}' for i in range(14)),
    'y_fraction', 'x_fraction',
)
SAMPLE_TYPES = ('matched_h0', 'matched_negative', 'natural_q')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n',encoding='utf-8')


def ensure_disk_floor(path:Path, floor_bytes:int)->None:
    available=shutil.disk_usage(path).free
    if available<floor_bytes:
        raise RuntimeError(f'disk floor reached at {path}: {available} < {floor_bytes}')


def stable_rng(seed: int, case: str, member: int, label: str) -> np.random.Generator:
    digest = hashlib.sha256(f'{seed}|{case}|{member}|{label}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little'))


def sample_indices(pool: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    flat = np.asarray(pool, dtype=np.int64).reshape(-1)
    if len(flat) <= cap:
        return np.sort(flat)
    return np.sort(rng.choice(flat, size=cap, replace=False))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0/(1.0+np.exp(-np.clip(np.asarray(x,dtype=np.float32),-40,40)))


def make_q(source: np.ndarray, final: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Inference-available candidate Q, independent of GT/H0."""
    if source.shape != final.shape or source.shape[0] != N_MEMBERS:
        raise ValueError('source/final must be [14,H,W]')
    if valid.shape != source.shape[1:]:
        raise ValueError('valid shape mismatch')
    pred = final >= 0.0  # exact equivalent of FP32 sigmoid >= .5
    another = pred.sum(axis=0)[None] - pred.astype(np.int16) >= 1
    return valid[None] & (source < Q_SOURCE_CUTOFF) & (~pred) & another


def image_gradient_numpy(image: np.ndarray) -> np.ndarray:
    """Exact layout of R327 image_gradient, with the same 1e-8 stabilizer."""
    x = torch.from_numpy(np.asarray(image,dtype=np.float32))[None,None]
    with torch.inference_mode():
        return image_gradient(x)[0,0].numpy().astype(np.float32)


def feature_at(record: dict[str,np.ndarray], member: np.ndarray,
               y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Return strong mature B and one backbone H for aligned pixel/member rows."""
    k = np.asarray(member,dtype=np.int64)
    y = np.asarray(y,dtype=np.int64)
    x = np.asarray(x,dtype=np.int64)
    if not (len(k)==len(y)==len(x)):
        raise ValueError('coordinate lengths differ')
    if len(k)==0:
        return np.empty((0,len(B_NAMES)),dtype=np.float32),np.empty((0,H_DIM),dtype=np.float32)
    source, step1, step2 = (record[name] for name in ('source','step1','step2'))
    src_p, one_p, end_p = (sigmoid(value[:,y,x]).T for value in (source,step1,step2))
    n = len(k)
    rows = np.arange(n)
    def own_other(prob):
        own = prob[rows,k]
        without = prob.copy()
        without[rows,k] = -np.inf
        other_max = without.max(axis=1)
        other_sum = np.clip(prob.sum(axis=1)-own,0,2)/2
        return own,other_max,other_sum
    src_own,src_other_max,src_other_sum = own_other(src_p)
    one_own,one_other_max,one_other_sum = own_other(one_p)
    end_own,end_other_max,end_other_sum = own_other(end_p)
    high = np.asarray(record['backbone_h'],dtype=np.float32)[:,y,x].T
    refine = np.asarray(record['refiner_h'],dtype=np.float32)[k,:,y,x]
    if high.shape!=(n,H_DIM) or refine.shape!=(n,12):
        raise ValueError(f'feature shape mismatch: {high.shape}, {refine.shape}')
    hw = source.shape[-2:]
    scalars = np.column_stack((
        record['image'][y,x], record['gradient'][y,x],
        one_own,one_other_max,one_other_sum,4*one_own*(1-one_own),
        source[k,y,x], step1[k,y,x], step2[k,y,x],
        src_own,end_own,src_other_max,src_other_sum,
        end_other_max,end_other_sum,
    ))
    out = np.column_stack((scalars,refine,src_p,end_p,
                           y/float(hw[0]),x/float(hw[1]))).astype(np.float32)
    if out.shape!=(n,len(B_NAMES)) or not np.isfinite(out).all() or not np.isfinite(high).all():
        raise ValueError(f'B/H malformed: {out.shape}, {high.shape}')
    return out, high.astype(np.float32,copy=False)


def match_negatives(source: np.ndarray, final: np.ndarray, positives: np.ndarray,
                    negatives: np.ndarray, k: int, height: int, width: int,
                    rng: np.random.Generator, pool_cap: int) -> tuple[np.ndarray,np.ndarray]:
    """Match within one case/member using only C0-visible variables.

    The negative label is used solely to choose legitimate controls.  P1 H is
    not inspected.  Returned distances quantify when a close match was rare.
    """
    pos = np.asarray(positives,dtype=np.int64)
    neg = sample_indices(negatives,pool_cap,rng)
    if not len(pos) or not len(neg):
        return np.empty(0,dtype=np.int64),np.empty(0,dtype=np.float32)
    py,px = np.divmod(pos,width)
    ny,nx = np.divmod(neg,width)
    other = sigmoid(final).copy()
    other[k] = -np.inf
    other_max = other.max(axis=0)
    positive_src=source[k,py,px]
    negative_src=source[k,ny,nx]
    positive_other=other_max[py,px]
    negative_other=other_max[ny,nx]
    chosen=[];distances=[];available=np.ones(len(neg),dtype=bool)
    for j in range(len(pos)):
        # Same coarse 4x4 spatial cell is preferred; fallback remains explicit.
        cell = (ny//max(1,height//4)==py[j]//max(1,height//4)) & \
               (nx//max(1,width//4)==px[j]//max(1,width//4))
        candidates=np.flatnonzero(available&cell)
        if not len(candidates):
            candidates=np.flatnonzero(available)
        if not len(candidates):
            break
        distance=(np.abs(negative_src[candidates]-positive_src[j])/5.0
                  +np.abs(negative_other[candidates]-positive_other[j])/0.25
                  +np.abs(ny[candidates]-py[j])/height
                  +np.abs(nx[candidates]-px[j])/width)
        best=candidates[int(np.argmin(distance))]
        chosen.append(neg[best]);distances.append(float(np.min(distance)))
        available[best]=False
    return np.asarray(chosen,dtype=np.int64),np.asarray(distances,dtype=np.float32)


def build_case_samples(record: dict[str,np.ndarray], target: np.ndarray,
                       h0: np.ndarray, valid: np.ndarray, q: np.ndarray,
                       case: str, seed: int, matched_cap: int, natural_cap: int,
                       neg_pool_cap: int) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    height,width=valid.shape
    members=[];ys=[];xs=[];labels=[];h0_flags=[];kinds=[];inclusion=[]
    pair_distances=[];matched_requested=0;matched_realized=0
    natural_total=int(q.sum())
    for k in range(N_MEMBERS):
        flat_q=np.flatnonzero(q[k])
        flat_h=np.flatnonzero(q[k]&h0[k])
        pos=sample_indices(flat_h,matched_cap,stable_rng(seed,case,k,'h0'))
        matched_requested+=len(pos)
        flat_negative=np.flatnonzero(q[k]&~target[k])
        neg,dist=match_negatives(record['source'],record['step2'],pos,flat_negative,
                                 k,height,width,stable_rng(seed,case,k,'negpool'),neg_pool_cap)
        keep=min(len(pos),len(neg))
        pos=pos[:keep];neg=neg[:keep];dist=dist[:keep]
        matched_realized+=keep
        pair_distances.extend(dist.tolist())
        nat=sample_indices(flat_q,natural_cap,stable_rng(seed,case,k,'natural'))
        for flat,kind,pi in (
            (pos,'matched_h0',1.0),
            (neg,'matched_negative',1.0),
            (nat,'natural_q',len(nat)/max(len(flat_q),1)),
        ):
            if not len(flat):
                continue
            yy,xx=np.divmod(flat,width)
            members.extend([k]*len(flat));ys.extend(yy.tolist());xs.extend(xx.tolist())
            labels.extend(target[k,yy,xx].astype(np.int8).tolist())
            h0_flags.extend(h0[k,yy,xx].astype(np.int8).tolist())
            kinds.extend([SAMPLE_TYPES.index(kind)]*len(flat));inclusion.extend([pi]*len(flat))
    k=np.asarray(members,dtype=np.int16)
    yy=np.asarray(ys,dtype=np.int16)
    xx=np.asarray(xs,dtype=np.int16)
    B,H=feature_at(record,k,yy,xx)
    output={'B':B,'H':H,'member':k,'y':yy,'x':xx,
            'label':np.asarray(labels,dtype=np.int8),
            'h0':np.asarray(h0_flags,dtype=np.int8),
            'kind':np.asarray(kinds,dtype=np.int8),
            'inclusion':np.asarray(inclusion,dtype=np.float32),
            'case':np.asarray([case]*len(k),dtype='U64')}
    report={'case':case,'Q_members':natural_total,'H0_members':int(h0.sum()),
            'H0_in_Q':int((h0&q).sum()),'sampled_rows':len(k),
            'matched_requested':matched_requested,'matched_pairs':matched_realized,
            'match_distance_mean':float(np.mean(pair_distances)) if pair_distances else None,
            'natural_sample_rows':int((output['kind']==2).sum()),
            'natural_sample_positive':int(output['label'][output['kind']==2].sum())}
    return output,report


def load_case_and_capture(dataset: Any, index: int, model: torch.nn.Module,
                          refiner: torch.nn.Module, device: torch.device,
                          oracle_root: Path, split: str, parity: bool=False
                          ) -> tuple[dict[str,np.ndarray],np.ndarray,np.ndarray,np.ndarray,dict[str,Any]]:
    sample=dataset[index]
    case=str(sample['case'])
    image=sample['image'][None].to(device,dtype=torch.float32)
    before=None
    if parity:
        with torch.inference_mode():
            zero=torch.zeros_like(image)
            unhooked=model(torch.cat((image,zero),1))
            unhooked_states,_=legacy.iterative_refine_o(image,unhooked,refiner,2,zero)
            before=[value.detach().cpu().numpy()[0].astype(np.float32) for value in unhooked_states]
    captured=capture_case(model,refiner,image)
    record={name:np.asarray(captured[name],dtype=np.float32) for name in
            ('source','step1','step2','backbone_h','refiner_h')}
    if parity:
        maxdiff=[float(np.max(np.abs(left-record[name])))
                 for left,name in zip(before,('source','step1','step2'))]
        if max(maxdiff)>1e-6:
            raise RuntimeError(f'hook changed C0 forward for {case}: {maxdiff}')
    image_np=sample['image'].numpy()[0].astype(np.float32)
    record['image']=image_np
    with torch.inference_mode():
        record['gradient']=image_gradient(image)[0,0].detach().float().cpu().numpy().copy()
    target=np.asarray(sample['mask'].numpy()>0.5,dtype=bool)
    valid=np.asarray(sample['valid'].numpy()[0],dtype=bool)
    expected=(N_MEMBERS,*valid.shape)
    cache=old_audit.load_h0_cache(oracle_root/split/f'{case}.npz',expected)
    h0=np.asarray(cache['h0'],dtype=bool)
    membership=target.sum(axis=0)
    formula=(target & (membership[None]>=2) & valid[None] &
             (record['step2']<0.0) & (record['source']<Q_SOURCE_CUTOFF))
    mismatch=int(np.count_nonzero(h0^formula))
    if mismatch:
        raise RuntimeError(f'fixed H0 cache mismatch in {split}/{case}: {mismatch}')
    q=make_q(record['source'],record['step2'],valid)
    info={'case':case,'split':split,'input_shape':list(image.shape),
          'backbone_h_shape':list(record['backbone_h'].shape),
          'refiner_h_shape':list(record['refiner_h'].shape),
          'h0_members':int(h0.sum()),'q_members':int(q.sum()),
          'h0_in_q':int((h0&q).sum()),
          'hook_parity_maxdiff':maxdiff if parity else None,
          'h0_cache_sha256':cache['sha256']}
    return record,target,h0,q,info


def stack_samples(rows: list[dict[str,np.ndarray]]) -> dict[str,np.ndarray]:
    if not rows:
        raise ValueError('no sample rows')
    keys=set(rows[0])
    if any(set(row)!=keys for row in rows):
        raise ValueError('inconsistent sample row keys')
    return {key:np.concatenate([row[key] for row in rows],axis=0) for key in keys}


def profile_one(lock: dict[str,Any], output: Path) -> dict[str,Any]:
    started=time.monotonic()
    torch.set_num_threads(int(lock['cpu_threads']))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,refiner,_=load_frozen_c0(Path(lock['c0']),device)
    dataset=legacy.NativeWristDataset(Path(lock['dataset']),'train',False)
    case=lock['profile_case']
    index=next(i for i,path in enumerate(dataset.mask_files) if path.stem==case)
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    capture_start=time.monotonic()
    record,target,h0,q,info=load_case_and_capture(dataset,index,model,refiner,device,
                                                   Path(lock['oracle_h0_root']),'train',parity=True)
    capture_seconds=time.monotonic()-capture_start
    sampled,report=build_case_samples(record,target,h0,
                                       np.asarray(dataset[index]['valid'].numpy()[0],dtype=bool),
                                       q,case,int(lock['seed']),
                                       int(lock['matched_cap']),int(lock['natural_train_cap_per_member']),
                                       int(lock['negative_pool_cap']))
    flat=np.flatnonzero(q.reshape(-1))
    eval_take=min(int(lock['profile_natural_eval_points']),len(flat))
    selected=sample_indices(flat,eval_take,stable_rng(int(lock['seed']),case,0,'profile'))
    k,y,x=np.unravel_index(selected,q.shape)
    score_start=time.monotonic()
    B,H=feature_at(record,k,y,x)
    # Profile a real 32-wide MLP forward over the selected candidates; this
    # is a zero-update throughput measurement, not trained performance.
    synthetic=torch.cat((torch.from_numpy(B),torch.from_numpy(H)),dim=1)
    dummy=torch.nn.Sequential(torch.nn.Linear(synthetic.shape[1],32),
                              torch.nn.SiLU(),torch.nn.Linear(32,1)).eval()
    with torch.inference_mode():
        for batch in synthetic.split(8192):
            _=dummy(batch)
    score_seconds=time.monotonic()-score_start
    result={'status':'PROFILE_ONLY_ZERO_UPDATES','case':case,'device':str(device),
            'capture_seconds':capture_seconds,'score_points':int(eval_take),
            'feature_and_dummy_score_seconds':score_seconds,
            'total_seconds':time.monotonic()-started,
            'peak_gpu_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
            'sampling':report,'capture':info,
            'estimated_capture_494_seconds':capture_seconds*494,
            'estimated_natural_eval_238_seconds':(score_seconds/max(eval_take,1))*info['q_members']*238,
            'test_used':False,'optimizer_updates':0}
    write_json(output,result)
    print(json.dumps(result))
    return result


def extract_sample_cache(lock: dict[str,Any], out: Path, model: torch.nn.Module,
                         refiner: torch.nn.Module, device: torch.device,
                         deadline: float) -> dict[str,Any]:
    """One frozen C0 pass over train425/val69; cache only selected points."""
    splits=json.loads(Path(lock['split_manifest']).read_text(encoding='utf-8'))
    if splits.get('status')!='LOCKABLE_PATIENT_GROUPED':
        raise ValueError('patient-grouped split manifest is required')
    groups=splits['groups']
    memberships={case:name for name,cases in groups.items() for case in cases}
    memberships.update({case:'val' for case in splits['val_cases']})
    if len(memberships)!=494:
        raise ValueError(f'expected 494 train/val case IDs, got {len(memberships)}')
    datasets={name:legacy.NativeWristDataset(Path(lock['dataset']),name,False)
              for name in ('train','val')}
    if [len(datasets[s]) for s in ('train','val')]!=[425,69]:
        raise ValueError('remote RAM case counts differ')
    cache={name:[] for name in ('probe_fit','probe_calibration','probe_holdout','val')}
    case_rows=[]
    started=time.monotonic()
    with (out/'CASE_EXTRACTION_TRACE.jsonl').open('w',encoding='utf-8') as trace:
        for source_split,dataset in datasets.items():
            for i,path in enumerate(dataset.mask_files):
                if time.monotonic()>deadline:
                    raise TimeoutError('locked budget exceeded during frozen extraction')
                case=path.stem
                subset=memberships.get(case)
                if subset is None or (source_split=='val')!=(subset=='val'):
                    raise ValueError(f'split mapping mismatch: {source_split}/{case}/{subset}')
                case_started=time.monotonic()
                record,target,h0,q,info=load_case_and_capture(
                    dataset,i,model,refiner,device,Path(lock['oracle_h0_root']),
                    source_split,parity=(source_split=='train' and i==0))
                valid=np.zeros(q.shape[-2:],dtype=bool)
                original_hw=old_audit.load_h0_cache(
                    Path(lock['oracle_h0_root'])/source_split/f'{case}.npz',q.shape)['original_hw']
                if original_hw is None:
                    raise ValueError(f'H0 cache has no original_hw: {case}')
                valid[:original_hw[0],:original_hw[1]]=True
                samples,report=build_case_samples(
                    record,target,h0,valid,q,case,int(lock['seed']),
                    int(lock['matched_cap']),int(lock['natural_train_cap_per_member']),
                    int(lock['negative_pool_cap']))
                cache[subset].append(samples)
                row={**info,**report,'probe_subset':subset,
                     'capture_and_sample_seconds':time.monotonic()-case_started}
                case_rows.append(row)
                trace.write(json.dumps(row,allow_nan=False)+'\n')
                trace.flush()
                if (i+1)%25==0:
                    ensure_disk_floor(out,int(lock['disk_free_floor_bytes']))
                    print(json.dumps({'phase':'extract','split':source_split,
                                      'cases':i+1,'elapsed':time.monotonic()-started}),flush=True)
    summaries={}
    for name,rows in cache.items():
        ensure_disk_floor(out,int(lock['disk_free_floor_bytes']))
        data=stack_samples(rows)
        path=out/f'SAMPLES_{name}.npz'
        np.savez_compressed(path,**data)
        summaries[name]={'cases':len(rows),'sample_rows':len(data['label']),
                         'matched_h0_rows':int((data['kind']==0).sum()),
                         'matched_negative_rows':int((data['kind']==1).sum()),
                         'natural_rows':int((data['kind']==2).sum()),
                         'natural_positive_rows':int(((data['kind']==2)&(data['label']==1)).sum()),
                         'cache_bytes':path.stat().st_size,
                         'cache_sha256':sha256_file(path)}
    total_cache=sum(row['cache_bytes'] for row in summaries.values())
    if total_cache>int(lock['cache_bytes_limit']):
        raise RuntimeError(f'feature cache exceeds locked limit: {total_cache}')
    result={'status':'COMPLETE','case_rows':len(case_rows),
            'train_H0_members':sum(r['h0_members'] for r in case_rows if r['split']=='train'),
            'val_H0_members':sum(r['h0_members'] for r in case_rows if r['split']=='val'),
            'train_H0_in_Q':sum(r['h0_in_q'] for r in case_rows if r['split']=='train'),
            'val_H0_in_Q':sum(r['h0_in_q'] for r in case_rows if r['split']=='val'),
            'subsets':summaries,'total_cache_bytes':total_cache,
            'elapsed_seconds':time.monotonic()-started,
            'test_used':False,'c0_updates':0}
    if result['val_H0_members']!=71335:
        raise ValueError(f'val H0 count differs from lock: {result["val_H0_members"]}')
    write_json(out/'SAMPLE_EXTRACTION_RESULT.json',result)
    return result


class Probe(torch.nn.Module):
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.net=torch.nn.Sequential(torch.nn.Linear(input_dim,hidden),
                                     torch.nn.SiLU(),torch.nn.Linear(hidden,1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).squeeze(-1)


class FeatureTransform:
    """Fit-only normalization and equal-width fixed random projections."""
    def __init__(self, arrays: dict[str,np.ndarray], seed: int):
        B=np.asarray(arrays['B'],dtype=np.float64)
        H=np.asarray(arrays['H'],dtype=np.float64)
        self.b_mean=B.mean(axis=0).astype(np.float32)
        self.b_std=B.std(axis=0).astype(np.float32)
        self.h_mean=H.mean(axis=0).astype(np.float32)
        self.h_std=H.std(axis=0).astype(np.float32)
        self.b_std[self.b_std<1e-6]=1.0
        self.h_std[self.h_std<1e-6]=1.0
        rng=np.random.default_rng(seed+101)
        self.a_b=(rng.standard_normal((B.shape[1],PROJECTION_DIM))/math.sqrt(B.shape[1])).astype(np.float32)
        self.a_h=(rng.standard_normal((H.shape[1],PROJECTION_DIM))/math.sqrt(H.shape[1])).astype(np.float32)
        self.input_dim=B.shape[1]+PROJECTION_DIM+N_MEMBERS
        if B.shape[1]!=len(B_NAMES) or H.shape[1]!=H_DIM:
            raise ValueError('fit feature dimensions differ from locked names')

    def apply(self, B: np.ndarray, H: np.ndarray, member: np.ndarray,
              arm: str) -> np.ndarray:
        b=(np.asarray(B,dtype=np.float32)-self.b_mean)/self.b_std
        h=(np.asarray(H,dtype=np.float32)-self.h_mean)/self.h_std
        extra=b@self.a_b if arm=='P0' else h@self.a_h
        if arm not in ('P0','P1'):
            raise ValueError(arm)
        member=np.asarray(member,dtype=np.int64)
        onehot=np.eye(N_MEMBERS,dtype=np.float32)[member]
        value=np.column_stack((b,extra,onehot)).astype(np.float32)
        if value.shape[1]!=self.input_dim or not np.isfinite(value).all():
            raise ValueError('transformed probe input malformed')
        return value

    def save(self,path:Path)->None:
        np.savez_compressed(path,b_mean=self.b_mean,b_std=self.b_std,
                            h_mean=self.h_mean,h_std=self.h_std,
                            a_b=self.a_b,a_h=self.a_h)


def load_samples(path: Path) -> dict[str,np.ndarray]:
    with np.load(path,allow_pickle=False) as archive:
        return {key:archive[key] for key in archive.files}


def probe_scores(model: Probe, value: np.ndarray, device: torch.device,
                 batch_size: int=8192) -> np.ndarray:
    model.eval()
    values=[]
    with torch.inference_mode():
        for start in range(0,len(value),batch_size):
            x=torch.from_numpy(np.asarray(value[start:start+batch_size],dtype=np.float32)).to(device)
            values.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(values) if values else np.empty(0,dtype=np.float32)


def classification_metrics(y: np.ndarray, score: np.ndarray) -> dict[str,Any]:
    y=np.asarray(y,dtype=np.int8)
    p=np.asarray(score,dtype=np.float64)
    result={'n':len(y),'positive':int(y.sum()),'negative':int(len(y)-y.sum())}
    if len(y) and np.unique(y).size==2:
        result.update({'auc':float(roc_auc_score(y,p)),
                       'ap':float(average_precision_score(y,p)),
                       'logloss':float(log_loss(y,np.clip(p,1e-6,1-1e-6)))})
    else:
        result.update({'auc':None,'ap':None,'logloss':None})
    return result


def fit_probes(lock:dict[str,Any],out:Path,device:torch.device
               )->tuple[FeatureTransform,dict[str,Probe],dict[str,Any]]:
    fit=load_samples(out/'SAMPLES_probe_fit.npz')
    n=len(fit['label'])
    if n<1000 or not set(np.unique(fit['label']))=={0,1}:
        raise ValueError('fit library is too small or one-class')
    transform=FeatureTransform(fit,int(lock['seed']))
    transform_path=out/'NORMALIZATION_AND_PROJECTION.npz'
    transform.save(transform_path)
    x={arm:transform.apply(fit['B'],fit['H'],fit['member'],arm) for arm in ('P0','P1')}
    y=torch.from_numpy(fit['label'].astype(np.float32)).to(device)
    torch.manual_seed(int(lock['seed'])+211)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(lock['seed'])+211)
    initial=copy.deepcopy(Probe(transform.input_dim,int(lock['hidden_width'])).state_dict())
    epochs=int(lock['epochs'])
    batch_size=int(lock['batch_size'])
    rng=np.random.default_rng(int(lock['seed'])+313)
    schedules=[rng.permutation(n) for _ in range(epochs)]
    expected_steps=sum(math.ceil(n/batch_size) for _ in range(epochs))
    models={}
    reports={}
    with (out/'PROBE_TRAINING_TRACE.jsonl').open('w',encoding='utf-8') as trace:
        for arm in ('P0','P1'):
            probe=Probe(transform.input_dim,int(lock['hidden_width'])).to(device)
            probe.load_state_dict(copy.deepcopy(initial),strict=True)
            optimizer=torch.optim.AdamW(probe.parameters(),
                lr=float(lock['learning_rate']),weight_decay=float(lock['weight_decay']))
            features=torch.from_numpy(x[arm]).to(device)
            step=0
            arm_start=time.monotonic()
            for epoch,order in enumerate(schedules,1):
                probe.train()
                for start in range(0,n,batch_size):
                    if step>=int(lock['max_optimizer_steps_per_arm']):
                        raise RuntimeError('prelocked P0/P1 step cap exceeded')
                    idx=order[start:start+batch_size]
                    device_idx=torch.as_tensor(idx,device=device,dtype=torch.long)
                    xt=features[device_idx]
                    yt=y[device_idx]
                    before=[p.detach().clone() for p in probe.parameters()]
                    optimizer.zero_grad(set_to_none=True)
                    logits=probe(xt)
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,yt)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f'{arm} nonfinite loss at {step+1}')
                    loss.backward()
                    grad_norm=math.sqrt(sum(float(p.grad.detach().square().sum()) for p in probe.parameters()))
                    optimizer.step()
                    delta_norm=math.sqrt(sum(float((p.detach()-b).square().sum())
                                             for p,b in zip(probe.parameters(),before)))
                    step+=1
                    row={'arm':arm,'step':step,'epoch':epoch,'sample_rows':len(idx),
                         'positive':int(fit['label'][idx].sum()),
                         'H0':int(fit['h0'][idx].sum()),
                         'matched_h0':int((fit['kind'][idx]==0).sum()),
                         'natural_q':int((fit['kind'][idx]==2).sum()),
                         'loss':float(loss.detach().cpu()),
                         'grad_l2':grad_norm,'delta_theta_l2':delta_norm,
                         'lr':float(lock['learning_rate']),'success':True}
                    trace.write(json.dumps(row)+'\n')
                    if step==1 or step%50==0 or step==expected_steps:
                        print(json.dumps({'phase':'fit',**row}),flush=True)
            if step!=expected_steps:
                raise RuntimeError(f'{arm} step mismatch: {step} != {expected_steps}')
            scores=probe_scores(probe,x[arm],device)
            reports[arm]={'steps':step,'seconds':time.monotonic()-arm_start,
                          'train':classification_metrics(fit['label'],scores),
                          'checkpoint':str(out/f'{arm}_PROBE.pth')}
            torch.save({'model':probe.cpu().state_dict(),'input_dim':transform.input_dim,
                        'hidden_width':int(lock['hidden_width']),'arm':arm,
                        'steps':step,'C0_sha256':lock['C0_sha256'],
                        'lock_sha256':sha256_file(Path(lock['_path']))},out/f'{arm}_PROBE.pth')
            probe.to(device).eval()
            models[arm]=probe
    if reports['P0']['steps']!=reports['P1']['steps']:
        raise RuntimeError('unequal P0/P1 optimizer steps')
    write_json(out/'PROBE_FIT_RESULT.json',
               {'status':'COMPLETE','fit_rows':n,'input_dim':transform.input_dim,
                'trainable_parameters':sum(p.numel() for p in models['P0'].parameters()),
                'transform_sha256':sha256_file(transform_path),
                'arms':reports,'same_sample_schedule':True,
                'same_initial_state':True,'C0_updated':False})
    return transform,models,reports


def matched_view(lock:dict[str,Any],out:Path,transform:FeatureTransform,
                 probes:dict[str,Probe],device:torch.device,subset:str,
                 suffix:str='') -> dict[str,Any]:
    samples=load_samples(out/f'SAMPLES_{subset}.npz')
    selected=samples['kind']<2
    y=samples['label'][selected]
    if len(y)==0:
        raise ValueError(f'no matched sample rows in {subset}')
    cases=samples['case'][selected].astype(str)
    members=samples['member'][selected]
    result={'subset':subset,'n':len(y),'positive':int(y.sum()),
            'negative':int(len(y)-y.sum()),'arms':{}}
    rows=[]
    for arm in ('P0','P1'):
        X=transform.apply(samples['B'][selected],samples['H'][selected],members,arm)
        score=probe_scores(probes[arm],X,device)
        result['arms'][arm]=classification_metrics(y,score)
        for case in np.unique(cases):
            mask=cases==case
            row={'subset':subset,'case':case,'member':-1,'arm':arm,
                 **classification_metrics(y[mask],score[mask])}
            rows.append(row)
        for k in range(N_MEMBERS):
            mask=members==k
            rows.append({'subset':subset,'case':'ALL','member':k,'arm':arm,
                         **classification_metrics(y[mask],score[mask])})
    with (out/f'MATCHED_PER_CASE_MEMBER_{subset}{suffix}.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    write_json(out/f'MATCHED_{subset}{suffix}.json',result)
    return result


def weighted_metrics(y:np.ndarray,score:np.ndarray,weight:np.ndarray)->dict[str,Any]:
    if len(y)==0 or np.unique(y).size<2:
        return {'auc':None,'ap':None,'positive_weight':float(weight[y==1].sum()),
                'negative_weight':float(weight[y==0].sum())}
    return {'auc':float(roc_auc_score(y,score,sample_weight=weight)),
            'ap':float(average_precision_score(y,score,sample_weight=weight)),
            'positive_weight':float(weight[y==1].sum()),
            'negative_weight':float(weight[y==0].sum())}


def threshold_for_fpr(scores:np.ndarray,weights:np.ndarray,alpha:float)->float:
    if not len(scores):
        raise ValueError('no calibration negatives')
    order=np.argsort(scores,kind='stable')[::-1]
    sorted_score=scores[order]
    cumulative=np.cumsum(weights[order],dtype=np.float64)
    allowed=float(alpha)*float(weights.sum())
    first_exceeds=int(np.searchsorted(cumulative,allowed,side='right'))
    if first_exceeds>=len(sorted_score):
        return 0.0
    # Excluding all ties at this score conservatively guarantees the locked
    # empirical sampled FPR is no greater than alpha.
    return float(np.nextafter(np.float32(sorted_score[first_exceeds]),np.float32(np.inf)))


def evaluate_natural(lock:dict[str,Any],out:Path,model:torch.nn.Module,
                     refiner:torch.nn.Module,device:torch.device,
                     transform:FeatureTransform,probes:dict[str,Probe],
                     subset:str,thresholds:dict[str,float]|None,
                     deadline:float)->tuple[dict[str,Any],dict[str,float]|None]:
    splits=json.loads(Path(lock['split_manifest']).read_text(encoding='utf-8'))
    cases=splits['val_cases'] if subset=='val' else splits['groups'][subset]
    source_split='val' if subset=='val' else 'train'
    dataset=legacy.NativeWristDataset(Path(lock['dataset']),source_split,False)
    indices={path.stem:i for i,path in enumerate(dataset.mask_files)}
    if not set(cases).issubset(indices):
        raise ValueError(f'{subset} contains case missing in {source_split}')
    score_parts={arm:[] for arm in ('P0','P1')}
    label_parts=[];h0_parts=[];member_parts=[];weight_parts=[];case_parts=[]
    case_rows=[]
    population={'q_members':0,'q_positive':0,'q_negative':0,
                'h0_members':0,'h0_in_q':0,'sampled_q':0,
                'member_q_positive':np.zeros(N_MEMBERS,dtype=np.int64),
                'member_q_negative':np.zeros(N_MEMBERS,dtype=np.int64)}
    started=time.monotonic()
    for i,case in enumerate(cases):
        if time.monotonic()>deadline:
            raise TimeoutError('locked budget exceeded during natural Q evaluation')
        if (i+1)%20==0:
            ensure_disk_floor(out,int(lock['disk_free_floor_bytes']))
        record,target,h0,q,info=load_case_and_capture(
            dataset,indices[case],model,refiner,device,Path(lock['oracle_h0_root']),
            source_split,parity=False)
        full_n=int(q.sum())
        population['q_members']+=full_n
        full_pos=int((q&target).sum())
        population['q_positive']+=full_pos
        population['q_negative']+=full_n-full_pos
        population['h0_members']+=int(h0.sum())
        population['h0_in_q']+=int((h0&q).sum())
        for k in range(N_MEMBERS):
            positive=int((q[k]&target[k]).sum())
            population['member_q_positive'][k]+=positive
            population['member_q_negative'][k]+=int(q[k].sum())-positive
        pool=np.flatnonzero(q.reshape(-1))
        cap=int(lock['natural_eval_cap_per_case'])
        take=len(pool) if cap<=0 else min(cap,len(pool))
        selected=sample_indices(pool,take,stable_rng(int(lock['seed']),case,-1,'natural_eval'))
        k,y,x=np.unravel_index(selected,q.shape)
        inclusion=len(selected)/max(full_n,1)
        weights=np.full(len(selected),1.0/max(inclusion,1e-12),dtype=np.float32)
        labels=target[k,y,x].astype(np.int8)
        h0_selected=h0[k,y,x].astype(np.int8)
        arm_scores={arm:[] for arm in ('P0','P1')}
        chunk=int(lock['score_chunk'])
        for start in range(0,len(selected),chunk):
            stop=start+chunk
            B,H=feature_at(record,k[start:stop],y[start:stop],x[start:stop])
            for arm in ('P0','P1'):
                X=transform.apply(B,H,k[start:stop],arm)
                arm_scores[arm].append(probe_scores(probes[arm],X,device,
                                                    batch_size=int(lock['score_batch'])))
        for arm in ('P0','P1'):
            value=np.concatenate(arm_scores[arm]) if arm_scores[arm] else np.empty(0,dtype=np.float32)
            score_parts[arm].append(value)
        label_parts.append(labels);h0_parts.append(h0_selected)
        member_parts.append(k.astype(np.int8));weight_parts.append(weights)
        case_parts.append(np.full(len(selected),i,dtype=np.int16))
        population['sampled_q']+=len(selected)
        case_rows.append({'subset':subset,'case':case,'q_exact':full_n,
                          'q_positive_exact':full_pos,'q_negative_exact':full_n-full_pos,
                          'h0_exact':int(h0.sum()),'h0_in_q_exact':int((h0&q).sum()),
                          'sampled_q':len(selected),'sampled_positive':int(labels.sum()),
                          'inclusion_probability':inclusion})
        if (i+1)%20==0:
            print(json.dumps({'phase':'natural','subset':subset,'cases':i+1,
                              'elapsed':time.monotonic()-started}),flush=True)
    y=np.concatenate(label_parts)
    h0_flags=np.concatenate(h0_parts)
    members=np.concatenate(member_parts)
    weights=np.concatenate(weight_parts)
    case_index=np.concatenate(case_parts)
    scores={arm:np.concatenate(score_parts[arm]) for arm in ('P0','P1')}
    if len(y)!=population['sampled_q']:
        raise RuntimeError('sample count mismatch')
    if thresholds is None:
        if subset!='probe_calibration':
            raise ValueError('thresholds may be learned only on calibration')
        thresholds={arm:threshold_for_fpr(scores[arm][y==0],weights[y==0],
                                          float(lock['low_fpr_alpha']))
                    for arm in ('P0','P1')}
    result={'status':'COMPLETE','subset':subset,'approximate':int(lock['natural_eval_cap_per_case'])>0,
            'sampling':'uniform within each case Q independent of GT; weighted by inverse inclusion',
            'population_exact':{key:(value.tolist() if isinstance(value,np.ndarray) else value)
                                for key,value in population.items()},
            'low_fpr_alpha':float(lock['low_fpr_alpha']),
            'thresholds_from_calibration':thresholds,'arms':{},
            'elapsed_seconds':time.monotonic()-started,
            'test_used':False,'C0_updates':0}
    detail_rows=[]
    for arm in ('P0','P1'):
        score=scores[arm]
        prediction=score>=thresholds[arm]
        tp=float(weights[(prediction)&(y==1)].sum(dtype=np.float64))
        fp=float(weights[(prediction)&(y==0)].sum(dtype=np.float64))
        pos=float(weights[y==1].sum(dtype=np.float64))
        neg=float(weights[y==0].sum(dtype=np.float64))
        h0_total=float(weights[h0_flags==1].sum(dtype=np.float64))
        h0_tp=float(weights[(prediction)&(h0_flags==1)].sum(dtype=np.float64))
        result['arms'][arm]={**weighted_metrics(y,score,weights),
            'weighted_tp_estimate':tp,'weighted_fp_estimate':fp,
            'weighted_positive_estimate':pos,'weighted_negative_estimate':neg,
            'sample_fpr_weighted':fp/max(neg,1),
            'sample_precision_weighted':tp/max(tp+fp,1),
            'sample_recall_weighted':tp/max(pos,1),
            'H0_in_Q_recall_weighted':h0_tp/max(h0_total,1),
            'H0_full_recall_estimate':(h0_tp/max(h0_total,1))*
                                      (population['h0_in_q']/max(population['h0_members'],1)),
            'H0_score_sample_count':int((h0_flags==1).sum()),
            'positive_score_sample_count':int((y==1).sum())}
        for index,base_row in enumerate(case_rows):
            mask=case_index==index
            ym=y[mask];sc=score[mask];w=weights[mask];hm=h0_flags[mask]
            pred=sc>=thresholds[arm]
            t=float(w[pred&(ym==1)].sum(dtype=np.float64))
            f=float(w[pred&(ym==0)].sum(dtype=np.float64))
            hp=float(w[pred&(hm==1)].sum(dtype=np.float64))
            row={**base_row,'arm':arm,
                 'weighted_tp_estimate':t,'weighted_fp_estimate':f,
                 'weighted_h0_recovered_estimate':hp,
                 'weighted_fpr':f/max(float(w[ym==0].sum()),1),
                 'weighted_precision':t/max(t+f,1),
                 **weighted_metrics(ym,sc,w)}
            detail_rows.append(row)
        for k in range(N_MEMBERS):
            mask=members==k
            ym=y[mask];sc=score[mask];w=weights[mask];hm=h0_flags[mask]
            pred=sc>=thresholds[arm]
            t=float(w[pred&(ym==1)].sum(dtype=np.float64))
            f=float(w[pred&(ym==0)].sum(dtype=np.float64))
            hp=float(w[pred&(hm==1)].sum(dtype=np.float64))
            detail_rows.append({'subset':subset,'case':'ALL','member':k,'arm':arm,
                 'q_exact':int(population['member_q_positive'][k]+population['member_q_negative'][k]),
                 'q_positive_exact':int(population['member_q_positive'][k]),
                 'q_negative_exact':int(population['member_q_negative'][k]),
                 'weighted_tp_estimate':t,'weighted_fp_estimate':f,
                 'weighted_h0_recovered_estimate':hp,
                 'weighted_fpr':f/max(float(w[ym==0].sum()),1),
                 'weighted_precision':t/max(t+f,1),
                 **weighted_metrics(ym,sc,w)})
    write_json(out/f'NATURAL_Q_{subset}.json',result)
    # Per-case and per-member rows have different optional fields; CSV uses
    # their union so no unfavorable subset is silently dropped.
    fields=list(dict.fromkeys(key for row in detail_rows for key in row))
    with (out/f'NATURAL_Q_{subset}_PER_CASE_MEMBER.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(detail_rows)
    return result,thresholds


def verify_lock(path:Path,lock:dict[str,Any])->dict[str,Any]:
    if lock.get('status')!='LOCKED_BEFORE_PROBE_TRAINING':
        raise ValueError('readout run lock is not executable')
    if lock.get('test_used') or lock.get('TSRS_used') or lock.get('C0_updates')!=0:
        raise ValueError('forbidden split or C0 update in lock')
    if lock.get('C0_sha256')!='ae4c4eb21f6e9424e30c731b2c388d24233d389198a6371b1bd3090d3d788384':
        raise ValueError('wrong C0 identity')
    if lock.get('P0_hidden_hook')!='refiner.dec1' or lock.get('P1_hook')!='model.backbone.decoder.stages[4]':
        raise ValueError('hook point was changed')
    checked={}
    for filename,expected in lock['file_hashes'].items():
        current=sha256_file(Path(filename))
        if current!=expected:
            raise ValueError(f'locked file SHA differs: {filename}: {current} != {expected}')
        checked[filename]=current
    if sha256_file(Path(lock['c0']))!=lock['C0_sha256']:
        raise ValueError('C0 file SHA differs')
    return {'status':'PASS','lock_sha256':sha256_file(path),'files_checked':len(checked)}


def run_locked(lock_path:Path,out:Path,phase:str)->None:
    if phase=='extract' and out.exists():
        raise FileExistsError(f'refusing to overwrite {out}')
    if phase=='fit_eval' and not (out/'SAMPLE_EXTRACTION_RESULT.json').is_file():
        raise FileNotFoundError('fit_eval requires completed frozen sample extraction')
    lock=json.loads(lock_path.read_text(encoding='utf-8'))
    verification=verify_lock(lock_path,lock)
    lock['_path']=str(lock_path.resolve())
    started=time.monotonic()
    deadline=started+float(lock['max_seconds'])
    torch.set_num_threads(int(lock['cpu_threads']))
    random.seed(int(lock['seed']))
    np.random.seed(int(lock['seed']))
    torch.manual_seed(int(lock['seed']))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(lock['seed']))
        torch.cuda.reset_peak_memory_stats()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda':
        raise RuntimeError('locked probe requires CUDA server')
    if phase=='extract':
        out.mkdir(parents=True)
        write_json(out/'RUN_MANIFEST.json',{'status':'EXTRACTING_NO_PROBE_UPDATES',
                                          'lock':str(lock_path),
                                          'lock_verification':verification,
                                          'started_unix':time.time(),
                                          'test_used':False,'C0_updates':0})
    model,refiner,c0_info=load_frozen_c0(Path(lock['c0']),device)
    original_state=(old_audit._module_state_digest(model),old_audit._module_state_digest(refiner))
    if phase=='extract':
        extraction=extract_sample_cache(lock,out,model,refiner,device,deadline)
        after_extract=(old_audit._module_state_digest(model),old_audit._module_state_digest(refiner))
        if after_extract!=original_state:
            raise RuntimeError('C0 state changed during feature extraction')
        write_json(out/'RUN_MANIFEST.json',{
            'status':'EXTRACTED_NO_PROBE_UPDATES','lock_sha256':verification['lock_sha256'],
            'C0':c0_info,'C0_state_digest_before':list(original_state),
            'C0_state_digest_after':list(after_extract),'C0_frozen_state_preserved':True,
            'C0_updates':0,'P0_P1_updates':0,
            'extraction':extraction,'elapsed_seconds':time.monotonic()-started,
            'peak_cuda_bytes':int(torch.cuda.max_memory_allocated()),
            'test_used':False})
        print(json.dumps({'status':'EXTRACTED_NO_PROBE_UPDATES',
                          'seconds':time.monotonic()-started,
                          'subsets':extraction['subsets']}),flush=True)
        return
    extraction=json.loads((out/'SAMPLE_EXTRACTION_RESULT.json').read_text(encoding='utf-8'))
    for subset,expected in lock['sample_cache_sha256'].items():
        actual=sha256_file(out/f'SAMPLES_{subset}.npz')
        if actual!=expected or actual!=extraction['subsets'][subset]['cache_sha256']:
            raise ValueError(f'cached {subset} features changed before probe training')
    expected_steps=math.ceil(extraction['subsets']['probe_fit']['sample_rows']/int(lock['batch_size']))*int(lock['epochs'])
    if expected_steps!=int(lock['exact_optimizer_steps_per_arm']):
        raise ValueError(f'prelocked optimizer steps {lock["exact_optimizer_steps_per_arm"]} != actual {expected_steps}')
    write_json(out/'RUN_MANIFEST.json',{'status':'PROBE_RUNNING','lock_sha256':verification['lock_sha256'],
                                      'C0':c0_info,'C0_state_digest_before':list(original_state),
                                      'extraction':extraction,'planned_steps_per_arm':expected_steps,
                                      'test_used':False,'C0_updates':0})
    transform,probes,fit_report=fit_probes(lock,out,device)
    fit_matched=matched_view(lock,out,transform,probes,device,'probe_fit')
    cal_matched=matched_view(lock,out,transform,probes,device,'probe_calibration')
    hold_matched=matched_view(lock,out,transform,probes,device,'probe_holdout')
    cal_natural,thresholds=evaluate_natural(lock,out,model,refiner,device,transform,
                    probes,'probe_calibration',None,deadline)
    hold_natural,_=evaluate_natural(lock,out,model,refiner,device,transform,
                    probes,'probe_holdout',thresholds,deadline)
    write_json(out/'PROBE_RULES_LOCKED_BEFORE_VAL.json',{
        'status':'LOCKED_AFTER_TRAIN_HOLDOUT_BEFORE_VAL',
        'P0_checkpoint_sha256':sha256_file(out/'P0_PROBE.pth'),
        'P1_checkpoint_sha256':sha256_file(out/'P1_PROBE.pth'),
        'transform_sha256':sha256_file(out/'NORMALIZATION_AND_PROJECTION.npz'),
        'thresholds':thresholds,'alpha':float(lock['low_fpr_alpha']),
        'candidate':'Q from image geometry and frozen C0 predictions only',
        'H':'model.backbone.decoder.stages[4] output',
        'val_used_for_choice':False})
    # One development confirmation, after both probe models and both low-FPR
    # thresholds have been saved.  No parameter or rule is changed afterward.
    val_matched=matched_view(lock,out,transform,probes,device,'val')
    val_natural,_=evaluate_natural(lock,out,model,refiner,device,transform,
                    probes,'val',thresholds,deadline)
    final_state=(old_audit._module_state_digest(model),old_audit._module_state_digest(refiner))
    if final_state!=original_state:
        raise RuntimeError('C0 model/refiner state changed during readout')
    if any(parameter.grad is not None for module in (model,refiner)
           for parameter in module.parameters()):
        raise RuntimeError('C0 gradient unexpectedly created')
    results=[]
    for subset,matched,natural in (
        ('probe_calibration',cal_matched,cal_natural),
        ('probe_holdout',hold_matched,hold_natural),
        ('val',val_matched,val_natural),
    ):
        for arm in ('P0','P1'):
            m=matched['arms'][arm];n=natural['arms'][arm]
            results.append({'subset':subset,'arm':arm,'view':'matched',
                            'auc':m['auc'],'ap':m['ap'],'logloss':m['logloss'],
                            'positive':m['positive'],'negative':m['negative'],
                            'threshold':None,'fpr':None,'fp_estimate':None,
                            'h0_coverage':None})
            results.append({'subset':subset,'arm':arm,'view':'natural_Q',
                            'auc':n['auc'],'ap':n['ap'],'logloss':None,
                            'positive':natural['population_exact']['q_positive'],
                            'negative':natural['population_exact']['q_negative'],
                            'threshold':thresholds[arm],
                            'fpr':n['sample_fpr_weighted'],
                            'fp_estimate':n['weighted_fp_estimate'],
                            'h0_coverage':natural['population_exact']['h0_in_q']/
                                           max(natural['population_exact']['h0_members'],1),
                            'h0_in_q_recall':n['H0_in_Q_recall_weighted'],
                            'h0_full_recall_estimate':n['H0_full_recall_estimate']})
    fields=list(dict.fromkeys(key for row in results for key in row))
    with (out/'PROBE_RESULTS.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(results)
    elapsed=time.monotonic()-started
    manifest={'status':'COMPLETE','lock_sha256':verification['lock_sha256'],
              'C0':c0_info,'C0_state_digest_before':list(original_state),
              'C0_state_digest_after':list(final_state),
              'C0_frozen_state_preserved':True,
              'C0_gradients_present':False,
              'C0_optimizer_created':False,'C0_updates':0,
              'P0_P1_optimizer_steps':{arm:fit_report[arm]['steps'] for arm in ('P0','P1')},
              'P0_P1_same_training_schedule':True,
              'extraction':extraction,'matched_fit':fit_matched,
              'holdout_matched':hold_matched,
              'holdout_natural':hold_natural,
              'val_matched':val_matched,'val_natural':val_natural,
              'elapsed_seconds':elapsed,
              'peak_cuda_bytes':int(torch.cuda.max_memory_allocated()),
              'cache_bytes':extraction['total_cache_bytes'],
              'test_used':False,'TSRS_used':False,
              'full_segmentation_training':'NOT_RUN',
              'new_segmentation_metrics':'NOT_RUN'}
    write_json(out/'RUN_MANIFEST.json',manifest)
    print(json.dumps({'status':'COMPLETE','seconds':elapsed,
                      'fit_steps':manifest['P0_P1_optimizer_steps'],
                      'holdout_Q':{arm:hold_natural['arms'][arm] for arm in ('P0','P1')},
                      'val_Q':{arm:val_natural['arms'][arm] for arm in ('P0','P1')}},
                     allow_nan=False),flush=True)


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('profile','extract','fit_eval'),required=True)
    p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    lock_path=a.lock.resolve()
    lock=json.loads(lock_path.read_text(encoding='utf-8'))
    if a.mode=='profile':
        if a.out.exists(): raise FileExistsError(a.out)
        profile_one(lock,a.out)
    else:
        run_locked(lock_path,a.out.resolve(),a.mode)


if __name__=='__main__':
    main()
