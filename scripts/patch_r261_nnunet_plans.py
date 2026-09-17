#!/usr/bin/env python3
"""Keep X-ray z-score normalization but preserve prior amplitudes in [0,255]."""
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('--plans',type=Path,required=True); a=p.parse_args(); d=json.loads(a.plans.read_text())
 for name,cfg in d['configurations'].items():
  schemes=cfg.get('normalization_schemes',[])
  if len(schemes)!=13: raise RuntimeError((name,schemes))
  cfg['normalization_schemes']=['ZScoreNormalization']+['NoNormalization']*12; cfg['use_mask_for_norm']=[False]*13
 a.plans.write_text(json.dumps(d,indent=2)); print(json.dumps({'plans':str(a.plans),'xray':'ZScoreNormalization','prior_channels':12,'prior_normalization':'NoNormalization'}))
if __name__=='__main__': main()

