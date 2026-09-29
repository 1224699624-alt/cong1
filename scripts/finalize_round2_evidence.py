"""Bind Round2 local and remote evidence without claiming a training result."""
import hashlib,json,sys,unittest
from pathlib import Path
from test_local_supervision_v3 import LocalSupervisionTests

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/v3_round2'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def main():
    remote=read('CURRENT_REMOTE_SPLIT.json');local=read('SPLIT_RECONCILIATION.json')
    reconciliation={}
    for d in ['TSRS','RAM']:
        reconciliation[d]={}
        for s in ['train','val']:
            li=local['splits'][d][s]['local_ids'];ri=remote['datasets'][d][s]['ids']
            reconciliation[d][s]={'local_count':len(li),'remote_count':len(ri),'missing_local':sorted(set(ri)-set(li)),
                                  'extra_local':sorted(set(li)-set(ri)),'duplicate_remote':remote['datasets'][d][s]['duplicate_ids']}
    copies=[]
    for path in (ROOT/'data/raw_variants').glob('TSRS_RSNA-Epiphysis*/train_labels/1833.png'):
        if 'test' in path.parts[-3].lower():continue
        copies.append({'path':str(path),'sha256':sha(path),'matches_server_original':sha(path)==remote['1833_label']['sha256']})
    metadata=[]
    for row in read('SOURCE_TRACE.json')['roots']:
        for p in row['annotation_candidates']:
            p=Path(p)
            if p.suffix=='.json':
                x=json.loads(p.read_text(encoding='utf-8-sig'))
                if isinstance(x,dict):metadata.append({'path':str(p),'sha256':sha(p),'keys':list(x)[:20],
                    'source_path':x.get('source_path'),'policy':x.get('policy',x.get('scope'))})
    write('SOURCE_FOLLOWUP.json',{'status':'PASS bounded source tracing','metadata_checked':metadata,
          'outside_root_source':'reannotation_metadata references G:/gutou/标注/掌骨数据标注作业返回/flat_output_suffix; not traversed outside authorized root',
          'overlap_conclusion':'No complete multi-member annotation verified among these exported masks/metadata; source could hold more information but is not inspected',
          'current_split_reconciliation':reconciliation,'1833_local_variant_candidates':copies,
          'raw_dataset_modified':False,'test_labels_read':False})
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LocalSupervisionTests))
    write('LOCAL_SUPERVISION_TESTS.json',{'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,
            'scope':'CPU synthetic state/unknown/padding checks','version':sha(ROOT/'scripts/local_supervision_v3.py'),
            'command':f'{sys.executable} scripts/finalize_round2_evidence.py'})
    evidence={};cases=[]
    for d in ['TSRS','RAM']:
        x=read('N0_EQUIVALENCE_'+d+'.json')
        versions={name:sha(ROOT/'scripts'/name)==v for name,v in x['source_version'].items()}
        evidence[d]={'batch_status':x['status'],'local_matches_remote_code':versions,'checkpoint':x['source_sha256'],
                     'cases':[y['case'] for y in x['native_equivalence']], 'arm_status':x['arm_status'],
                     'input_channel_semantics':['unmodified normalized image','aligned heatmap in B/D; zero in A/C'],
                     'full_val_this_round':'NOT_RUN','performance_claim':'NONE'}
        for y in x['native_equivalence']:cases.append({'dataset':d,**y})
    write('REAL_BATCH_CASES.json',{'cases':cases,'test_used':False,'per_dataset_optimizer_steps_on_temporary_copy':1})
    write('ACCEPTANCE_STATUS.json',{'status':'PARTIAL_READY','datasets':evidence,
          'image_only_prior_weights':'BLOCKED no validated trained image-only checkpoint',
          'TSRS_overlap_supervision':'UNKNOWN; C/D research definition pending',
          'pilot':'NOT_RUN','training_epochs':0,'new_validation_metrics':'NOT_RUN',
          'local_assets':'ASSET_LOCK.json','isolated_interface':'structural_prior_v3.py',
          'legacy_files_preserved':True})
    if not result.wasSuccessful() or not all(all(v['local_matches_remote_code'].values()) for v in evidence.values()):raise SystemExit(1)
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
