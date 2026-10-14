"""Step 3: compare original and expanded features using development-only nested CV."""
import argparse
import json
from pathlib import Path
import shutil
import pandas as pd
from campuspulse.compare import compare

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prepared',type=Path,default=Path('data/private_studentlife_v2'))
    p.add_argument('--output',type=Path,default=Path('artifacts/studentlife_v2'))
    p.add_argument('--original-model-dir',type=Path,default=Path('artifacts/studentlife'))
    args=p.parse_args()
    if not (args.prepared/'features.csv').exists():
        p.error('Run engineer_studentlife.py first.')
    metadata=json.loads((args.prepared/'metadata.json').read_text(encoding='utf-8'))
    split=json.loads((args.prepared/'split_manifest.json').read_text(encoding='utf-8'))
    df=pd.read_csv(args.prepared/'features.csv',dtype={'student_id':str})
    print('Comparing five configurations on development students only. This may take a minute.',flush=True)
    report,private=compare(df,metadata,split,args.output)
    (args.prepared/'cv_predictions_private.json').write_text(json.dumps(private,indent=2),encoding='utf-8')
    for directory in ['eda','features_v2']:
        source=args.original_model_dir/directory
        if source.exists():shutil.copytree(source,args.output/directory,dirs_exist_ok=True)
    print('Selected:',report['selected_model'])
    for r in report['comparison']:print(r['model'], 'CV MAE:',round(r['cv_mae'],3))
    print('Nested-CV selection procedure MAE:',round(report['validation']['mae'],3))
    print('Baseline out-of-fold MAE:',round(report['validation']['baseline_mae'],3))
    print('Reserved students excluded:',report['reserved_students'])
    print('Model and report:',args.output)

if __name__=='__main__':main()
