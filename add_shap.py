"""Step 4: attach SHAP to the selected model without changing its predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from campuspulse.explain import prepare, plot_summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-dir',type=Path,default=Path('artifacts/studentlife_v2'))
    p.add_argument('--prepared',type=Path,default=Path('data/private_studentlife_v2'))
    args=p.parse_args()
    if not (args.prepared/'features.csv').exists():p.error('Run engineer_studentlife.py first.')
    bundle=joblib.load(args.model_dir/'model.joblib')
    if bundle['report'].get('evaluation_kind')!='nested_cv_development_only':
        p.error('Use the development-only model produced by compare_models.py.')
    data=pd.read_csv(args.prepared/'features.csv',dtype={'student_id':str})
    split=json.loads((args.prepared/'split_manifest.json').read_text())
    if set(data.student_id)!=set(split['development_ids']) or set(data.student_id)&set(split['reserved_ids']):
        p.error('Background participants differ from the locked development split.')
    fingerprint=hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()
    if fingerprint!=bundle['report']['dataset_sha256']:
        p.error('Prepared data differs from the model training data. Use the matching model and candidate CSV.')
    before=bundle['model'].predict(data[bundle['report']['features']])
    summary=prepare(bundle,data)
    assert np.array_equal(before,bundle['model'].predict(data[bundle['report']['features']]))
    output=args.model_dir/'shap';output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    plot_summary(summary,output)
    joblib.dump(bundle,args.model_dir/'model.joblib')
    (args.model_dir/'metrics.json').write_text(json.dumps(bundle['report'],indent=2),encoding='utf-8')
    print('SHAP prepared for',summary['background_students'],'development students; no reserved students used.')
    print('Maximum reconstruction error:',summary['max_additivity_error'])
    print('Model predictions unchanged. Restart the app to enable the explanations.')

if __name__=='__main__':main()
