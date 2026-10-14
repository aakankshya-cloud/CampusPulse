import argparse
import json
from pathlib import Path
import pandas as pd
from campuspulse.model import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True,
                        help='Folder containing features.csv and metadata.json')
    parser.add_argument('--output', default='artifacts')
    args = parser.parse_args()
    metadata = json.loads((args.prepared / 'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('development_only'):
        parser.error('These candidate features contain development students only. Use a development-only CV comparison, not another holdout split.')
    df = pd.read_csv(args.prepared / 'features.csv', dtype={'student_id': str})
    report = train(df, args.output, source=metadata['source'],
        target_description=metadata['target'], schema=metadata['features'],
        target_bounds=metadata['target_bounds'], provenance=metadata)
    print(json.dumps({k: report[k] for k in ['source', 'selected_model', 'train_students',
        'test_students', 'holdout', 'beats_baseline_on_holdout']}, indent=2))

if __name__ == '__main__':
    main()
