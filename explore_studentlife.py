"""Run: python explore_studentlife.py --prepared data/private_studentlife"""
import argparse
import json
from pathlib import Path
import pandas as pd
from campuspulse.explore import analyze, render

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, default=Path('data/private_studentlife'))
    p.add_argument('--output', type=Path, default=Path('artifacts/studentlife/eda'))
    args = p.parse_args()
    metadata = json.loads((args.prepared / 'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('source') != 'studentlife_real':
        p.error('This command is specifically for the documented StudentLife experiment.')
    df = pd.read_csv(args.prepared / 'features.csv', dtype={'student_id': str})
    coverage = pd.read_csv(args.prepared / 'coverage.csv', dtype={'student_id': str})
    summary, development, development_coverage = analyze(df, metadata, coverage)
    render(summary, development, development_coverage, args.output)
    print(f'Explored {summary["development_students"]} development students; {summary["reserved_students"]} holdout students excluded.')
    for feature in summary['features']:
        print(feature['label'], 'missing:', feature['missing_students'], 'Spearman r:', round(feature['spearman_r'],3))
    print('Charts and summary:', args.output)

if __name__ == '__main__':
    main()
