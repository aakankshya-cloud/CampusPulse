"""Step 2: generate candidate features, without retraining the model."""
import argparse
import json
from pathlib import Path
import zlib
import pandas as pd
from campuspulse.features import engineer, plot_candidates
from campuspulse.studentlife import read_rds
from prepare_studentlife import download_member

EXERCISE = {'offset':49583, 'compressed_size':5386, 'size':5410,
            'crc32':283883494, 'method':8}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, default=Path('data/private_studentlife'))
    parser.add_argument('--output', type=Path, default=Path('data/private_studentlife_v2'))
    parser.add_argument('--model-dir', type=Path, default=Path('artifacts/studentlife'))
    args = parser.parse_args()
    if not (args.prepared / 'features.csv').is_file():
        parser.error('Run prepare_studentlife.py first to download and prepare the source data.')
    cache = args.prepared / 'raw'
    cache.mkdir(parents=True, exist_ok=True)
    exercise_path = cache / 'Exercise.Rds'
    if not exercise_path.exists():
        print('Fetching Exercise.Rds (about 5 KB)', flush=True)
        exercise_path.write_bytes(download_member('dataset_rds/EMA/Exercise.Rds', EXERCISE))
    raw = exercise_path.read_bytes()
    if len(raw) != EXERCISE['size'] or zlib.crc32(raw) != EXERCISE['crc32']:
        raise ValueError('Exercise file failed archive integrity check.')
    tables = {n:read_rds(cache / (n+'.Rds')) for n in ['Sleep','conversation','phonelock','Exercise']}
    metadata = json.loads((args.prepared / 'metadata.json').read_text(encoding='utf-8'))
    base = pd.read_csv(args.prepared / 'features.csv', dtype={'student_id':str})
    features, updated, summary, split = engineer(base, metadata, tables['Sleep'],
        tables['conversation'], tables['phonelock'], tables['Exercise'])
    if (args.model_dir / 'metrics.json').exists():
        model_report = json.loads((args.model_dir / 'metrics.json').read_text())
        if model_report.get('dataset_sha256') != summary['original_dataset_sha256']:
            raise ValueError('Base cohort differs from this model. Use the original prepared data to preserve its split.')
    args.output.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output / 'features.csv', index=False)
    (args.output / 'metadata.json').write_text(json.dumps(updated, indent=2), encoding='utf-8')
    (args.output / 'split_manifest.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    public = args.model_dir / 'features_v2'
    public.mkdir(parents=True, exist_ok=True)
    (public / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    plot_candidates(features, public)
    print(f'Prepared {len(summary["features"])} new features for {len(features)} development students; {summary["reserved_students"]} reserved students excluded.')
    print('Candidate data:', args.output)
    print('The existing prediction model is unchanged. Step 3 will evaluate these candidates.')

if __name__ == '__main__':
    main()
