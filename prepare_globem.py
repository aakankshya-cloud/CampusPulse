"""Local adapter for legitimately obtained GLOBEM files. Does not download data.

Inspect headers first. You must provide exact raw feature and UCLA post-score
column names; the adapter deliberately does not guess a questionnaire total.
"""
import argparse
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

def prepare(features, post, columns, target, min_days=7, window_days=28):
    if not columns or len(set(columns)) != len(columns):
        raise ValueError('Provide distinct raw daily feature columns.')
    for column in columns:
        if not re.fullmatch(r'f_(slp|screen|steps|loc|blue|call):[A-Za-z0-9_]+:allday', column):
            raise ValueError(f'Not a supported raw all-day sensor feature: {column}')
        if '_norm' in column or '_dis' in column:
            raise ValueError('Participant-wide normalized/discretized features are excluded to avoid future-information leakage.')
    if 'UCLA' not in target.upper() or 'POST' not in target.upper():
        raise ValueError('Choose the documented UCLA post-survey total column, not depression labels or item answers.')
    missing = (set(['pid', 'date', *columns]) - set(features)) | (set(['pid', 'date', target]) - set(post))
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    features, post = features.copy(), post.copy()
    if features.pid.isna().any() or post.pid.isna().any():
        raise ValueError('Participant IDs must not be missing.')
    features['pid'] = features.pid.astype(str)
    post['pid'] = post.pid.astype(str)
    features['date'] = pd.to_datetime(features.date, errors='raise').dt.normalize()
    post['date'] = pd.to_datetime(post.date, errors='raise').dt.normalize()
    if features.date.isna().any() or post.date.isna().any():
        raise ValueError('Dates must not be missing.')
    if features.duplicated(['pid', 'date']).any() or post.pid.duplicated().any():
        raise ValueError('Expect one feature row per participant/date and one post survey per participant in a single cohort.')
    for column in columns:
        features[column] = pd.to_numeric(features[column], errors='raise')
        if np.isinf(features[column]).any():
            raise ValueError('Infinite sensor value found.')
    outcome = pd.to_numeric(post[target], errors='raise')
    if (outcome.notna() & ~outcome.between(10, 40)).any():
        raise ValueError('Expected GLOBEM 10-item UCLA total on 10–40 scale. Verify the codebook and selected column.')
    post[target] = outcome
    rows = []
    for _, survey in post.dropna(subset=[target]).iterrows():
        window = features[(features.pid == survey.pid) & (features.date < survey.date)
            & (features.date >= survey.date - pd.Timedelta(days=window_days))]
        if len(window) < min_days or any(window[c].notna().sum() < min_days for c in columns):
            continue
        rows.append({'student_id': survey.pid, 'loneliness_score': float(survey[target]),
                     **{c: float(window[c].mean()) for c in columns}})
    df = pd.DataFrame(rows, columns=['student_id', 'loneliness_score', *columns])
    if df.empty:
        raise ValueError('No eligible rows. Check shifted dates, feature availability, and column mapping.')
    schema = []
    for c in columns:
        lo, hi = float(df[c].min()), float(df[c].max())
        if hi == lo:
            hi = lo + 1
        schema.append({'id': c, 'label': c, 'unit': 'source feature units; see GLOBEM/RAPIDS codebook',
                       'min': lo, 'max': hi, 'step': max((hi-lo)/1000, .001)})
    metadata = {'source': 'globem_real_local', 'dataset': 'GLOBEM single cohort (user-selected)',
        'source_url': 'https://physionet.org/content/globem/1.1/', 'target': target,
        'target_bounds': [10, 40], 'features': schema, 'included_students': len(df),
        'window_days': window_days, 'minimum_observed_days_per_feature': min_days,
        'limitations': ['Restricted data: retain locally under your PhysioNet agreement; do not publish data or derived artifacts without verifying permitted use.',
            'This adapter has only fixture tests until run against your authorized files.',
            'Only one cohort is supported per run. No cross-year generalization has been evaluated.',
            'Raw all-day features only; provided participant-wide normalization and history features are excluded.',
            'Window ends before the post-survey date. Missing feature-days are excluded, not set to zero.',
            'Daily averages and source units differ from StudentLife; no pooled or transferred model is claimed.',
            'Full dataset descriptive ranges are used for UI bounds, not training or model selection.']}
    return df, metadata

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cohort', type=Path, required=True, help='One authorized INS-W_* directory')
    p.add_argument('--inspect', action='store_true', help='Show headers only; writes no participant data')
    p.add_argument('--target-column')
    p.add_argument('--feature-columns', nargs='+')
    p.add_argument('--output', type=Path, default=Path('data/private_globem'))
    args = p.parse_args()
    fp, sp = args.cohort / 'FeatureData/rapids.csv', args.cohort / 'SurveyData/post.csv'
    if args.inspect:
        for path in [sp, fp]:
            print(path.name, json.dumps(pd.read_csv(path, nrows=0).columns.tolist()))
        return
    if not args.target_column or not args.feature_columns:
        p.error('Inspect first, then provide --target-column and --feature-columns explicitly.')
    df, metadata = prepare(pd.read_csv(fp, dtype={'pid': str}, usecols=['pid', 'date', *args.feature_columns]),
                           pd.read_csv(sp, dtype={'pid': str}, usecols=['pid', 'date', args.target_column]),
                           args.feature_columns, args.target_column)
    metadata['cohort'] = args.cohort.name
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / 'features.csv', index=False)
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Prepared', len(df), 'participants in', args.output)

if __name__ == '__main__':
    main()
