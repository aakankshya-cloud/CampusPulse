"""Descriptive analysis of development students only; never fits a model."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .model import partition


def analyze(df, metadata, coverage=None):
    if df.student_id.isna().any() or df.student_id.duplicated().any():
        raise ValueError('Expected one row per participant with a valid ID.')
    names = [f['id'] for f in metadata['features']]
    if len(df) < 20:
        raise ValueError('This analysis expects at least 20 eligible participants.')
    train_indices, test_indices = partition(df)
    development = df.iloc[train_indices].copy()
    heldout_ids = set(df.iloc[test_indices].student_id.astype(str))
    development_ids = set(development.student_id.astype(str))
    assert not development_ids & heldout_ids
    rows = []
    for item in metadata['features']:
        values = pd.to_numeric(development[item['id']], errors='raise')
        pair = pd.DataFrame({'x': values, 'y': development.loneliness_score}).dropna()
        sufficient = len(pair) >= 3 and pair.x.nunique() > 1 and pair.y.nunique() > 1
        rows.append({'feature': item['id'], 'label': item['label'], 'unit': item['unit'],
            'observed_students': int(values.notna().sum()),
            'missing_students': int(values.isna().sum()),
            'median': float(values.median()), 'min': float(values.min()),
            'max': float(values.max()), 'q25': float(values.quantile(.25)),
            'q75': float(values.quantile(.75)),
            'pearson_r': float(pair.x.corr(pair.y)) if sufficient else None,
            'spearman_r': float(pair.x.corr(pair.y, method='spearman')) if sufficient else None})
    coverage_dev = None
    coverage_summary = []
    if coverage is not None:
        coverage_dev = coverage[coverage.student_id.astype(str).isin(development_ids)].copy()
        if coverage_dev.student_id.duplicated().any() or len(coverage_dev) != len(development):
            raise ValueError('Coverage must contain exactly one matching row per development student.')
        for column, label in [('sleep_days', 'Sleep report days'),
                              ('conversation_days', 'Conversation event-days'),
                              ('locked_days', 'Phone-lock event-days')]:
            v = pd.to_numeric(coverage_dev[column], errors='raise')
            coverage_summary.append({'channel': label, 'median_days': float(v.median()),
                                     'min_days': int(v.min()), 'max_days': int(v.max())})
    summary = {
        'source': metadata['source'], 'scope': 'Development participants only; holdout outcomes and features excluded.',
        'development_students': len(development), 'reserved_students': len(test_indices),
        'target_bounds': metadata['target_bounds'], 'features': rows,
        'coverage': coverage_summary,
        'flow': {'complete_post_surveys': metadata.get('post_surveys'),
                 'eligible_students': len(df), 'coverage_exclusions': metadata.get('coverage_exclusions')},
        'input_sha256': hashlib.sha256(df.to_csv(index=False).encode()).hexdigest(),
        'limitations': ['These correlations are exploratory associations, not causes or feature-selection rules.',
            'Only 27 development students in the supplied StudentLife experiment; patterns are unstable.',
            'Zero missing prepared values does not imply complete raw sensor data.',
            'Event-day counts do not distinguish no activity from failed recording.',
            'Activity and actual screen use are not measured in this adapter; no plots are fabricated.',
            'The prior seven-person holdout remains reserved; this analysis does not evaluate a new model.']}
    return summary, development, coverage_dev


def render(summary, development, coverage, output):
    # Lazy import: the app can still run without plotting dependencies installed.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titleweight': 'bold', 'axes.labelcolor': '#304438',
        'text.color': '#19382d', 'axes.edgecolor': '#c9d3cc'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    for axis, feature in zip(axes.flat, summary['features']):
        axis.scatter(development[feature['feature']], development.loneliness_score,
                     s=38, color='#237b54', alpha=.8, edgecolors='white', linewidths=.5)
        axis.set_title(feature['label'], loc='left', fontsize=11, pad=12)
        axis.set_xlabel(feature['unit'])
        axis.set_ylabel('Post-study loneliness score (20–80)')
        axis.set_ylim(*summary['target_bounds'])
        axis.grid(axis='y', color='#e6ebe6', linewidth=.7)
        axis.set_axisbelow(True)
        r = feature['spearman_r']
        axis.text(.03, .95, f"n={feature['observed_students']} · Spearman r={r:.2f}" if r is not None else 'Correlation undefined',
                  transform=axis.transAxes, va='top', fontsize=9,
                  bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9})
    fig.suptitle('StudentLife: explore the development data', fontsize=19, fontweight='bold')
    fig.supxlabel('Each dot is one development student. No holdout students. Associations do not establish causes.', fontsize=10)
    fig.savefig(output / 'relationships.png', dpi=160, facecolor='white')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    bounds = summary['target_bounds']
    axes[0].hist(development.loneliness_score, bins=np.arange(bounds[0], bounds[1]+5, 5),
                 color='#237b54', edgecolor='white')
    axes[0].set(title='Questionnaire scores', xlabel='Post-study score', ylabel='Development students')
    axes[0].set_xlim(*bounds)
    if coverage is not None:
        columns = ['sleep_days', 'conversation_days', 'locked_days']
        for i, (column, label) in enumerate(zip(columns, ['Sleep reports', 'Conversation events', 'Phone-lock events'])):
            values = coverage[column].to_numpy()
            # Deterministic offsets expose overlapping points without changing y-values.
            offsets = np.linspace(-.13, .13, len(values))
            axes[1].scatter(i+offsets, values, s=25, color='#237b54', alpha=.6)
            axes[1].plot([i-.2, i+.2], [np.median(values)]*2, color='#ae5c2a', linewidth=3)
        axes[1].set_xticks(range(3), ['Sleep reports', 'Conversation events', 'Phone-lock events'])
        axes[1].axhline(7, color='#8a7257', linestyle='--', linewidth=1, label='Eligibility minimum: 7 days')
        axes[1].set_ylim(0, 29)
        axes[1].set(title='Observed days in the 28-day window', ylabel='Days with valid report / recorded event')
        axes[1].legend(loc='lower left', fontsize=8)
    else:
        axes[1].text(.5,.5,'Coverage table not provided',ha='center',va='center')
    fig.suptitle('Data coverage comes before model claims', fontsize=17, fontweight='bold')
    fig.supxlabel('Development students only. A day without a recorded event is not automatically zero activity.', fontsize=10)
    fig.savefig(output / 'coverage.png', dpi=160, facecolor='white')
    plt.close(fig)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
