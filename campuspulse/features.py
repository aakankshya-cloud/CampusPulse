"""Predeclared feature definitions; only development participants are summarized."""
import hashlib
import numpy as np
import pandas as pd
from .model import partition
from .studentlife import SCHEMA, decode_sleep_hours, interval_daily

NEW_SCHEMA = [
    {'id': 'exercise_day_share', 'label': 'Reported exercise frequency', 'unit': 'fraction of reported days',
     'min': 0, 'max': 1, 'step': .01,
     'description': 'Share of observed exercise-report days with a Yes response to vigorous exercise. Self-report, not sensor-derived exercise minutes.'},
    {'id': 'sleep_change_hours', 'label': 'Change in reported sleep', 'unit': 'hours: second half minus first half',
     'min': -12, 'max': 12, 'step': .1,
     'description': 'Mean sleep in days 15–28 minus days 1–14; at least three observed days in each half. Negative means less sleep later.'},
    {'id': 'conversation_sd_minutes', 'label': 'Day-to-day conversation variation', 'unit': 'minutes (SD)',
     'min': 0, 'max': 1440, 'step': 1,
     'description': 'Sample SD of detected conversation minutes across event-days. Not a measure of total social interaction.'},
    {'id': 'conversation_change_minutes', 'label': 'Change in detected conversation', 'unit': 'minutes: second half minus first half',
     'min': -1440, 'max': 1440, 'step': 1,
     'description': 'Second-half minus first-half mean, with three recorded event-days per half. Negative means less detected conversation later; missing event-days are not zeros.'},
    {'id': 'phone_lock_sd_hours', 'label': 'Day-to-day phone-lock variation', 'unit': 'hours (SD)',
     'min': 0, 'max': 24, 'step': .1,
     'description': 'Sample SD of phone-locked duration across recorded event-days. Not variation in screen-use duration.'},
    {'id': 'phone_lock_change_hours', 'label': 'Change in phone-locked duration', 'unit': 'hours: second half minus first half',
     'min': -24, 'max': 24, 'step': .1,
     'description': 'Second-half minus first-half mean of recorded phone-locked hours, requiring three event-days in each half.'},
    {'id': 'night_phone_locked_hours', 'label': 'Nighttime phone-locked duration', 'unit': 'hours/night with lock events',
     'min': 0, 'max': 6, 'step': .1,
     'description': 'Mean recorded locked time between midnight and 06:00, requiring three nights with recorded lock events. This is not nighttime phone-use duration.'}]


def half_change(daily, value, start, end, minimum=3):
    """Never treat unobserved dates as zero. Half-window boundaries are local dates."""
    daily = daily[(daily.date.astype(str) >= start.date().isoformat()) &
                  (daily.date.astype(str) < end.date().isoformat())]
    midpoint = (start + pd.DateOffset(days=14)).date().isoformat()
    first = daily[daily.date.astype(str) < midpoint].groupby('student_id')[value].agg(['mean', 'count'])
    last = daily[daily.date.astype(str) >= midpoint].groupby('student_id')[value].agg(['mean', 'count'])
    result = last['mean'] - first['mean']
    return result.where((first['count'] >= minimum) & (last['count'] >= minimum))


def daily_sleep(frame, start, end):
    frame = frame.copy()
    timestamps = pd.to_datetime(frame.timestamp, unit='s', utc=True).dt.tz_convert('America/New_York')
    frame = frame[(timestamps >= start) & (timestamps < end)].copy()
    frame['student_id'] = frame.uid.astype(str)
    frame['date'] = timestamps.loc[frame.index].dt.date.astype(str)
    frame['hours'] = decode_sleep_hours(frame.hour)
    return frame.groupby(['student_id', 'date'], as_index=False).hours.median().dropna(subset=['hours'])


def daily_exercise(frame, start, end):
    frame = frame.copy()
    timestamps = pd.to_datetime(frame.timestamp, unit='s', utc=True).dt.tz_convert('America/New_York')
    frame = frame[(timestamps >= start) & (timestamps < end)].copy()
    frame['student_id'] = frame.uid.astype(str)
    frame['date'] = timestamps.loc[frame.index].dt.date.astype(str)
    # Source EMA definition: have=1 Yes, have=2 No. Other values are unknown.
    frame['exercise'] = pd.to_numeric(frame['have'], errors='coerce').map({1: 1., 2: 0.})
    # Conflicting answers on a date are excluded, not guessed or weighted twice.
    grouped = frame.groupby(['student_id', 'date']).exercise
    return grouped.mean().where(grouped.nunique() == 1).dropna().reset_index()


def engineer(base, metadata, sleep, conversation, phonelock, exercise):
    if base.student_id.duplicated().any() or base.student_id.isna().any():
        raise ValueError('Base cohort requires one row per participant.')
    train_idx, test_idx = partition(base)
    development = base.iloc[train_idx].copy()
    dev_ids = set(development.student_id.astype(str))
    test_ids = set(base.iloc[test_idx].student_id.astype(str))
    # Exclude reserved participants before processing raw channels.
    sleep, conversation, phonelock, exercise = [
        d[d.uid.astype(str).isin(dev_ids)].copy() for d in [sleep, conversation, phonelock, exercise]]
    start, end = [pd.Timestamp(v).tz_convert('America/New_York') for v in metadata['feature_window']]
    sleeps = daily_sleep(sleep, start, end)
    conversations = interval_daily(conversation, start, end)
    locks = interval_daily(phonelock, start, end)
    locks['hours'] = locks.minutes / 60
    nights = interval_daily(phonelock, start, end, hour_window=(0, 6))
    nights['hours'] = nights.minutes / 60
    exercises = daily_exercise(exercise, start, end)
    result = development.set_index('student_id')
    exercise_stats = exercises.groupby('student_id').exercise.agg(['mean', 'count'])
    result['exercise_day_share'] = exercise_stats['mean'].where(exercise_stats['count'] >= 3)
    result['sleep_change_hours'] = half_change(sleeps, 'hours', start, end)
    result['conversation_sd_minutes'] = conversations.groupby('student_id').minutes.std()
    result['conversation_change_minutes'] = half_change(conversations, 'minutes', start, end)
    result['phone_lock_sd_hours'] = locks.groupby('student_id').hours.std()
    result['phone_lock_change_hours'] = half_change(locks, 'hours', start, end)
    night_stats = nights.groupby('student_id').hours.agg(['mean', 'count'])
    result['night_phone_locked_hours'] = night_stats['mean'].where(night_stats['count'] >= 3)
    schema = [*SCHEMA, *NEW_SCHEMA]
    # Preserve unknowns. Training later must impute inside each validation fold.
    result = result[[s['id'] for s in schema] + ['loneliness_score']].reset_index()
    for s in schema:
        result[s['id']] = pd.to_numeric(result[s['id']], errors='raise').astype(float)
        v = result[s['id']]
        if np.isinf(v).any() or (v.notna() & ~v.between(s['min'], s['max'])).any():
            raise ValueError('Engineered feature violates definition: ' + s['id'])
    summaries = []
    for s in NEW_SCHEMA:
        values = result[s['id']]
        summaries.append({**s, 'observed_students': int(values.notna().sum()),
                          'missing_students': int(values.isna().sum()),
                          'median': float(values.median()) if values.notna().any() else None})
    summary = {'source': 'studentlife_real', 'stage': 'Candidate feature definitions; the model comparison report identifies which features are used.',
        'original_dataset_sha256': hashlib.sha256(base.to_csv(index=False).encode()).hexdigest(),
        'development_students': len(development), 'reserved_students': len(test_idx),
        'original_feature_count': len(SCHEMA), 'candidate_feature_count': len(schema),
        'features': summaries,
        'limitations': ['Exercise is an active self-report, not passive activity sensing.',
            'Locked duration is not phone-use duration. Nighttime use remains unmeasured.',
            'Conversation change is a change in detected events, not a confirmed drop in social interaction.',
            'Event-day means and changes depend on recording coverage.',
            'Seven reserved participants were excluded before raw feature processing.',
            'No feature selection by outcome correlation, no model retraining, and no new holdout evaluation performed.']}
    updated = {**metadata, 'features': schema, 'development_only': True,
        'included_students': len(result), 'base_eligible_students': len(base),
        'reserved_students': len(test_ids), 'source': 'studentlife_development_candidates',
        'limitations': metadata['limitations'] + summary['limitations']}
    split = {'development_ids': sorted(dev_ids), 'reserved_ids': sorted(test_ids),
             'original_dataset_sha256': summary['original_dataset_sha256']}
    return result, updated, summary, split


def plot_candidates(features, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    selected = ['exercise_day_share', 'sleep_change_hours',
                'conversation_change_minutes', 'night_phone_locked_hours']
    definitions = {s['id']:s for s in NEW_SCHEMA}
    fig, axes = plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for axis, name in zip(axes.flat, selected):
        rows = features[[name, 'loneliness_score']].dropna()
        axis.scatter(rows[name],rows.loneliness_score,s=35,color='#237b54',alpha=.8,
                     edgecolors='white',linewidths=.5)
        axis.set_title(definitions[name]['label'],loc='left',fontsize=12,fontweight='bold')
        axis.set_xlabel(definitions[name]['unit'],fontsize=10)
        axis.set_ylabel('Post-study questionnaire score')
        axis.set_ylim(20,80)
        axis.text(.03,.94,f'n={len(rows)} development students',transform=axis.transAxes,va='top',fontsize=9)
        axis.spines[['top','right']].set_visible(False)
        axis.grid(axis='y',color='#e6ebe6');axis.set_axisbelow(True)
    fig.suptitle('New behavior summaries: development data only',fontsize=18,fontweight='bold',color='#19382d')
    fig.supxlabel('Candidate feature exploration only. Associations do not prove causes or prediction quality.',fontsize=10)
    fig.savefig(Path(output)/'candidates.png',dpi=160,facecolor='white')
    plt.close(fig)
