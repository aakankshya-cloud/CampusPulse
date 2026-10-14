"""Measured StudentLife features. Never maps absent sensor events to zero."""
import warnings
import numpy as np
import pandas as pd

# Revised UCLA item wording in the archive: Q4 says 'I do not feel alone'.
# It is positively worded and must be reversed, along with the other nine.
REVERSE = {1, 4, 5, 6, 9, 10, 15, 16, 19, 20}
RESPONSES = {'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4}
SCHEMA = [
    {'id': 'reported_sleep_hours', 'label': 'Reported sleep duration', 'unit': 'hours/night',
     'min': 0, 'max': 12, 'step': .1},
    {'id': 'sleep_duration_sd', 'label': 'Sleep duration variability', 'unit': 'hours (SD)',
     'min': 0, 'max': 6, 'step': .1},
    {'id': 'conversation_event_minutes', 'label': 'Detected conversation on recorded days',
     'unit': 'minutes/day with events', 'min': 0, 'max': 1440, 'step': 1},
    {'id': 'phone_locked_hours', 'label': 'Phone locked on recorded days',
     'unit': 'hours/day with events', 'min': 0, 'max': 24, 'step': .1}]


def read_rds(path):
    try:
        import rdata
    except ImportError:
        raise RuntimeError('Install real-data dependencies: pip install -r requirements-real.txt') from None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return rdata.read_rds(path)


def decode_sleep_hours(series):
    codes = pd.to_numeric(series, errors='coerce')
    return (2.5 + codes * .5).where(codes.isin(range(2, 20)))


def loneliness_scores(survey):
    post = survey[survey['type'].astype(str).str.lower().eq('post')].copy()
    if post.uid.duplicated().any():
        raise ValueError('Duplicate post-study survey for a participant.')
    items = pd.DataFrame(index=post.index)
    for number in range(1, 21):
        values = post[f'Q{number}'].astype('string').str.strip().map(RESPONSES)
        items[number] = 5 - values if number in REVERSE else values
    # Require all 20 observed answers. No questionnaire imputation.
    complete = items.notna().all(axis=1)
    result = pd.DataFrame({'student_id': post.loc[complete, 'uid'].astype(str),
        'loneliness_score': items.loc[complete].sum(axis=1).astype(float)})
    return result.reset_index(drop=True), int((~complete).sum())


def interval_daily(frame, start, end, hour_window=None):
    """Union overlapping intervals, clip the window, then split at local midnight."""
    values = {}
    starts = pd.to_numeric(frame.start_timestamp, errors='coerce')
    ends = pd.to_numeric(frame.end_timestamp, errors='coerce')
    for uid in frame.uid.astype(str).unique():
        mask = frame.uid.astype(str).eq(uid)
        intervals = []
        for a, b in zip(starts[mask], ends[mask]):
            if not np.isfinite(a) or not np.isfinite(b) or b <= a:
                continue
            a, b = max(a, start.timestamp()), min(b, end.timestamp())
            if b > a:
                intervals.append((a, b))
        intervals.sort()
        merged = []
        for a, b in intervals:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        for a, b in merged:
            current = pd.Timestamp(a, unit='s', tz='UTC').tz_convert('America/New_York')
            finish = pd.Timestamp(b, unit='s', tz='UTC').tz_convert('America/New_York')
            while current < finish:
                midnight = current.normalize() + pd.DateOffset(days=1)
                stop = min(finish, midnight)
                counted_start, counted_stop = current, stop
                if hour_window is not None:
                    lo, hi = hour_window
                    if not 0 <= lo < hi <= 24:
                        raise ValueError('Hour window must satisfy 0 <= start < end <= 24.')
                    day = current.date().isoformat()
                    window_start = pd.Timestamp(f'{day} {lo:02d}:00', tz='America/New_York')
                    window_end = midnight if hi == 24 else pd.Timestamp(f'{day} {hi:02d}:00', tz='America/New_York')
                    counted_start, counted_stop = max(current, window_start), min(stop, window_end)
                if counted_stop > counted_start:
                    key = (uid, current.date().isoformat())
                    values[key] = values.get(key, 0.) + (counted_stop - counted_start).total_seconds() / 60
                current = stop
    return pd.DataFrame([{'student_id': uid, 'date': day, 'minutes': minutes}
                         for (uid, day), minutes in values.items()],
                        columns=['student_id', 'date', 'minutes'])


def prepare(survey, sleep, conversation, phonelock):
    start = pd.Timestamp('2013-03-27', tz='America/New_York')
    end = pd.Timestamp('2013-04-24', tz='America/New_York')
    labels, incomplete = loneliness_scores(survey)
    sleep = sleep.copy()
    sleep['student_id'] = sleep.uid.astype(str)
    times = pd.to_datetime(sleep.timestamp, unit='s', utc=True).dt.tz_convert('America/New_York')
    sleep = sleep[(times >= start) & (times < end)].copy()
    sleep['date'] = times.loc[sleep.index].dt.date
    # EMA_definition.json encodes categories, NOT literal hours:
    # 1=<3, 2=3.5, 3=4, ..., 19=12. 0 is undocumented.
    # Exclude category 1 (left-censored) and invalid codes instead of inventing hours.
    sleep['hours'] = decode_sleep_hours(sleep.hour)
    excluded_sleep = int(sleep.hours.isna().sum())
    daily_sleep = sleep.groupby(['student_id', 'date']).hours.median().dropna()
    sf = daily_sleep.groupby(level=0).agg(['mean', 'std', 'count'])
    sf.columns = ['reported_sleep_hours', 'sleep_duration_sd', 'sleep_days']
    conversation_daily = interval_daily(conversation, start, end)
    locked_daily = interval_daily(phonelock, start, end)
    cf = conversation_daily.groupby('student_id').minutes.agg(['mean', 'count'])
    cf.columns = ['conversation_event_minutes', 'conversation_days']
    pf = locked_daily.groupby('student_id').minutes.agg(['mean', 'count'])
    pf.columns = ['phone_locked_hours', 'locked_days']
    pf['phone_locked_hours'] /= 60
    joined = labels.set_index('student_id').join(sf).join(cf).join(pf)
    # Require at least a week of observed sleep reports and event-days per channel.
    eligible = (joined.sleep_days >= 7) & (joined.conversation_days >= 7) & (joined.locked_days >= 7)
    coverage = joined[['sleep_days', 'conversation_days', 'locked_days']].fillna(0).copy()
    coverage['included'] = eligible
    result = joined.loc[eligible, [s['id'] for s in SCHEMA] + ['loneliness_score']].reset_index()
    provenance = {
        'source': 'studentlife_real',
        'dataset': 'StudentLife (Dartmouth), Zenodo RDS conversion v1.0.0',
        'source_url': 'https://zenodo.org/records/3529253',
        'original_study': 'Wang et al., StudentLife, UbiComp 2014',
        'license_as_listed_by_archive_api': 'CC-BY-4.0',
        'target': 'Post-study revised UCLA 20-item total; 10 positive items reversed including Q4.',
        'target_bounds': [20, 80], 'feature_window': [start.isoformat(), end.isoformat()],
        'post_surveys': len(labels), 'incomplete_post_surveys_dropped': incomplete,
        'included_students': len(result), 'coverage_exclusions': int((~eligible).sum()),
        'unmapped_or_censored_sleep_responses_excluded': excluded_sleep,
        'features': SCHEMA,
        'limitations': [
            'Small single-campus observational cohort; no clinical validation.',
            'Survey table provides pre/post type, not individual completion timestamps. Early four-week window precedes intended end-term survey; exact individual ordering cannot be audited.',
            'Sleep is self-reported EMA, not passive sensing. Left-censored <3-hour responses and invalid codes excluded; this can bias sleep summaries.',
            'Conversation and phone-lock means use days with recorded events only. Missing event-days are not assumed to be zero and complete sensor coverage is unknown.',
            'Detected conversation is not total social time. Locked duration is not screen time.',
            'Only participants with >=7 observed days per channel are included; coverage filtering may introduce selection bias.',
            'Features are four-week averages; single-day user inputs are out of task.',
            'Different target scale from GLOBEM: no pooling or cross-dataset accuracy claim.']}
    return result, provenance, coverage.reset_index()
