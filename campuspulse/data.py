from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ['sleep_hours', 'active_minutes', 'social_minutes', 'screen_hours',
            'study_hours', 'routine_variability']
BOUNDS = {'sleep_hours': (0, 16), 'active_minutes': (0, 360),
          'social_minutes': (0, 600), 'screen_hours': (0, 20),
          'study_hours': (0, 16), 'routine_variability': (0, 6)}
LABELS = dict(zip(FEATURES, ['Sleep (hours/day)', 'Activity (minutes/day)',
    'Social time (minutes/day)', 'Screen time (hours/day)',
    'Study (hours/day)', 'Bedtime variability (hours)']))
TARGET = 'loneliness_score'


def synthetic_data(seed=42, students=180, weeks=6):
    """Invented behavior and invented 20–80 scores; not a validated questionnaire."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(students):
        latent = rng.normal()
        person_noise = rng.normal(0, 6)
        for week in range(weeks):
            sleep = np.clip(rng.normal(7 - .3 * latent, .9), 3, 11)
            active = np.clip(rng.normal(45 - 8 * latent, 20), 0, 180)
            social = np.clip(rng.normal(65 - 20 * latent, 30), 0, 240)
            screen = np.clip(rng.normal(4.5 + .5 * latent, 1.4), .5, 12)
            study = np.clip(rng.normal(4, 1.2), 0, 10)
            variability = np.clip(rng.normal(1.2 + .3 * latent, .6), 0, 4)
            score = np.clip(47 - .15 * (social - 60) - .07 * (active - 40)
                + 1.2 * (screen - 4) + 2.5 * variability
                + 1.4 * abs(sleep - 7.5) + person_noise + rng.normal(0, 4), 20, 80)
            rows.append([f'demo_{i:03d}', week + 1, sleep, active, social,
                         screen, study, variability, score])
    return pd.DataFrame(rows, columns=['student_id', 'week', *FEATURES, TARGET])


def validate_training(df):
    required = ['student_id', *FEATURES, TARGET]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError('Missing CSV columns: ' + ', '.join(sorted(missing)))
    if df.student_id.isna().any() or df.student_id.astype(str).str.strip().eq('').any():
        raise ValueError('Every row needs a pseudonymous student_id.')
    if df.student_id.nunique() < 20:
        raise ValueError('At least 20 distinct students are required for this evaluation.')
    for feature in FEATURES:
        values = pd.to_numeric(df[feature], errors='raise')
        lo, hi = BOUNDS[feature]
        if np.isinf(values).any() or ((values < lo) | (values > hi)).any():
            raise ValueError(f'{feature} must be within {lo}–{hi}.')
        if values.notna().sum() < 2:
            raise ValueError(f'{feature} needs at least two observed values.')
        df[feature] = values
    y = pd.to_numeric(df[TARGET], errors='raise')
    if y.isna().any() or not np.isfinite(y).all() or not y.between(20, 80).all():
        raise ValueError('This prototype expects complete scores on a 20–80 scale.')
    df[TARGET] = y
    return df


def validate_input(payload):
    if not isinstance(payload, dict):
        raise ValueError('Expected a JSON object.')
    result = {}
    for key in FEATURES:
        value = payload.get(key)
        if value is None or isinstance(value, bool):
            raise ValueError(f'Provide a number for {key}.')
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f'Provide a number for {key}.') from None
        lo, hi = BOUNDS[key]
        if not np.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'{key} must be within {lo}–{hi}.')
        result[key] = value
    return result
