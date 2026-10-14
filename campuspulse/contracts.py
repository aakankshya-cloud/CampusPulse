import numpy as np
import pandas as pd

def validate_schema_input(payload, schema):
    if not isinstance(payload, dict):
        raise ValueError('Expected a JSON object.')
    values = {}
    for item in schema:
        name = item['id']
        value = payload.get(name)
        if value is None or isinstance(value, bool):
            raise ValueError(f'Provide a number for {name}.')
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f'Provide a number for {name}.') from None
        if not np.isfinite(value) or not item['min'] <= value <= item['max']:
            raise ValueError(f'{name} must be between {item["min"]} and {item["max"]}.')
        values[name] = value
    return values

def validate_real_training(df, schema, bounds):
    names = [s['id'] for s in schema]
    if len(names) != len(set(names)) or set(names) & {'student_id', 'loneliness_score'}:
        raise ValueError('Feature names must be unique and exclude ID and target.')
    for name in ['student_id', 'loneliness_score', *names]:
        if name not in df:
            raise ValueError(f'Missing column: {name}')
    if df.student_id.isna().any() or df.student_id.astype(str).str.strip().eq('').any():
        raise ValueError('Participant IDs must be present.')
    if df.student_id.nunique() < 20:
        raise ValueError('At least 20 participants are required by this splitter; this does not establish adequate power.')
    for s in schema:
        v = pd.to_numeric(df[s['id']], errors='raise')
        if np.isinf(v).any() or v.notna().sum() < 2 or (v.notna() & ~v.between(s['min'], s['max'])).any():
            raise ValueError(f'Invalid values for {s["id"]}.')
        df[s['id']] = v
    y = pd.to_numeric(df.loneliness_score, errors='raise')
    if y.isna().any() or not y.between(*bounds).all():
        raise ValueError('Outcome is missing or outside the documented instrument scale.')
    df['loneliness_score'] = y
    return df
