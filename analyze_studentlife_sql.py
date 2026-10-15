"""Build a local development-only SQLite database and run four SQL queries."""
import argparse
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3

import pandas as pd
from campuspulse.model import partition

ROOT = Path(__file__).resolve().parent
CHANNELS = ('sleep_days', 'conversation_days', 'locked_days')


def analyze(data_dir, sql_path=ROOT / 'studentlife_analysis.sql'):
    data_dir = Path(data_dir)
    features = pd.read_csv(data_dir / 'features.csv', dtype={'student_id': str})
    coverage = pd.read_csv(data_dir / 'coverage.csv', dtype={'student_id': str})
    metadata = json.loads((data_dir / 'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('source') != 'studentlife_real':
        raise ValueError('Expected prepared StudentLife metadata.')
    for frame in (features, coverage):
        if frame.student_id.isna().any() or frame.student_id.duplicated().any():
            raise ValueError('Expected unique, nonmissing participant IDs.')
    if len(features) < 20:
        raise ValueError('Expected at least 20 eligible participants.')
    development_idx, heldout_idx = partition(features)
    development = features.iloc[development_idx]
    ids = development.student_id.tolist()
    if not set(ids).issubset(set(coverage.student_id)):
        raise ValueError('Missing development coverage rows.')
    selected_coverage = coverage.set_index('student_id').loc[ids]
    start, end = map(pd.Timestamp, metadata['feature_window'])
    # Calendar-day difference avoids daylight-saving duration errors.
    window_days = (end.date() - start.date()).days
    if window_days <= 0:
        raise ValueError('Invalid observation window.')
    feature_rows = []
    for item in metadata['features']:
        for ident, raw in zip(ids, development[item['id']]):
            value = None if pd.isna(raw) else float(raw)
            if value is not None and not math.isfinite(value):
                raise ValueError('Non-finite prepared feature.')
            feature_rows.append((ident, item['id'], item['unit'], value))
    coverage_rows = []
    for ident, row in selected_coverage.iterrows():
        for channel in CHANNELS:
            count = float(row[channel])
            if not math.isfinite(count) or not count.is_integer() or not 0 <= count <= window_days:
                raise ValueError('Coverage counts must be whole days inside the window.')
            coverage_rows.append((ident, channel, int(count), window_days))

    output = data_dir / 'sql'
    output.mkdir(parents=True, exist_ok=True)
    # In-memory construction avoids partially replacing an existing database.
    with closing(sqlite3.connect(':memory:')) as db:
        db.execute('PRAGMA foreign_keys = ON')
        db.executescript('''
            CREATE TABLE participants (student_id TEXT PRIMARY KEY);
            CREATE TABLE feature_values (
                student_id TEXT REFERENCES participants(student_id),
                feature TEXT NOT NULL, unit TEXT NOT NULL, value REAL,
                PRIMARY KEY (student_id, feature));
            CREATE TABLE coverage (
                student_id TEXT REFERENCES participants(student_id),
                channel TEXT NOT NULL, observed_days INTEGER NOT NULL,
                window_days INTEGER NOT NULL,
                PRIMARY KEY (student_id, channel),
                CHECK (observed_days BETWEEN 0 AND window_days));
        ''')
        db.executemany('INSERT INTO participants VALUES (?)', [(i,) for i in ids])
        db.executemany('INSERT INTO feature_values VALUES (?, ?, ?, ?)', feature_rows)
        db.executemany('INSERT INTO coverage VALUES (?, ?, ?, ?)', coverage_rows)
        db.commit()
        sql = '\n'.join(line for line in sql_path.read_text(encoding='utf-8').splitlines()
                        if not line.lstrip().startswith('--'))
        queries = [q.strip() for q in sql.split(';') if q.strip()]
        for number, query in enumerate(queries, 1):
            pd.read_sql_query(query, db).to_csv(output / f'query_{number}.csv', index=False)
        with closing(sqlite3.connect(output / 'analysis.sqlite')) as destination:
            db.backup(destination)
    summary = {
        'scope': 'Development participants only; no holdout values or questionnaire scores stored.',
        'development_participants': len(ids), 'reserved_participants': len(heldout_idx),
        'window_days': window_days, 'queries': len(queries),
        'limitations': [
            'Days without recorded events are not proven sensor failures or zero activity.',
            'Prepared missing values and raw observation gaps are different quantities.',
            'Descriptive statistics do not demonstrate causation or predictive performance.',
            'Database and query CSVs contain participant-level research data; keep local.'
        ]
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary, output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/private_studentlife')
    args = parser.parse_args()
    summary, output = analyze(args.data_dir)
    print('Development participants:', summary['development_participants'])
    print('Reserved participants:', summary['reserved_participants'])
    print('SQL queries completed:', summary['queries'])
    print('Local output:', output)
    print('Keep the database and query CSV files out of Git.')
