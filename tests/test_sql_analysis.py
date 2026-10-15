import json
from pathlib import Path
import sqlite3
from contextlib import closing
import tempfile
import unittest

import pandas as pd
from campuspulse.model import partition
from analyze_studentlife_sql import analyze


class SqlAnalysisTests(unittest.TestCase):
    def fixture(self, root):
        df = pd.DataFrame({'student_id': [f'u{i:02}' for i in range(25)],
                           'sleep': [None if i % 3 == 0 else 7.0 for i in range(25)]})
        df.to_csv(root / 'features.csv', index=False)
        pd.DataFrame({'student_id': df.student_id, 'sleep_days': 10,
                      'conversation_days': 20, 'locked_days': 28}).to_csv(root / 'coverage.csv', index=False)
        (root / 'metadata.json').write_text(json.dumps({
            'source': 'studentlife_real', 'feature_window': ['2013-03-27', '2013-04-24'],
            'features': [{'id': 'sleep', 'unit': 'hours'}]}))
        return df

    def test_sql_counts_missingness_and_holdout_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            df = self.fixture(root)
            dev, held = partition(df)
            summary, out = analyze(root)
            self.assertEqual(summary['queries'], 4)
            with closing(sqlite3.connect(out / 'analysis.sqlite')) as db:
                ids = {r[0] for r in db.execute('SELECT student_id FROM participants')}
                self.assertEqual(ids, set(df.iloc[dev].student_id))
                self.assertFalse(ids & set(df.iloc[held].student_id))
                self.assertEqual(db.execute('SELECT COUNT(*) FROM feature_values WHERE value IS NULL').fetchone()[0], int(df.iloc[dev].sleep.isna().sum()))
            before = [(out / f'query_{i}.csv').read_bytes() for i in range(1, 5)]
            df.loc[held, 'sleep'] = 99999
            df.to_csv(root / 'features.csv', index=False)
            analyze(root)
            self.assertEqual(before, [(out / f'query_{i}.csv').read_bytes() for i in range(1, 5)])

    def test_duplicate_coverage_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            coverage = pd.read_csv(root / 'coverage.csv')
            pd.concat([coverage, coverage.iloc[:1]]).to_csv(root / 'coverage.csv', index=False)
            with self.assertRaises(ValueError):
                analyze(root)
            self.assertFalse((root / 'sql').exists())
