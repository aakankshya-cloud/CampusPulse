import unittest
import pandas as pd
from campuspulse.explore import analyze
from campuspulse.model import partition

class ExplorationTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'student_id':[str(i) for i in range(25)],
            'fixture_feature':range(25), 'loneliness_score':[20+i for i in range(25)]})
        self.meta = {'source':'fixture', 'target_bounds':[20,80],
            'features':[{'id':'fixture_feature','label':'Fixture','unit':'units'}]}

    def test_holdout_values_cannot_change_analysis(self):
        a, _, _ = analyze(self.df, self.meta)
        _, test_indices = partition(self.df)
        changed = self.df.copy()
        changed.loc[test_indices, 'fixture_feature'] = -99999
        changed.loc[test_indices, 'loneliness_score'] = 99999
        b, _, _ = analyze(changed, self.meta)
        self.assertEqual(a['features'],b['features'])
        self.assertEqual(a['development_students'],20)
        self.assertEqual(a['reserved_students'],5)

    def test_undefined_correlation_is_reported_not_fabricated(self):
        self.df['fixture_feature'] = 1
        summary, _, _ = analyze(self.df, self.meta)
        self.assertIsNone(summary['features'][0]['spearman_r'])

    def test_duplicate_person_rows_rejected(self):
        with self.assertRaises(ValueError):
            analyze(pd.concat([self.df,self.df.iloc[:1]]), self.meta)
