import unittest
import numpy as np
import pandas as pd
from campuspulse.studentlife import loneliness_scores, decode_sleep_hours, interval_daily, REVERSE
from campuspulse.contracts import validate_schema_input
from prepare_globem import prepare as prepare_globem

class StudentLifeTests(unittest.TestCase):
    def test_scoring_direction_and_q4(self):
        row = {'uid': 'fixture', 'type': 'post', **{f'Q{i}': 'Often' if i in REVERSE else 'Never' for i in range(1,21)}}
        scores, missing = loneliness_scores(pd.DataFrame([row]))
        self.assertEqual(scores.loneliness_score.iloc[0], 20)
        self.assertEqual(missing, 0)
        row['Q4'] = 'Never'
        scores, _ = loneliness_scores(pd.DataFrame([row]))
        self.assertEqual(scores.loneliness_score.iloc[0], 23)

    def test_incomplete_outcome_is_dropped(self):
        row = {'uid': 'fixture', 'type': 'post', **{f'Q{i}': 'Never' for i in range(1,21)}}
        row['Q5'] = None
        scores, missing = loneliness_scores(pd.DataFrame([row]))
        self.assertTrue(scores.empty)
        self.assertEqual(missing, 1)

    def test_sleep_codes_are_not_literal_hours(self):
        values = decode_sleep_hours(pd.Series([0, 1, 2, 3, 19, 20]))
        self.assertTrue(values.iloc[:2].isna().all())
        self.assertEqual(values.iloc[2], 3.5)
        self.assertEqual(values.iloc[3], 4)
        self.assertEqual(values.iloc[4], 12)
        self.assertTrue(np.isnan(values.iloc[5]))

    def test_overlap_midnight_and_window_clip(self):
        start = pd.Timestamp('2013-03-27', tz='America/New_York')
        end = pd.Timestamp('2013-03-29', tz='America/New_York')
        first = pd.Timestamp('2013-03-27 23:30', tz='America/New_York').timestamp()
        df = pd.DataFrame({'uid': ['fixture']*3,
            'start_timestamp': [first, first+900, end.timestamp()+1],
            'end_timestamp': [first+3600, first+2700, end.timestamp()+100]})
        daily = interval_daily(df, start, end)
        self.assertEqual(len(daily), 2)
        self.assertEqual(daily.minutes.sum(), 60)
        self.assertEqual(daily.minutes.tolist(), [30, 30])

    def test_real_input_contract(self):
        schema = [{'id':'example','min':0,'max':10}]
        self.assertEqual(validate_schema_input({'example':3},schema),{'example':3})
        with self.assertRaises(ValueError):
            validate_schema_input({'example':True},schema)

class GlobemFixtureTests(unittest.TestCase):
    def setUp(self):
        self.column = 'f_screen:phone_screen_rapids_sumdurationunlock:allday'
        self.features = pd.DataFrame({'pid':['fixture']*30,
            'date':pd.date_range('2020-01-04', periods=30),
            self.column:[2]*28+[99999,99999]})
        self.post = pd.DataFrame({'pid':['fixture'], 'date':['2020-02-01'], 'UCLA_POST':[25]})

    def test_no_survey_day_or_future_features(self):
        df, meta = prepare_globem(self.features, self.post, [self.column], 'UCLA_POST')
        self.assertEqual(df[self.column].iloc[0], 2)
        self.assertEqual(meta['target_bounds'], [10,40])
        self.assertEqual(df.loneliness_score.iloc[0], 25)

    def test_normalized_features_rejected(self):
        bad = self.column.replace('unlock:', 'unlock_norm:')
        with self.assertRaisesRegex(ValueError, 'normalized'):
            prepare_globem(self.features, self.post, [bad], 'UCLA_POST')

    def test_wrong_target_scale_rejected(self):
        self.post['UCLA_POST'] = 65
        with self.assertRaisesRegex(ValueError, '10–40'):
            prepare_globem(self.features, self.post, [self.column], 'UCLA_POST')

    def test_duplicate_days_rejected(self):
        with self.assertRaisesRegex(ValueError, 'one feature row'):
            prepare_globem(pd.concat([self.features,self.features.iloc[:1]]),
                           self.post, [self.column], 'UCLA_POST')

    def test_depression_target_rejected(self):
        with self.assertRaisesRegex(ValueError, 'UCLA'):
            prepare_globem(self.features, self.post, [self.column], 'dep')
