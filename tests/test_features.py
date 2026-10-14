import unittest
import numpy as np
import pandas as pd
from campuspulse.features import half_change, daily_exercise, engineer
from campuspulse.studentlife import SCHEMA
from campuspulse.model import partition
from campuspulse.studentlife import interval_daily

class EngineeringTests(unittest.TestCase):
    def setUp(self):
        self.start = pd.Timestamp('2013-03-27', tz='America/New_York')
        self.end = pd.Timestamp('2013-04-24', tz='America/New_York')

    def test_change_direction_and_future_exclusion(self):
        data = pd.DataFrame({'student_id':['fixture']*7,
            'date':['2013-03-28','2013-03-29','2013-03-30',
                    '2013-04-15','2013-04-16','2013-04-17','2013-04-24'],
            'minutes':[100,100,100,60,60,60,99999]})
        self.assertEqual(half_change(data,'minutes',self.start,self.end)['fixture'],-40)

    def test_sparse_half_does_not_become_zero(self):
        data = pd.DataFrame({'student_id':['fixture']*4,
            'date':['2013-03-28','2013-04-15','2013-04-16','2013-04-17'],
            'hours':[8,6,6,6]})
        self.assertTrue(np.isnan(half_change(data,'hours',self.start,self.end)['fixture']))

    def test_night_intervals_count_only_midnight_to_six(self):
        a = pd.Timestamp('2013-03-28 23:00',tz='America/New_York').timestamp()
        b = pd.Timestamp('2013-03-29 08:00',tz='America/New_York').timestamp()
        data = pd.DataFrame({'uid':['fixture','fixture'],
            'start_timestamp':[a,a+7200], 'end_timestamp':[b,b-3600]})
        result = interval_daily(data,self.start,self.end,hour_window=(0,6))
        self.assertEqual(result.date.tolist(),['2013-03-29'])
        self.assertEqual(result.minutes.iloc[0],360)

    def test_exercise_decoding_conflicts_and_future(self):
        stamps = [pd.Timestamp(v,tz='America/New_York').timestamp() for v in
                  ['2013-03-28 12:00','2013-03-29 12:00','2013-03-30 12:00',
                   '2013-03-30 14:00','2013-03-31 12:00','2013-04-24 12:00']]
        data = pd.DataFrame({'uid':['fixture']*6,'timestamp':stamps,'have':[1,2,1,2,0,1]})
        result = daily_exercise(data,self.start,self.end)
        self.assertEqual(result.exercise.tolist(),[1,0])
        self.assertEqual(len(result),2)

    def test_reserved_participants_cannot_change_candidates(self):
        ids = [str(i) for i in range(25)]
        base = pd.DataFrame({'student_id':ids, 'reported_sleep_hours':6.,
            'sleep_duration_sd':.5, 'conversation_event_minutes':10.,
            'phone_locked_hours':1., 'loneliness_score':35.})
        metadata = {'feature_window':[self.start.isoformat(),self.end.isoformat()],
                    'limitations':[], 'features':SCHEMA}
        day_strings = ['2013-03-28','2013-03-29','2013-03-30','2013-03-31',
                       '2013-04-15','2013-04-16','2013-04-17','2013-04-18']
        records = [(uid,pd.Timestamp(day+' 12:00',tz='America/New_York').timestamp())
                   for uid in ids for day in day_strings]
        sleep = pd.DataFrame(records,columns=['uid','timestamp']);sleep['hour']=9
        exercise = sleep[['uid','timestamp']].copy();exercise['have']=1
        intervals = sleep[['uid','timestamp']].rename(columns={'timestamp':'start_timestamp'})
        intervals['end_timestamp'] = intervals.start_timestamp+3600
        original, _, summary, split = engineer(base,metadata,sleep,intervals,intervals,exercise)
        self.assertEqual(len(original),20)
        self.assertEqual(summary['candidate_feature_count'],11)
        # No observed night event must remain unknown, not be called zero use.
        self.assertTrue(original.night_phone_locked_hours.isna().all())
        _, reserved = partition(base)
        changed = base.copy();changed.loc[reserved,'loneliness_score']=999
        changed_intervals=intervals.copy()
        mask=changed_intervals.uid.isin(set(base.iloc[reserved].student_id))
        changed_intervals.loc[mask,'end_timestamp']+=10000000
        updated, _, _, _ = engineer(changed,metadata,sleep,changed_intervals,changed_intervals,exercise)
        pd.testing.assert_frame_equal(original,updated)
        self.assertFalse(set(split['development_ids']) & set(split['reserved_ids']))
