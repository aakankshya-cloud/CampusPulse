import unittest
from unittest.mock import patch
from campuspulse.knowledge import retrieve
from campuspulse.coach import support
from campuspulse.data import FEATURES


class QueryScopeTests(unittest.TestCase):
    def test_sleep_inflection_matches_routine(self):
        self.assertEqual(retrieve('sleeping late every evening')[0]['topic'], 'routine')

    def test_phone_focus_prefers_distraction_card(self):
        self.assertEqual(retrieve('phone distracts me when studying')[0]['id'], 'balance_notifications')

    def test_bedtime_phone_keeps_sleep_context(self):
        self.assertEqual(retrieve('phone before bed')[0]['topic'], 'routine')

    def test_unsupported_requests_never_call_llm(self):
        example=dict(zip(FEATURES,[7,30,50,4,4,1]))
        with patch('campuspulse.coach.open_local_llm') as transport:
            for q in ['set a reminder for my coursework', 'make a revision plan',
                      'help me remember chemistry', 'public speaking tomorrow']:
                self.assertEqual(retrieve(q), [])
                result=support({**example,'support_query':q},True)
                self.assertEqual(result['mode'],'out_of_scope')
                self.assertEqual(result['sources'],[])
            transport.assert_not_called()

    def test_consent_precedes_scope_checks(self):
        self.assertEqual(support({'support_query':'remind me tomorrow'},False)['mode'],'disabled')
