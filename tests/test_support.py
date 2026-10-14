"""Behavioral tests for retrieval and constrained LLM selection."""
import io
import json
import os
import unittest
from unittest.mock import patch
from campuspulse.coach import FOCUS, support
from campuspulse.knowledge import BY_ID
from campuspulse.data import FEATURES
EXAMPLE=dict(zip(FEATURES,[7,30,50,4,4,1]))


def reply(text):
    return io.BytesIO(json.dumps({'message':{'content':text}}).encode())


class SupportTests(unittest.TestCase):
    def test_explicit_focus_retrieves_matching_card(self):
        with patch.dict(os.environ,{'OLLAMA_MODEL':''}):
            for focus in FOCUS.keys()-{'general'}:
                r=support({**EXAMPLE,'weekly_focus':focus},True)
                self.assertEqual(r['sources'][0]['id'],focus)
                self.assertEqual(r['message'],BY_ID[focus]['text'])

    def test_score_does_not_select_suggestion(self):
        with patch.dict(os.environ,{'OLLAMA_MODEL':''}):
            a=support({**EXAMPLE,'weekly_focus':'movement','prediction':20},True)
            b=support({**EXAMPLE,'weekly_focus':'movement','prediction':80},True)
        self.assertEqual(a,b)

    def test_invalid_focus_rejected_before_transport(self):
        with patch('campuspulse.coach.open_local_llm') as transport:
            for focus in [[],{},True,None,'invalid']:
                with self.assertRaises(ValueError):support({**EXAMPLE,'weekly_focus':focus},True)
            transport.assert_not_called()

    def test_opt_out_ignores_even_invalid_inputs(self):
        with patch('campuspulse.coach.open_local_llm') as transport:
            self.assertEqual(support({'weekly_focus':[]},False)['mode'],'disabled')
            transport.assert_not_called()

    def test_query_and_identifiers_stay_out_of_llm_payload(self):
        text=BY_ID['balance_notifications']['text']
        raw_query='Notifications interrupt studying MY_PRIVATE_NOTE'
        with patch.dict(os.environ,{'OLLAMA_MODEL':'test'}),patch('campuspulse.coach.open_local_llm',return_value=reply(text)) as transport:
            r=support({**EXAMPLE,'weekly_focus':'balance','support_query':raw_query,'student_id':'SECRET'},True)
        self.assertEqual(r['mode'],'local_llm')
        sent=json.loads(transport.call_args.args[0].data)
        content=json.loads(sent['messages'][1]['content'])
        self.assertEqual(set(content),{'topics','cards'})
        self.assertNotIn('MY_PRIVATE_NOTE',str(sent));self.assertNotIn('SECRET',str(sent))
        self.assertEqual(r['selected_card_id'],'balance_notifications')

    def test_new_claims_and_sentence_fragments_fall_back(self):
        for text in ['If you would like, enhance restful sleep.',
                     'If you would like, a consistent wind-down time.',
                     'If you would like, you have a disorder.']:
            with patch.dict(os.environ,{'OLLAMA_MODEL':'test'}),patch('campuspulse.coach.open_local_llm',return_value=reply(text)):
                r=support({**EXAMPLE,'weekly_focus':'routine'},True)
            self.assertEqual(r['mode'],'retrieval_fallback')
            self.assertEqual(r['fallback_reason'],'unapproved_text')
            self.assertEqual(r['message'],BY_ID['routine']['text'])

    def test_movement_assertions_and_lost_conditions_rejected(self):
        for text in ['If you would like, consider a short walk.',
                     'If you would like, a short walk would be comfortable and accessible.']:
            with patch.dict(os.environ,{'OLLAMA_MODEL':'test'}),patch('campuspulse.coach.open_local_llm',return_value=reply(text)):
                r=support({**EXAMPLE,'weekly_focus':'movement'},True)
            self.assertEqual(r['mode'],'retrieval_fallback')
            self.assertEqual(r['message'],BY_ID['movement']['text'])

    def test_approved_accessibility_variant_is_accepted(self):
        text='If you would like, and it feels comfortable and accessible, consider a short walk or another gentle activity you enjoy.'
        with patch.dict(os.environ,{'OLLAMA_MODEL':'test'}),patch('campuspulse.coach.open_local_llm',return_value=reply(text)):
            r=support({**EXAMPLE,'weekly_focus':'movement'},True)
        self.assertEqual(r['mode'],'local_llm');self.assertEqual(r['message'],text)

    def test_authored_if_you_want_opening_is_accepted(self):
        text=BY_ID['balance']['text']
        with patch.dict(os.environ,{'OLLAMA_MODEL':'test'}),patch('campuspulse.coach.open_local_llm',return_value=reply(text)):
            r=support({**EXAMPLE,'weekly_focus':'balance'},True)
        self.assertEqual(r['mode'],'local_llm')

    def test_scope_boundary_and_no_match_never_call_llm(self):
        with patch('campuspulse.coach.open_local_llm') as transport:
            for query,mode in [('medication dosage','out_of_scope'),('zxqvvj','no_match')]:
                r=support({**EXAMPLE,'support_query':query},True)
                self.assertEqual(r['mode'],mode)
            transport.assert_not_called()

    def test_query_limits_and_canonical_output(self):
        for query in [None,[],{},'a'*241]:
            with self.assertRaises(ValueError):support({**EXAMPLE,'support_query':query},True)
        text=BY_ID['connection']['text']
        with patch.dict(os.environ,{'OLLAMA_MODEL':'test'}),patch('campuspulse.coach.open_local_llm',return_value=reply(text.upper())):
            r=support({**EXAMPLE,'weekly_focus':'connection'},True)
        self.assertEqual(r['message'],text)
