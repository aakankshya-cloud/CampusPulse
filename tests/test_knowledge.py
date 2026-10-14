import json
from pathlib import Path
import socket
import unittest
from unittest.mock import patch
from campuspulse.knowledge import CARDS, BY_ID, approved_wordings, retrieve

class KnowledgeTests(unittest.TestCase):
    def test_unique_curated_records_and_sources(self):
        self.assertEqual(len(CARDS),20)
        self.assertEqual(len(BY_ID),20)
        for c in CARDS:
            self.assertTrue(c['source_url'].startswith(('https://www.nhs.uk/','https://lsc.cornell.edu/')))
            self.assertTrue(c['source_accessed'])
            self.assertIn('not a quotation',c['provenance'])

    def test_local_search_needs_no_network(self):
        with patch.object(socket.socket,'connect',side_effect=AssertionError('network forbidden')):
            self.assertTrue(retrieve('video call family'))

    def test_topic_filter_never_returns_other_topics(self):
        matches=retrieve('phone notifications study bedtime',topic='routine',limit=5)
        self.assertTrue(matches)
        self.assertTrue(all(c['topic']=='routine' for c in matches))

    def test_zero_match_is_not_fabricated_relevance(self):
        self.assertEqual(retrieve('zxqvvj'),[])

    def test_narrow_actions_have_distinct_candidates(self):
        self.assertEqual(retrieve('video call family')[0]['id'],'connection_video')
        self.assertEqual(retrieve('gardening plants')[0]['id'],'movement_garden')

    def test_movement_variants_keep_conditional_accessibility(self):
        for c in CARDS:
            for text in approved_wordings(c):
                if c['topic']=='movement':
                    self.assertIn('comfortable',text);self.assertIn('accessible',text)
                    self.assertTrue(text.startswith(('If comfortable and accessible,','If you would like, and it feels comfortable and accessible,')))

    def test_locked_case_labels_reference_real_cards(self):
        cases=json.loads((Path(__file__).resolve().parents[1]/'configs/retrieval_cases.json').read_text())
        self.assertEqual(len(cases),20)
        for c in cases:self.assertTrue(set(c['expected_ids'])<=set(BY_ID))

    def test_invalid_search_length_and_type(self):
        for query in [None,[], 'a'*241]:
            with self.assertRaises(ValueError):retrieve(query)
