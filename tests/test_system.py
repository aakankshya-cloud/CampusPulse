import json
import io
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import patch
import joblib
from campuspulse.data import FEATURES, synthetic_data, validate_input, validate_training
from campuspulse.model import partition, score, train
from campuspulse.coach import support
from campuspulse.store import demo_summary
from server import make_handler

EXAMPLE = dict(zip(FEATURES, [7, 30, 50, 4, 4, 1]))

class DataTests(unittest.TestCase):
    def test_participant_disjointness(self):
        df = synthetic_data(students=30)
        a, b = partition(df)
        self.assertFalse(set(df.iloc[a].student_id) & set(df.iloc[b].student_id))

    def test_reproducibility(self):
        self.assertTrue(synthetic_data().equals(synthetic_data()))

    def test_invalid_numeric_and_missing_inputs(self):
        for bad in [float('nan'), float('inf'), -1, 'oops', True, None]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_input({**EXAMPLE, 'sleep_hours': bad})
        with self.assertRaises(ValueError):
            validate_input({})

    def test_invalid_training_targets(self):
        df = synthetic_data(students=25)
        df.loc[0, 'loneliness_score'] = float('nan')
        with self.assertRaises(ValueError):
            validate_training(df)

    def test_opt_out_does_not_call_llm(self):
        with patch('campuspulse.coach.open_local_llm') as request:
            self.assertEqual(support(EXAMPLE, False)['mode'], 'disabled')
            request.assert_not_called()

    def test_offline_retrieval(self):
        with patch.dict(os.environ, {'OLLAMA_MODEL': ''}):
            result = support({**EXAMPLE, 'social_minutes': 0}, True)
        self.assertEqual(result['mode'], 'retrieval')
        self.assertEqual(result['sources'][0]['id'], 'connection')

    def test_llm_failure_falls_back(self):
        with patch.dict(os.environ, {'OLLAMA_MODEL': 'test'}), patch(
                'campuspulse.coach.open_local_llm', side_effect=OSError):
            result = support(EXAMPLE, True)
        self.assertEqual(result['mode'], 'retrieval_fallback')

    def test_llm_valid_response_and_minimal_payload(self):
        body = json.dumps({'message': {'content': 'If you would like, send one friend a short message or join a campus activity you enjoy.'}}).encode()
        with patch.dict(os.environ, {'OLLAMA_MODEL': 'test'}), patch(
                'campuspulse.coach.open_local_llm', return_value=io.BytesIO(body)) as request:
            result = support(EXAMPLE, True)
        self.assertEqual(result['mode'], 'local_llm')
        sent = json.loads(request.call_args.args[0].data)
        self.assertNotIn('sleep_hours', sent['messages'][1]['content'])
        self.assertNotIn('loneliness_score', sent['messages'][1]['content'])

    def test_llm_prohibited_output_is_filtered(self):
        body = json.dumps({'message': {'content': 'You have a disorder and need medication.'}}).encode()
        with patch.dict(os.environ, {'OLLAMA_MODEL': 'test'}), patch(
                'campuspulse.coach.open_local_llm', return_value=io.BytesIO(body)):
            result = support(EXAMPLE, True)
        self.assertEqual(result['mode'], 'retrieval_fallback')

    def test_sql_demo_counts(self):
        rows = demo_summary()
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]['students'], 180)

class ModelAndAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        df = synthetic_data(students=30, weeks=3)
        df.loc[0, 'sleep_hours'] = float('nan')
        cls.report = train(df, cls.temp.name)
        cls.bundle = joblib.load(os.path.join(cls.temp.name, 'model.joblib'))
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(cls.bundle))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.temp.cleanup()

    def test_training_and_loaded_inference(self):
        self.assertEqual(self.report['student_overlap'], 0)
        self.assertEqual(len(self.report['comparison']), 3)
        self.assertEqual(self.report['source'], 'synthetic')
        result = score(self.bundle, EXAMPLE)
        self.assertTrue(0 < result['estimate'] < 100)
        self.assertEqual(len(result['explanations']), 6)

    def test_api_predict(self):
        request = urllib.request.Request(self.url + '/api/predict',
            data=json.dumps(EXAMPLE).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request) as response:
            data = json.load(response)
        self.assertIn('estimate', data)

    def test_api_rejects_invalid_values(self):
        request = urllib.request.Request(self.url + '/api/predict', data=b'{}',
            headers={'Content-Type': 'application/json'})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 400)

    def test_api_rejects_cross_origin(self):
        request = urllib.request.Request(self.url + '/api/predict', data=b'{}',
            headers={'Content-Type': 'application/json', 'Origin': 'https://example.com'})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 403)

    def test_page_and_missing_route(self):
        with urllib.request.urlopen(self.url) as response:
            self.assertIn(b'CampusPulse', response.read())
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(self.url + '/missing')
        self.assertEqual(error.exception.code, 404)

if __name__ == '__main__':
    unittest.main()
