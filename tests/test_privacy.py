import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import socket
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch
import numpy as np
import pandas as pd
from http.server import ThreadingHTTPServer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from campuspulse.studentlife import SCHEMA
from campuspulse.model import score
from campuspulse.coach import support
from campuspulse.privacy import LOCAL_CHAT_URL, NoLocalRedirect, open_local_llm
from server import make_handler, ROOT

class PrivacyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(18)
        names=[f['id'] for f in SCHEMA]
        frame=pd.DataFrame({f['id']:rng.uniform(f['min'],f['max'],12) for f in SCHEMA})
        model=make_pipeline(StandardScaler(),Ridge()).fit(frame,rng.uniform(25,60,12))
        cls.bundle={'model':model,'medians':frame.median().to_dict(),
            'report':{'source':'privacy_fixture','features':names,'feature_schema':SCHEMA,'warning':'Fixture'}}
        cls.values={f['id']:float(frame[f['id']].median()) for f in SCHEMA}
        cls.frame=frame
        score(cls.bundle,cls.values)  # Warm imports before observing handler file activity.
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(cls.bundle))
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.server.server_port}'
        cls.client=urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()

    def post(self,path,payload):
        req=urllib.request.Request(self.url+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with self.client.open(req) as response:return json.load(response)

    def test_prediction_requires_no_network_connection(self):
        # Direct inference must not contact even a localhost service.
        with patch('socket.socket.connect',side_effect=AssertionError('Unexpected connection')):
            result=score(self.bundle,self.values)
        self.assertIn('estimate',result)

    @unittest.skipUnless(importlib.util.find_spec('shap'),'Install requirements-shap.txt to check SHAP privacy')
    def test_shap_explanation_requires_no_network_connection(self):
        from campuspulse.explain import prepare
        bundle=copy.deepcopy(self.bundle)
        prepare(bundle,self.frame)
        with patch('socket.socket.connect',side_effect=AssertionError('Unexpected connection')):
            result=score(bundle,self.values)
        self.assertTrue(result['shap']['available'])

    def test_opt_out_never_calls_llm(self):
        with patch.dict(os.environ,{'OLLAMA_MODEL':'fixture-local'}), patch('campuspulse.coach.open_local_llm') as transport:
            result=self.post('/api/support',{**self.values,'consent':False})
        transport.assert_not_called();self.assertEqual(result['mode'],'disabled')

    def test_offline_retrieval_requires_no_network_connection(self):
        with patch.dict(os.environ,{'OLLAMA_MODEL':''}), patch('socket.socket.connect',side_effect=AssertionError('Unexpected connection')):
            result=support(self.values,True,SCHEMA)
        self.assertEqual(result['mode'],'retrieval')

    def test_handlers_do_not_open_files_for_writing_or_log_inputs(self):
        original_open,original_io_open=open,io.open
        writes=[]
        def guard(original):
            def wrapped(file,mode='r',*args,**kwargs):
                if isinstance(mode,str) and any(c in mode for c in 'wax+'):
                    writes.append(str(file));raise AssertionError('Unexpected scenario file write')
                return original(file,mode,*args,**kwargs)
            return wrapped
        logs=io.StringIO()
        with patch.dict(os.environ,{'OLLAMA_MODEL':''}), contextlib.redirect_stderr(logs), \
                patch('builtins.open',side_effect=guard(original_open)),patch('io.open',side_effect=guard(original_io_open)):
            self.post('/api/predict',self.values)
            self.post('/api/support',{**self.values,'consent':True})
        self.assertEqual(writes,[]);self.assertEqual(logs.getvalue(),'')

    def test_llm_payload_omits_identifiers_values_and_scores(self):
        payload={**self.values,'conversation_event_minutes':123.456,
                 'student_id':'NEVER_SEND_THIS_ID','prediction':77.123}
        body=json.dumps({'message':{'content':'If you would like, send a friend a short message.'}}).encode()
        with patch.dict(os.environ,{'OLLAMA_MODEL':'fixture-local'}), patch('campuspulse.coach.open_local_llm',return_value=io.BytesIO(body)) as transport:
            result=support(payload,True,SCHEMA)
        self.assertEqual(result['mode'],'local_llm')
        request=transport.call_args.args[0]
        self.assertEqual(request.full_url,LOCAL_CHAT_URL)
        sent=json.loads(request.data)
        content=sent['messages'][1]['content']
        self.assertEqual(set(json.loads(content)),{'topics','cards'})
        for secret in ['NEVER_SEND_THIS_ID','123.456','77.123',*self.values]:self.assertNotIn(secret,content)

    def test_local_llm_bypasses_environment_proxies(self):
        req=urllib.request.Request(LOCAL_CHAT_URL,data=b'{}',headers={'Content-Type':'application/json'})
        with patch.dict(os.environ,{'HTTP_PROXY':'http://example.invalid:8080'}), patch('campuspulse.privacy.urllib.request.build_opener') as builder:
            open_local_llm(req)
        self.assertEqual(builder.call_args.args[0].proxies,{})
        self.assertIsInstance(builder.call_args.args[1],NoLocalRedirect)

    def test_local_llm_redirect_is_rejected(self):
        req=urllib.request.Request(LOCAL_CHAT_URL,data=b'{}')
        with self.assertRaises(urllib.error.HTTPError):
            NoLocalRedirect().redirect_request(req,None,302,'Moved',{},'https://example.invalid/collect')

    def test_remote_llm_target_is_rejected(self):
        req=urllib.request.Request('https://example.invalid/api/chat',data=b'{}')
        with patch('campuspulse.privacy.urllib.request.build_opener') as builder,self.assertRaises(ValueError):
            open_local_llm(req)
        builder.assert_not_called()

    def test_no_store_and_browser_connection_policy_and_host_check(self):
        with self.client.open(self.url+'/api/health') as response:
            self.assertEqual(response.headers['Cache-Control'],'no-store')
            self.assertIn("connect-src 'self'",response.headers['Content-Security-Policy'])
            self.assertEqual(response.headers['Referrer-Policy'],'no-referrer')
        request=urllib.request.Request(self.url+'/api/health',headers={'Host':'example.invalid'})
        with self.assertRaises(urllib.error.HTTPError) as error:self.client.open(request)
        self.assertEqual(error.exception.code,403)

    def test_frontend_has_no_persistent_scenario_storage_or_default_consent(self):
        html=(ROOT/'web/index.html').read_text()
        script=html.split('<script>')[1].split('</script>')[0]
        self.assertIsNone(re.search(r'(?:localStorage|sessionStorage)\s*\.|document\s*\.\s*cookie',script))
        checkbox=re.search(r'<input[^>]*id="consent"[^>]*>',html).group()
        self.assertNotIn('checked',checkbox)
