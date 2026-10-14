"""Step 5: repeatable, scoped privacy checks; no external security certification."""
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import unittest
from campuspulse.privacy import code_fingerprint

LABELS={
    'test_shap_explanation_requires_no_network_connection':'Calculate prepared SHAP explanations without a network connection',
    'test_prediction_requires_no_network_connection':'Calculate predictions without a network connection',
    'test_opt_out_never_calls_llm':'Require opt-in before contacting the AI service',
    'test_offline_retrieval_requires_no_network_connection':'Retrieve authored suggestions without a network connection',
    'test_handlers_do_not_open_files_for_writing_or_log_inputs':'Handle tested scenarios without Python file writes or request logs',
    'test_llm_payload_omits_identifiers_values_and_scores':'Exclude identifiers, exact values and scores from the AI payload',
    'test_local_llm_bypasses_environment_proxies':'Bypass configured proxies for the local AI request',
    'test_local_llm_redirect_is_rejected':'Reject redirects from the local AI request',
    'test_remote_llm_target_is_rejected':'Reject external AI addresses',
    'test_no_store_and_browser_connection_policy_and_host_check':'Use no-store responses and reject unexpected hostnames',
    'test_frontend_has_no_persistent_scenario_storage_or_default_consent':'Keep default consent off and avoid persistent browser storage'}

class Results(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.checks=[]
    def addSuccess(self,test):super().addSuccess(test);self.checks.append({'check':test._testMethodName,'status':'passed'})
    def addFailure(self,test,err):super().addFailure(test,err);self.checks.append({'check':test._testMethodName,'status':'failed'})
    def addError(self,test,err):super().addError(test,err);self.checks.append({'check':test._testMethodName,'status':'error'})
    def addSkip(self,test,reason):super().addSkip(test,reason);self.checks.append({'check':test._testMethodName,'status':'skipped'})

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('artifacts/privacy'));args=p.parse_args()
    root=Path(__file__).resolve().parent
    suite=unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_privacy.py')
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=Results).run(suite)
    for check in result.checks:check['label']=LABELS.get(check['check'],check['check'])
    report={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'code_fingerprint':code_fingerprint(root),
        'scope':'Automated checks of the supplied application code and tested paths. Not a device-wide audit.',
        'passed':result.wasSuccessful() and result.testsRun>0 and not result.skipped,
        'tests_run':result.testsRun,'checks':result.checks,
        'limits':['File-write checks observe Python file-open APIs in tested prediction/support handlers, not all operating-system writes.',
            'LLM transport tests use mocks; a live Ollama service and its downstream network/logging are not audited.',
            'Static frontend checks and response headers are tested; browser extension behavior, caches outside app control and memory erasure are not certified.',
            'This report applies to the fingerprinted build; run again after code changes.']}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (args.output/'checks.txt').write_text(stream.getvalue(),encoding='utf-8')
    print(stream.getvalue())
    print('Privacy verification:', 'PASSED' if report['passed'] else 'FAILED', '—', args.output)
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
