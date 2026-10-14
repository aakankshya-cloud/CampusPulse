"""Live CampusPulse smoke evaluation: four fixed topics, repeated requests.
Run while the app is running. Saves example-only replies, never real user data.
This is not a clinical benchmark or automatic proof of relevance/safety.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request

BASE = 'http://127.0.0.1:8000'
FOCUSES = ('connection', 'routine', 'movement', 'balance')

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, 'Redirect refused', headers, fp)


def api(path, payload=None):
    request = urllib.request.Request(BASE + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type':'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=45) as response:
        return json.load(response)


def evaluate(repeats, request=api):
    metrics = request('/api/metrics')
    schema = metrics.get('feature_schema')
    if not isinstance(schema, list) or not schema:
        raise ValueError('Start the StudentLife app: no real-data feature schema was returned.')
    defaults = {f['id']: f['default'] for f in schema}
    rows = []
    for attempt in range(1, repeats + 1):
        for focus in FOCUSES:
            started = time.perf_counter()
            row = {'focus':focus, 'attempt':attempt}
            try:
                result = request('/api/support', {**defaults, 'weekly_focus':focus, 'consent':True})
                mode = result.get('mode', 'unknown')
                sources = result.get('sources', [])
                message = result.get('message', '')
                row.update(mode=mode, response=message,
                    source_ids=[s.get('id') for s in sources],
                    expected_source_retrieved=bool(sources and sources[0].get('id') == focus),
                    word_count=len(message.split()),
                    exactly_matches_authored_card=any(message == s.get('text') for s in sources),
                    human_review={'topic_relevant':None, 'grounded_in_card':None,
                                  'no_unsupported_health_claims':None, 'notes':''})
            except Exception as error:
                row.update(mode='request_error', error=str(error))
            row['elapsed_seconds'] = round(time.perf_counter() - started, 3)
            rows.append(row)
            print(f"{focus:12} run {attempt}: {row['mode']:20} {row['elapsed_seconds']}s", flush=True)
    try:
        disabled = request('/api/support', {**defaults, 'weekly_focus':'connection', 'consent':False})
        consent = {'observed_mode':disabled.get('mode'), 'passed':disabled.get('mode')=='disabled',
                   'scope':'HTTP response only; no transport-call observation in this live run.'}
    except Exception as error:
        consent = {'passed':False, 'error':str(error)}
    counts = dict(Counter(r['mode'] for r in rows))
    llm_times = [r['elapsed_seconds'] for r in rows if r['mode']=='local_llm']
    fingerprint_files = ['campuspulse/coach.py','campuspulse/privacy.py','server.py']
    root = Path(__file__).resolve().parent
    fingerprints = {n:hashlib.sha256((root/n).read_bytes()).hexdigest()
                    for n in fingerprint_files if (root/n).is_file()}
    return {'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'evaluation_kind':'live_fixed_topic_smoke_test', 'source':metrics.get('source'),
        'app_code_sha256':fingerprints, 'total_requests':len(rows), 'mode_counts':counts,
        'local_llm_rate':counts.get('local_llm',0)/len(rows),
        'local_llm_median_seconds':statistics.median(llm_times) if llm_times else None,
        'retrieval_match_count':sum(r.get('expected_source_retrieved',False) for r in rows),
        'consent_opt_out':consent, 'cases':rows,
        'limitations':[
            'Four fixed topic choices repeated with example median inputs; not independent varied scenarios.',
            'Mode measures accepted generation, not quality, safety or clinical benefit.',
            'Matching source ID tests retrieval; human review is required for generated-text relevance and grounding.',
            'Latency includes HTTP, model loading and generation; first calls can be slower.',
            'The same prompt/model is used throughout. This does not demonstrate prompt optimization.',
            'Model identity and weights are not independently verified by this script.',
            'Reports intentionally save these fictional test replies; interactive user scenarios remain separate.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error('--repeats must be between 1 and 10')
    try:
        report = evaluate(args.repeats)
    except Exception as error:
        raise SystemExit(f'Could not start evaluation: {error}\nKeep CampusPulse running at {BASE}.')
    directory = Path(__file__).resolve().parent/'artifacts'/'llm_evaluation'
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = directory/f'report_{stamp}.json'
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nSUMMARY')
    print('Requests:', report['total_requests'])
    print('Modes:', report['mode_counts'])
    print(f"Accepted LLM responses: {report['local_llm_rate']:.0%}")
    print('Median accepted-LLM response time (seconds):', report['local_llm_median_seconds'])
    print('Matching retrieved sources:', report['retrieval_match_count'])
    print('Opt-out response check:', 'PASS' if report['consent_opt_out']['passed'] else 'FAIL')
    print('Human response-quality review: PENDING')
    print('Saved:', output)

if __name__ == '__main__':
    main()
