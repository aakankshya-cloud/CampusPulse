"""Small, explicit local transport boundary and a factual privacy description."""
import os
import hashlib
from pathlib import Path
import urllib.error
import urllib.request

LOCAL_CHAT_URL = 'http://127.0.0.1:11434/api/chat'

def code_fingerprint(root):
    digest=hashlib.sha256()
    for name in ['server.py','web/index.html','campuspulse/privacy.py','campuspulse/coach.py','campuspulse/knowledge.py','configs/support_cards.json',
                 'campuspulse/model.py','campuspulse/explain.py','campuspulse/contracts.py',
                 'campuspulse/data.py','campuspulse/store.py','tests/test_privacy.py','tests/test_support.py']:
        digest.update(name.encode());digest.update((Path(root)/name).read_bytes())
    return digest.hexdigest()

class NoLocalRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, 'Local LLM redirects are disabled', headers, fp)

def open_local_llm(request, timeout=20):
    if request.full_url != LOCAL_CHAT_URL or request.get_method() != 'POST':
        raise ValueError('Only the fixed local chat endpoint is allowed.')
    # Never let HTTP_PROXY/HTTPS_PROXY send this request through another server.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoLocalRedirect())
    return opener.open(request, timeout=timeout)

def privacy_info(bind_address):
    return {
        'bind_address':bind_address,
        'loopback_binding':bind_address in {'127.0.0.1','::1'},
        'support_mode':'Local endpoint configured' if os.getenv('OLLAMA_MODEL','').strip() else 'Offline retrieval',
        'facts':[
            {'title':'Predictions stay in this app', 'text':'ML prediction, feature sensitivity and SHAP calculation run in the Python process on this computer.'},
            {'title':'Slider scenarios are not saved', 'text':'Tested prediction/support handlers do not write entered scenarios to files or a database, and request logging is disabled.'},
            {'title':'Suggestions start switched off', 'text':'A support request needs explicit opt-in. Without consent, the LLM transport is not called.'},
            {'title':'Optional LLM uses a fixed local address', 'text':'The app contacts only 127.0.0.1:11434 for generation, bypasses proxy settings and rejects redirects. The optional search is used in local retrieval. It sends broad topic names and approved card wording, not exact slider values, participant IDs or predicted scores.'},
            {'title':'The browser uses this app’s origin', 'text':'No scenario cookies, localStorage or sessionStorage are used. Responses request no-store caching; the page’s connection policy allows its own origin only.'},
            {'title':'Research files are different from entered scenarios', 'text':'Downloaded study records, prepared features, trained models and generated reports remain in local project folders. They are not erased when you clear the scenario.'}],
        'limits':[
            'These checks cover this application and tested paths, not browser extensions, operating-system memory/swap, malware or other programs.',
            'Ollama is a separate service: its logs, model configuration and downstream network behavior are not audited here. A localhost endpoint alone does not certify that service is offline.',
            'Dataset/package/model downloads require internet access. Inference and offline retrieval are separate from setup.',
            'This is a local-computer prototype, not a deployed phone app, federated-learning system, legal compliance certification or secure-erasure guarantee.',
            'Binding to 0.0.0.0 or another non-loopback address changes exposure. The runtime binding is shown on this page.']}
