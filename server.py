
"""Local, single-user demonstration server. Do not expose it to the public internet."""
import json
import argparse
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import joblib
from campuspulse.model import score
from campuspulse.coach import support
from campuspulse.store import demo_summary
from campuspulse.privacy import privacy_info, code_fingerprint

ROOT = Path(__file__).resolve().parent

def make_handler(bundle, model_dir=None):
    eda_dir = Path(model_dir) / 'eda' if model_dir else None
    candidate_dir = Path(model_dir) / 'features_v2' if model_dir else None
    shap_dir = Path(model_dir) / 'shap' if model_dir else None
    source_fingerprint = bundle['report'].get('original_dataset_sha256', bundle['report'].get('dataset_sha256'))

    def candidates():
        if candidate_dir is None or not (candidate_dir / 'summary.json').exists():
            return None
        summary = json.loads((candidate_dir / 'summary.json').read_text(encoding='utf-8'))
        if summary.get('original_dataset_sha256') != source_fingerprint:
            return None
        return summary

    def exploration():
        if eda_dir is None or not (eda_dir / 'summary.json').exists():
            return None
        summary = json.loads((eda_dir / 'summary.json').read_text(encoding='utf-8'))
        if summary.get('input_sha256') != source_fingerprint:
            return None  # Never display an analysis from a different training dataset.
        return summary

    class Handler(BaseHTTPRequestHandler):
        def valid_host(self):
            if self.server.server_address[0] not in {'127.0.0.1','::1'}:
                return True  # Explicit non-loopback binding is reported, not mislabeled local-only.
            try:
                host = urlsplit('http://' + self.headers.get('Host',''))
                return host.hostname in {'127.0.0.1','localhost','::1'} and host.port in {None,self.server.server_port}
            except ValueError:
                return False

        def log_message(self, *_):
            pass  # No behavior inputs or request paths logged.

        def send_bytes(self, content, content_type, status=200):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
            self.end_headers()
            self.wfile.write(content)

        def json(self, data, status=200):
            self.send_bytes(json.dumps(data, allow_nan=False).encode(), 'application/json', status)

        def do_GET(self):
            if not self.valid_host():
                return self.json({'error':'Unexpected host for a local application.'},403)
            path = urlsplit(self.path).path
            if path == '/':
                self.send_bytes((ROOT / 'web/index.html').read_bytes(), 'text/html; charset=utf-8')
            elif path == '/api/health':
                self.json({'status': 'ok', 'source': bundle['report']['source']})
            elif path == '/api/metrics':
                self.json(bundle['report'])
            elif path == '/api/privacy':
                info = privacy_info(self.server.server_address[0])
                report_path = ROOT / 'artifacts/privacy/report.json'
                if report_path.exists():
                    info['verification'] = json.loads(report_path.read_text(encoding='utf-8'))
                    info['verification']['matches_current_code'] = info['verification'].get('code_fingerprint') == code_fingerprint(ROOT)
                self.json(info)
            elif path == '/api/cohort':
                self.json({'source': 'synthetic', 'rows': demo_summary()})
            elif path == '/api/exploration':
                summary = exploration()
                self.json({'available': summary is not None, 'summary': summary})
            elif path == '/api/engineered-features':
                summary = candidates()
                self.json({'available': summary is not None, 'summary': summary})
            elif path == '/features/candidates.png':
                image_path = candidate_dir / 'candidates.png' if candidate_dir else None
                if candidates() is None or not image_path.is_file():
                    return self.json({'error':'Run engineer_studentlife.py first.'},404)
                self.send_bytes(image_path.read_bytes(),'image/png')
            elif path == '/shap/global.png':
                image_path = shap_dir / 'global.png' if shap_dir else None
