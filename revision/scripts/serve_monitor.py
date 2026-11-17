"""Loopback-only, read-only monitor with an explicit public-file allowlist."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import os
ROOT=Path(__file__).resolve().parents[1]/'live'
ALLOWED={'index.html','status.json','usage.json','usage.js','replay.json',
         'passage_mechanism_v2.png','rolling_execution_v2.png','frame.png',
         'paper_preview.pdf','focused_preview.pdf','mujoco_qualification.png','cad16_scene_preview.png','cad16_platform_preview.png',
         'bending_envelope_check.png','model_checkpoint.pdf','model_paper_page.pdf'}
class Handler(SimpleHTTPRequestHandler):
    def allowed(self):
        name=urlsplit(self.path).path.lstrip('/') or 'index.html'
        return name in ALLOWED and (ROOT/name).is_file()
    def do_GET(self):
        if not self.allowed():return self.send_error(404)
        super().do_GET()
    def do_HEAD(self):
        if not self.allowed():return self.send_error(404)
        super().do_HEAD()
    def end_headers(self):
        self.send_header('Cache-Control','no-cache')
        self.send_header('X-Content-Type-Options','nosniff')
        super().end_headers()
os.chdir(ROOT)
ThreadingHTTPServer(('127.0.0.1',8767),Handler).serve_forever()
