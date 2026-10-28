"""Chat web server using only the Python standard library.

Run:   python -m intyoza_srh                 (http://localhost:8000)
       python -m intyoza_srh --port 9000 --host 0.0.0.0

Endpoints:
  GET  /             chat page (web/index.html)
  POST /api/chat     {"message": "..."} or {"title": "..."} (suggestion chip)  -> reply JSON
  GET  /api/topics   {topic: [titles]}
  GET  /api/health   {"status": "ok", ...}

Privacy by design: no database, no log of questions, nothing written to disk.
Set INTYOZA_CORS_ORIGIN (e.g. https://your-site.rw) to call the API from another web app."""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .chatbot import IntyozaChatbot

WEB = Path(__file__).resolve().parent / "web"
PUBLIC_KEYS = ("type", "intent", "title", "topic", "answer", "disclaimer", "suggestions", "source", "reviewed",
               "confidence", "best_guess")


def make_handler(bot):
    cors = os.getenv("INTYOZA_CORS_ORIGIN", "")
    page = (WEB / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "Intyoza"

        def log_message(self, *args):          # never log questions
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype + ("; charset=utf-8" if "json" in ctype or "html" in ctype else ""))
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if cors:
                self.send_header("Access-Control-Allow-Origin", cors)
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self._send(204, b"")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, page, "text/html")
            elif path == "/api/topics":
                self._send(200, bot.topics())
            elif path == "/api/health":
                self._send(200, {"status": "ok", "backend": bot.backend, "answers": len(bot.kb)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.split("?")[0] != "/api/chat":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > 4000:
                    return self._send(413, {"error": "message too long"})
                body = json.loads(self.rfile.read(n) or b"{}")
                msg, title = str(body.get("message", ""))[:500], str(body.get("title", ""))[:200]
            except (ValueError, TypeError):
                return self._send(400, {"error": "send JSON like {\"message\": \"your question\"}"})
            if not msg.strip() and not title.strip():
                return self._send(400, {"error": "empty message"})
            r = bot.answer_by_title(title) if title.strip() else bot.ask(msg)
            self._send(200, {k: r[k] for k in PUBLIC_KEYS if k in r})   # never echo the question back

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description="Intyoza SRH chatbot server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=None, help="path to intyoza_srh_model.h5")
    ap.add_argument("--backend", default="auto", choices=["auto", "keras", "numpy"])
    a = ap.parse_args(argv)
    bot = IntyozaChatbot(a.model, backend=a.backend)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(bot))
    print(f"Intyoza is running on http://{a.host}:{a.port}  ({len(bot.kb)} answers, backend={bot.backend}). Ctrl+C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
