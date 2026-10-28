"""Chat web server using only the Python standard library.

Run:   python -m intyoza_srh                 (http://localhost:8000)
       python -m intyoza_srh --port 9000 --host 0.0.0.0

Endpoints:
  GET  /             chat page (web/index.html)
  POST /api/chat     {"message": "..."} or {"title": "..."} (suggestion chip),
                     optional "conversation_id"  -> reply JSON (+ "conversation_id" when history is on)
  GET  /api/topics   {topic: [titles]}
  GET  /api/health   {"status": "ok", ...}
  GET  /api/conversations          this browser's saved chats   (header X-Client-Id: <uuid>)
  GET  /api/conversations/<id>     messages of one chat         (header X-Client-Id)
  DELETE /api/conversations/<id>   delete one chat              (header X-Client-Id)

Chat history: when DATABASE (a PostgreSQL URL) is set in the environment or a .env file, each question and
reply is saved under a random id the browser makes (no names, no accounts), so chats survive a refresh.
Without DATABASE nothing is written anywhere. Requests are never logged.
Set INTYOZA_CORS_ORIGIN (e.g. https://your-site.rw) to call the API from another web app."""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .chatbot import IntyozaChatbot
from .history import ChatHistory, as_uuid

WEB = Path(__file__).resolve().parent / "web"
PUBLIC_KEYS = ("type", "intent", "title", "topic", "answer", "disclaimer", "suggestions", "source", "reviewed",
               "confidence", "best_guess")


def make_handler(bot, history=None):
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
                self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Client-Id")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self._send(204, b"")

        def _client(self):
            return as_uuid(self.headers.get("X-Client-Id"))

        def _conversation_route(self, path):
            """(client_id, conversation_id) for /api/conversations/<id>, or None after sending an error."""
            if history is None:
                return self._send(404, {"error": "chat history is off"})
            client, conv = self._client(), as_uuid(path.rsplit("/", 1)[-1])
            if not client or not conv:
                return self._send(400, {"error": "bad client or conversation id"})
            return client, conv

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, page, "text/html")
            elif path == "/api/topics":
                self._send(200, bot.topics())
            elif path == "/api/health":
                self._send(200, {"status": "ok", "backend": bot.backend, "answers": len(bot.kb),
                                 "history": history is not None})
            elif path == "/api/conversations":
                if history is None:
                    return self._send(200, {"history": False, "conversations": []})
                client = self._client()
                if not client:
                    return self._send(400, {"error": "bad client id"})
                self._send(200, {"history": True, "conversations": history.list_conversations(client)})
            elif path.startswith("/api/conversations/"):
                ids = self._conversation_route(path)
                if ids:
                    msgs = history.get_messages(*ids)
                    if msgs is None:
                        return self._send(404, {"error": "not found"})
                    self._send(200, {"messages": msgs})
            else:
                self._send(404, {"error": "not found"})

        def do_DELETE(self):
            path = self.path.split("?")[0]
            if not path.startswith("/api/conversations/"):
                return self._send(404, {"error": "not found"})
            ids = self._conversation_route(path)
            if ids:
                self._send(200, {"deleted": history.delete_conversation(*ids)})

        def do_POST(self):
            if self.path.split("?")[0] != "/api/chat":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > 4000:
                    return self._send(413, {"error": "message too long"})
                body = json.loads(self.rfile.read(n) or b"{}")
                msg, title = str(body.get("message", ""))[:500], str(body.get("title", ""))[:200]
                conv = as_uuid(body.get("conversation_id"))
            except (ValueError, TypeError):
                return self._send(400, {"error": "send JSON like {\"message\": \"your question\"}"})
            if not msg.strip() and not title.strip():
                return self._send(400, {"error": "empty message"})
            r = bot.answer_by_title(title) if title.strip() else bot.ask(msg)
            out = {k: r[k] for k in PUBLIC_KEYS if k in r}               # never echo the question back
            client = self._client()
            if history is not None and client:
                try:
                    out["conversation_id"] = str(history.save_turn(client, conv, (title or msg).strip(), out))
                except Exception:                                      # a database hiccup must not lose the answer
                    pass
            self._send(200, out)

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description="Intyoza SRH chatbot server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=None, help="path to intyoza_srh_model.h5")
    ap.add_argument("--backend", default="auto", choices=["auto", "keras", "numpy"])
    a = ap.parse_args(argv)
    bot = IntyozaChatbot(a.model, backend=a.backend)
    history = ChatHistory.from_env()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(bot, history))
    print(f"Intyoza is running on http://{a.host}:{a.port}  ({len(bot.kb)} answers, backend={bot.backend}, "
          f"history={'on' if history else 'off'}). Ctrl+C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
