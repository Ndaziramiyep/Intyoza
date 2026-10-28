"""Chat history in PostgreSQL, so a conversation survives a refresh or a return visit.

The connection string comes from DATABASE (or DATABASE_URL), read from the environment or from a .env file
in this package folder or the current folder. No names or accounts: each browser makes a random client id
and keeps it in localStorage; only that id can read or delete its chats."""
import json
import os
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          uuid PRIMARY KEY,
    client_id   uuid NOT NULL,
    title       text NOT NULL DEFAULT '',
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS conversations_client_idx ON conversations (client_id, updated_at DESC);
CREATE TABLE IF NOT EXISTS messages (
    id               bigserial PRIMARY KEY,
    conversation_id  uuid NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             text NOT NULL CHECK (role IN ('user', 'bot')),
    content          text NOT NULL,
    reply            jsonb,
    created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_conversation_idx ON messages (conversation_id, id);
"""


def load_env():
    """Read KEY=VALUE lines from .env files without overriding variables that are already set."""
    for path in (Path(__file__).resolve().parent / ".env", Path.cwd() / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def as_uuid(value):
    """Return a UUID, or None if value is not one (so bad ids never reach SQL)."""
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return None


class ChatHistory:
    def __init__(self, dsn):
        from psycopg_pool import ConnectionPool
        self.pool = ConnectionPool(dsn, min_size=1, max_size=5, open=True, timeout=10)
        with self.pool.connection() as c:
            c.execute(SCHEMA)

    @classmethod
    def from_env(cls):
        """ChatHistory if a database is configured and reachable, else None (the chat still works)."""
        load_env()
        dsn = os.getenv("DATABASE") or os.getenv("DATABASE_URL")
        if not dsn:
            return None
        try:
            return cls(dsn)
        except Exception as e:                       # driver missing or database down: run without history
            print(f"Chat history is off: {type(e).__name__}: {e}")
            return None

    def save_turn(self, client_id, conversation_id, user_text, reply):
        """Store one question + reply. Starts a new conversation when conversation_id is unknown.
        Returns the conversation id."""
        cid = conversation_id or uuid.uuid4()
        with self.pool.connection() as c:
            row = c.execute("SELECT client_id FROM conversations WHERE id = %s", (cid,)).fetchone()
            if row and row[0] != client_id:          # someone else's id: never write into it
                cid, row = uuid.uuid4(), None
            if row:
                c.execute("UPDATE conversations SET updated_at = now() WHERE id = %s", (cid,))
            else:
                c.execute("INSERT INTO conversations (id, client_id, title) VALUES (%s, %s, %s)",
                          (cid, client_id, user_text[:80]))
            c.execute("INSERT INTO messages (conversation_id, role, content) VALUES (%s, 'user', %s)",
                      (cid, user_text))
            c.execute("INSERT INTO messages (conversation_id, role, content, reply) VALUES (%s, 'bot', %s, %s)",
                      (cid, reply.get("answer", ""), json.dumps(reply)))
        return cid

    def list_conversations(self, client_id, limit=50):
        with self.pool.connection() as c:
            rows = c.execute("SELECT id, title, updated_at FROM conversations WHERE client_id = %s "
                             "ORDER BY updated_at DESC LIMIT %s", (client_id, limit)).fetchall()
        return [{"id": str(i), "title": t, "updated_at": u.isoformat()} for i, t, u in rows]

    def get_messages(self, client_id, conversation_id):
        """Messages of one conversation, or None if it does not exist for this client."""
        with self.pool.connection() as c:
            if not c.execute("SELECT 1 FROM conversations WHERE id = %s AND client_id = %s",
                             (conversation_id, client_id)).fetchone():
                return None
            rows = c.execute("SELECT role, content, reply FROM messages WHERE conversation_id = %s ORDER BY id",
                             (conversation_id,)).fetchall()
        return [{"role": r, "text": t} if r == "user" else {"role": r, "reply": rep} for r, t, rep in rows]

    def delete_conversation(self, client_id, conversation_id):
        with self.pool.connection() as c:
            n = c.execute("DELETE FROM conversations WHERE id = %s AND client_id = %s",
                          (conversation_id, client_id)).rowcount
        return n > 0
