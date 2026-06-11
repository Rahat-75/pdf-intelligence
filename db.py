"""Simple SQLite storage for chat history and indexed PDFs (multi-PDF per chat supported)."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "chat.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                pdf_id TEXT,
                pdf_name TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                pdf_id TEXT,
                pdf_name TEXT,
                sources_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id);
            """
        )

        # --- Schema migrations for existing old chat.db files ---
        # Older versions of the app created the tables without the per-message pdf_id/pdf_name columns.
        _ensure_column(conn, "conversations", "pdf_id", "TEXT")
        _ensure_column(conn, "conversations", "pdf_name", "TEXT")
        _ensure_column(conn, "messages", "pdf_id", "TEXT")
        _ensure_column(conn, "messages", "pdf_name", "TEXT")


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, col_type: str) -> None:
    """Add a column if it is missing (safe ALTER for schema upgrades)."""
    try:
        cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    except sqlite3.OperationalError:
        # Table doesn't exist yet or other transient case — ignore.
        pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_sources(sources: list[Any] | None) -> str | None:
    if not sources:
        return None
    payload = []
    for doc in sources:
        if hasattr(doc, "page_content"):
            payload.append(
                {
                    "page": doc.metadata.get("page"),
                    "content": doc.page_content[:2000],
                }
            )
        elif isinstance(doc, dict):
            payload.append(
                {
                    "page": doc.get("metadata", {}).get("page"),
                    "content": doc.get("page_content", "")[:2000],
                }
            )
    return json.dumps(payload)


def _deserialize_sources(raw: str | None) -> list[dict]:
    if not raw:
        return []
    items = json.loads(raw)
    return [
        {"metadata": {"page": item.get("page", "?")}, "page_content": item.get("content", "")}
        for item in items
    ]


# --- Documents (PDF library) ---

def upsert_document(doc_id: str, filename: str, chunk_count: int) -> None:
    now = _now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO documents (id, filename, chunk_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                filename = excluded.filename,
                chunk_count = excluded.chunk_count,
                updated_at = excluded.updated_at
            """,
            (doc_id, filename, chunk_count, now, now),
        )


def list_documents() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, filename, chunk_count, created_at, updated_at
            FROM documents
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_document(doc_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, filename, chunk_count FROM documents WHERE id = ?",
            (doc_id,),
        ).fetchone()
    return dict(row) if row else None


def delete_document(doc_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


# --- Conversations ---

def create_conversation(
    title: str,
    pdf_id: str | None = None,
    pdf_name: str | None = None,
) -> int:
    now = _now()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO conversations (title, pdf_id, pdf_name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (title, pdf_id, pdf_name, now, now),
        )
        return int(cur.lastrowid)


def _truncate_title(text: str, max_len: int = 48) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def update_conversation_title(conversation_id: int, title: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now(), conversation_id),
        )


def get_first_user_message(conversation_id: int) -> str | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT content FROM messages
            WHERE conversation_id = ? AND role = 'user'
            ORDER BY id ASC
            LIMIT 1
            """,
            (conversation_id,),
        ).fetchone()
    return row["content"] if row else None


def get_display_title(conversation_id: int, title: str) -> str:
    """Return a sidebar label; backfill legacy 'New chat' rows from the first user message."""
    if title != "New chat":
        return title
    first = get_first_user_message(conversation_id)
    if not first:
        return title
    resolved = _truncate_title(first)
    update_conversation_title(conversation_id, resolved)
    return resolved


# --- Messages (support per-message PDF target) ---

def add_message(
    conversation_id: int,
    role: str,
    content: str,
    pdf_id: str | None = None,
    pdf_name: str | None = None,
    sources: list[Any] | None = None,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO messages (
                conversation_id, role, content, pdf_id, pdf_name, sources_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (conversation_id, role, content, pdf_id, pdf_name, _serialize_sources(sources), _now()),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (_now(), conversation_id),
        )


def list_conversations(limit: int = 20) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, title, pdf_id, pdf_name, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def load_messages(conversation_id: int) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT role, content, pdf_id, pdf_name, sources_json
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()

    messages = []
    for row in rows:
        msg = {
            "role": row["role"],
            "content": row["content"],
            "pdf_id": row["pdf_id"],
            "pdf_name": row["pdf_name"],
        }
        sources = _deserialize_sources(row["sources_json"])
        if sources:
            msg["sources"] = sources
        messages.append(msg)
    return messages


def delete_conversation(conversation_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


def get_conversation(conversation_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, title, pdf_id, pdf_name FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
    return dict(row) if row else None
