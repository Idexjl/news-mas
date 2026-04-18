"""
Encrypted local file storage — repository pattern over Fernet-encrypted JSON files.

Each collection maps to a subdirectory under DATA_DIR. Every record is a separate
.enc file: DATA_DIR/<collection>/<id>.enc.  All bytes written to disk are Fernet
(AES-128-CBC + HMAC-SHA256) encrypted — plaintext never touches the filesystem.

Environment variables (can also be passed as constructor arguments):
  FERNET_KEY   — URL-safe base64-encoded 32-byte key (generate with scripts/generate_key.py)
  DATA_DIR     — root directory for all collections (default: ./data/)

Usage:
    storage = FernetStorage()               # reads FERNET_KEY + DATA_DIR from env
    storage.save("runs", "run-001", {...})
    data = storage.load("runs", "run-001")  # raises KeyError if missing
    ids  = storage.list("runs")             # ["run-001", ...]
    storage.delete("runs", "run-001")       # raises KeyError if missing
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet


# ── core storage ──────────────────────────────────────────────────────────────

class FernetStorage:
    """Fernet-encrypted, filesystem-backed key-value store organised into collections."""

    def __init__(
        self,
        *,
        data_dir: str | Path | None = None,
        fernet_key: str | bytes | None = None,
    ) -> None:
        raw_key = fernet_key or os.environ.get("FERNET_KEY", "")
        if not raw_key:
            raise RuntimeError(
                "FERNET_KEY is not set. "
                "Generate one with: python scripts/generate_key.py"
            )
        key_bytes = raw_key if isinstance(raw_key, bytes) else raw_key.encode()
        self._fernet = Fernet(key_bytes)
        self._root = Path(data_dir or os.environ.get("DATA_DIR", "./data"))

    # ── internal ──────────────────────────────────────────────────────────────

    def _path(self, collection: str, id: str) -> Path:
        return self._root / collection / f"{id}.enc"

    # ── public API ────────────────────────────────────────────────────────────

    def save(self, collection: str, id: str, data: dict[str, Any]) -> None:
        """Encrypt *data* and write it to <collection>/<id>.enc.  Creates directories."""
        path = self._path(collection, id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self._fernet.encrypt(json.dumps(data).encode()))

    def load(self, collection: str, id: str) -> dict[str, Any]:
        """Return the decrypted record. Raises KeyError if it does not exist."""
        path = self._path(collection, id)
        if not path.exists():
            raise KeyError(f"{collection!r}/{id!r} not found in storage")
        return json.loads(self._fernet.decrypt(path.read_bytes()))

    def list(self, collection: str) -> list[str]:
        """Return all record IDs in *collection*, sorted. Empty list if none exist."""
        collection_dir = self._root / collection
        if not collection_dir.exists():
            return []
        return sorted(p.stem for p in collection_dir.glob("*.enc"))

    def delete(self, collection: str, id: str) -> None:
        """Delete a record. Raises KeyError if it does not exist."""
        path = self._path(collection, id)
        if not path.exists():
            raise KeyError(f"{collection!r}/{id!r} not found in storage")
        path.unlink()


# ── collection constants ───────────────────────────────────────────────────────

_USERS = "users"
_RUNS = "runs"
_DIGESTS = "digests"
_FEEDBACK = "feedback"


# ── repositories ──────────────────────────────────────────────────────────────

class UserRepository:
    """Stores user profiles and topic preferences."""

    def __init__(self, storage: FernetStorage) -> None:
        self._store = storage

    def save(self, user_id: str, profile: dict[str, Any]) -> None:
        self._store.save(_USERS, user_id, profile)

    def load(self, user_id: str) -> dict[str, Any]:
        return self._store.load(_USERS, user_id)

    def list_users(self) -> list[str]:
        return self._store.list(_USERS)

    def delete(self, user_id: str) -> None:
        self._store.delete(_USERS, user_id)


class RunRepository:
    """Stores run state and history (maps to RunState schema)."""

    def __init__(self, storage: FernetStorage) -> None:
        self._store = storage

    def save(self, run_id: str, state: dict[str, Any]) -> None:
        self._store.save(_RUNS, run_id, state)

    def load(self, run_id: str) -> dict[str, Any]:
        return self._store.load(_RUNS, run_id)

    def list_runs(self) -> list[str]:
        return self._store.list(_RUNS)

    def delete(self, run_id: str) -> None:
        self._store.delete(_RUNS, run_id)


class DigestRepository:
    """Stores final digest output (list of DigestEntry records) keyed by run_id."""

    def __init__(self, storage: FernetStorage) -> None:
        self._store = storage

    def save(self, run_id: str, digest: dict[str, Any]) -> None:
        self._store.save(_DIGESTS, run_id, digest)

    def load(self, run_id: str) -> dict[str, Any]:
        return self._store.load(_DIGESTS, run_id)

    def list_digests(self) -> list[str]:
        return self._store.list(_DIGESTS)

    def delete(self, run_id: str) -> None:
        self._store.delete(_DIGESTS, run_id)


class FeedbackRepository:
    """
    Stores user feedback signals.

    Feedback types (stored as a 'type' field in the dict):
      summary_feedback     — thumbs up/down + free-text note, keyed on run_id+article_url
      topic_relevance      — per-topic relevance score over time
      missed_topic         — topics flagged as missing from a digest
      candidate_override   — articles manually added or removed from a digest
    """

    def __init__(self, storage: FernetStorage) -> None:
        self._store = storage

    def save(self, feedback_id: str, feedback: dict[str, Any]) -> None:
        self._store.save(_FEEDBACK, feedback_id, feedback)

    def load(self, feedback_id: str) -> dict[str, Any]:
        return self._store.load(_FEEDBACK, feedback_id)

    def list_feedback(self) -> list[str]:
        return self._store.list(_FEEDBACK)

    def delete(self, feedback_id: str) -> None:
        self._store.delete(_FEEDBACK, feedback_id)
