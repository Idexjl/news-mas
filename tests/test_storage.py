"""
Tests for src.common.storage — FernetStorage and the four repository classes.

All tests use tmp_path for the data directory so nothing persists between runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from src.common.storage import (
    DigestRepository,
    FeedbackRepository,
    FernetStorage,
    RunRepository,
    UserRepository,
)


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def key() -> str:
    return Fernet.generate_key().decode()


@pytest.fixture()
def storage(tmp_path: Path, key: str) -> FernetStorage:
    return FernetStorage(data_dir=tmp_path, fernet_key=key)


# ── FernetStorage — encrypt / decrypt roundtrip ───────────────────────────────

def test_save_and_load_roundtrip(storage: FernetStorage) -> None:
    original = {"run_id": "r1", "status": "completed", "score": 0.87}
    storage.save("runs", "r1", original)
    assert storage.load("runs", "r1") == original


def test_roundtrip_preserves_nested_structures(storage: FernetStorage) -> None:
    data = {"articles": [{"title": "A", "score": 1.0}, {"title": "B", "score": 0.5}]}
    storage.save("digests", "d1", data)
    assert storage.load("digests", "d1") == data


def test_roundtrip_preserves_all_json_types(storage: FernetStorage) -> None:
    data = {"str": "hello", "int": 42, "float": 3.14, "bool": True, "null": None, "list": [1, 2]}
    storage.save("misc", "m1", data)
    assert storage.load("misc", "m1") == data


def test_data_is_encrypted_on_disk(storage: FernetStorage, tmp_path: Path) -> None:
    secret = "do not store me in plaintext"
    storage.save("runs", "enc-test", {"secret": secret})
    raw_bytes = (tmp_path / "runs" / "enc-test.enc").read_bytes()
    assert secret.encode() not in raw_bytes


def test_encrypted_file_is_not_valid_json_on_disk(storage: FernetStorage, tmp_path: Path) -> None:
    storage.save("runs", "json-test", {"key": "value"})
    raw = (tmp_path / "runs" / "json-test.enc").read_bytes()
    with pytest.raises(Exception):  # noqa: BLE001
        json.loads(raw)


def test_overwrite_replaces_existing_record(storage: FernetStorage) -> None:
    storage.save("runs", "r1", {"status": "pending"})
    storage.save("runs", "r1", {"status": "completed"})
    assert storage.load("runs", "r1") == {"status": "completed"}


# ── FernetStorage — load / delete error paths ─────────────────────────────────

def test_load_missing_raises_key_error(storage: FernetStorage) -> None:
    with pytest.raises(KeyError, match=r"not found"):
        storage.load("runs", "does-not-exist")


def test_delete_missing_raises_key_error(storage: FernetStorage) -> None:
    with pytest.raises(KeyError, match=r"not found"):
        storage.delete("runs", "does-not-exist")


def test_load_after_delete_raises_key_error(storage: FernetStorage) -> None:
    storage.save("runs", "r1", {"x": 1})
    storage.delete("runs", "r1")
    with pytest.raises(KeyError):
        storage.load("runs", "r1")


# ── FernetStorage — list ───────────────────────────────────────────────────────

def test_list_empty_collection_returns_empty_list(storage: FernetStorage) -> None:
    assert storage.list("runs") == []


def test_list_nonexistent_collection_returns_empty_list(storage: FernetStorage) -> None:
    assert storage.list("nonexistent") == []


def test_list_returns_all_saved_ids(storage: FernetStorage) -> None:
    for i in ("b", "a", "c"):
        storage.save("runs", i, {"id": i})
    assert storage.list("runs") == ["a", "b", "c"]  # sorted


def test_list_excludes_deleted_ids(storage: FernetStorage) -> None:
    storage.save("runs", "keep", {})
    storage.save("runs", "drop", {})
    storage.delete("runs", "drop")
    assert storage.list("runs") == ["keep"]


# ── FernetStorage — collection isolation ──────────────────────────────────────

def test_collection_isolation(storage: FernetStorage) -> None:
    storage.save("users", "u1", {"name": "Alice"})
    assert storage.list("runs") == []
    with pytest.raises(KeyError):
        storage.load("runs", "u1")


def test_same_id_in_different_collections_is_independent(storage: FernetStorage) -> None:
    storage.save("users", "id1", {"type": "user"})
    storage.save("runs", "id1", {"type": "run"})
    assert storage.load("users", "id1")["type"] == "user"
    assert storage.load("runs", "id1")["type"] == "run"


def test_delete_in_one_collection_does_not_affect_another(storage: FernetStorage) -> None:
    storage.save("users", "id1", {"name": "Alice"})
    storage.save("runs", "id1", {"status": "done"})
    storage.delete("runs", "id1")
    assert storage.load("users", "id1") == {"name": "Alice"}


# ── FernetStorage — missing key ───────────────────────────────────────────────

def test_missing_fernet_key_raises_runtime_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FERNET_KEY", raising=False)
    with pytest.raises(RuntimeError, match=r"FERNET_KEY"):
        FernetStorage(data_dir=tmp_path)


def test_explicit_key_takes_precedence_over_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FERNET_KEY", raising=False)
    key = Fernet.generate_key().decode()
    store = FernetStorage(data_dir=tmp_path, fernet_key=key)  # should not raise
    store.save("x", "1", {"ok": True})
    assert store.load("x", "1") == {"ok": True}


def test_key_from_env_is_used_when_no_explicit_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("FERNET_KEY", key)
    store = FernetStorage(data_dir=tmp_path)
    store.save("x", "1", {"env": True})
    assert store.load("x", "1") == {"env": True}


def test_wrong_key_raises_on_decrypt(tmp_path: Path) -> None:
    key_a = Fernet.generate_key().decode()
    key_b = Fernet.generate_key().decode()
    store_a = FernetStorage(data_dir=tmp_path, fernet_key=key_a)
    store_b = FernetStorage(data_dir=tmp_path, fernet_key=key_b)
    store_a.save("runs", "r1", {"secret": "value"})
    with pytest.raises(Exception):  # cryptography.fernet.InvalidToken  # noqa: BLE001
        store_b.load("runs", "r1")


# ── UserRepository ─────────────────────────────────────────────────────────────

@pytest.fixture()
def users(storage: FernetStorage) -> UserRepository:
    return UserRepository(storage)


def test_user_save_and_load(users: UserRepository) -> None:
    profile = {"name": "Alice", "topics": ["AI", "policy"]}
    users.save("u1", profile)
    assert users.load("u1") == profile


def test_user_list(users: UserRepository) -> None:
    users.save("u2", {})
    users.save("u1", {})
    assert users.list_users() == ["u1", "u2"]


def test_user_delete(users: UserRepository) -> None:
    users.save("u1", {"name": "Alice"})
    users.delete("u1")
    with pytest.raises(KeyError):
        users.load("u1")


def test_user_load_missing_raises(users: UserRepository) -> None:
    with pytest.raises(KeyError):
        users.load("nobody")


# ── RunRepository ──────────────────────────────────────────────────────────────

@pytest.fixture()
def runs(storage: FernetStorage) -> RunRepository:
    return RunRepository(storage)


def test_run_save_and_load(runs: RunRepository) -> None:
    state = {"run_id": "r1", "status": "completed", "topics": ["AI"]}
    runs.save("r1", state)
    assert runs.load("r1") == state


def test_run_list(runs: RunRepository) -> None:
    runs.save("r2", {})
    runs.save("r1", {})
    assert runs.list_runs() == ["r1", "r2"]


def test_run_delete(runs: RunRepository) -> None:
    runs.save("r1", {})
    runs.delete("r1")
    with pytest.raises(KeyError):
        runs.load("r1")


# ── DigestRepository ───────────────────────────────────────────────────────────

@pytest.fixture()
def digests(storage: FernetStorage) -> DigestRepository:
    return DigestRepository(storage)


def test_digest_save_and_load(digests: DigestRepository) -> None:
    digest = {"run_id": "r1", "entries": [{"title": "Headline"}]}
    digests.save("r1", digest)
    assert digests.load("r1") == digest


def test_digest_list(digests: DigestRepository) -> None:
    digests.save("r1", {})
    digests.save("r2", {})
    assert digests.list_digests() == ["r1", "r2"]


def test_digest_delete(digests: DigestRepository) -> None:
    digests.save("r1", {})
    digests.delete("r1")
    with pytest.raises(KeyError):
        digests.load("r1")


# ── FeedbackRepository ─────────────────────────────────────────────────────────

@pytest.fixture()
def feedback(storage: FernetStorage) -> FeedbackRepository:
    return FeedbackRepository(storage)


def test_feedback_save_and_load(feedback: FeedbackRepository) -> None:
    entry = {"type": "summary_feedback", "run_id": "r1", "rating": 1, "note": "Good"}
    feedback.save("fb1", entry)
    assert feedback.load("fb1") == entry


def test_feedback_list(feedback: FeedbackRepository) -> None:
    feedback.save("fb1", {"type": "missed_topic"})
    feedback.save("fb2", {"type": "topic_relevance"})
    assert feedback.list_feedback() == ["fb1", "fb2"]


def test_feedback_delete(feedback: FeedbackRepository) -> None:
    feedback.save("fb1", {})
    feedback.delete("fb1")
    with pytest.raises(KeyError):
        feedback.load("fb1")


def test_feedback_collection_isolated_from_runs(
    storage: FernetStorage,
) -> None:
    fb = FeedbackRepository(storage)
    rr = RunRepository(storage)
    fb.save("shared-id", {"type": "missed_topic"})
    with pytest.raises(KeyError):
        rr.load("shared-id")
