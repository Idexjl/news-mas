"""
Topic subscription repository — encrypted storage for user topic subscriptions.

TopicSubscription records drive the scheduler: each active subscription
is fetched at run time and converted to a TopicInput for the pipeline.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from src.common.storage import FernetStorage

_TOPICS = "topics"


class TopicSubscription(BaseModel):
    topic_id: str
    user_id: str
    topic_text: str  # search query — never logged or recorded in spans
    constraints: list[str] = Field(default_factory=list)
    active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_run_id: Optional[str] = None
    last_heat_score: Optional[float] = None


class TopicRepository:
    """Fernet-backed repository for TopicSubscription records."""

    def __init__(self, storage: FernetStorage) -> None:
        self._store = storage

    def save_topic(self, topic: TopicSubscription) -> None:
        self._store.save(_TOPICS, topic.topic_id, topic.model_dump(mode="json"))

    def get_topic(self, topic_id: str) -> TopicSubscription:
        data = self._store.load(_TOPICS, topic_id)
        return TopicSubscription(**data)

    def get_all_topics(self) -> list[TopicSubscription]:
        """Return all active topic subscriptions, sorted by topic_id."""
        topics: list[TopicSubscription] = []
        for tid in self._store.list(_TOPICS):
            try:
                t = self.get_topic(tid)
                if t.active:
                    topics.append(t)
            except Exception:
                pass
        return topics

    def delete_topic(self, topic_id: str) -> None:
        self._store.delete(_TOPICS, topic_id)

    def update_topic(self, topic_id: str, updates: dict) -> None:
        topic = self.get_topic(topic_id)
        data = topic.model_dump(mode="json")
        data.update(updates)
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._store.save(_TOPICS, topic_id, data)
