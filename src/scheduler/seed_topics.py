"""
Seed script — populates the topic repository with test subscriptions.

Run with:
    python -m src.scheduler.seed_topics

Requires FERNET_KEY (and optionally DATA_DIR) to be set in the environment
or in a .env file in the project root.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from src.common.storage import FernetStorage
from src.common.topic_repository import TopicRepository, TopicSubscription

SEED_TOPICS: list[TopicSubscription] = [
    TopicSubscription(
        topic_id="topic-001",
        user_id="user-joe",
        topic_text="Colorado Rockies baseball news",
        constraints=[],
    ),
    TopicSubscription(
        topic_id="topic-002",
        user_id="user-joe",
        topic_text="cybersecurity news high profile "
                   "breaches exploits vulnerabilities",
        constraints=[
            "technical reporting only",
            "no vendor marketing content"
        ],
    ),
    TopicSubscription(
        topic_id="topic-003",
        user_id="user-joe",
        topic_text="professional bicycle racing news "
                   "Tour de France UCI WorldTour",
        constraints=[],
    ),
    TopicSubscription(
        topic_id="topic-004",
        user_id="user-joe",
        topic_text="Warhammer 40000 tabletop gaming news "
                   "Games Workshop new releases lore",
        constraints=[],
    ),
    TopicSubscription(
        topic_id="topic-005",
        user_id="user-joe",
        topic_text="tabletop gaming news board games "
                   "RPG Dungeons Dragons releases",
        constraints=[],
    ),
]


def seed(*, dry_run: bool = False) -> None:
    storage = FernetStorage()
    repo = TopicRepository(storage)

    for topic in SEED_TOPICS:
        if dry_run:
            print(f"[dry-run] would seed topic_id={topic.topic_id} user_id={topic.user_id}")
            continue
        repo.save_topic(topic)
        print(f"seeded topic_id={topic.topic_id} user_id={topic.user_id}")

    print(f"\n{'[dry-run] ' if dry_run else ''}seeded {len(SEED_TOPICS)} topics")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    seed(dry_run=dry_run)
