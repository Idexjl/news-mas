from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Keys whose values must never appear in structured logs
_BLOCKED_LOG_KEYS = frozenset(
    {"content", "text", "body", "article", "summary", "raw", "html", "snippet"}
)

_tracer: trace.Tracer | None = None


def setup_telemetry(service_name: str) -> trace.Tracer:
    """
    Initialise the OTLP trace exporter for *service_name*.
    Safe to call multiple times; subsequent calls are no-ops.
    """
    global _tracer
    if _tracer is not None:
        return _tracer

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)
    return _tracer


def get_tracer(service_name: str = "news-mas") -> trace.Tracer:
    if _tracer is not None:
        return _tracer
    return trace.get_tracer(service_name)


class _SafeJSONFormatter(logging.Formatter):
    """
    Emits structured JSON log lines.

    Sanitisation rules:
      • Never serialise values whose key is in _BLOCKED_LOG_KEYS.
      • Only emit scalar or dict/list-of-scalar extra fields.
      • run_id and agent are promoted to top-level if present.
    """

    def format(self, record: logging.LogRecord) -> str:
        log: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Promote trace / run context fields
        for key in ("run_id", "agent", "span_id", "trace_id"):
            if hasattr(record, key):
                log[key] = getattr(record, key)

        # Attach safe extra fields
        _stdlib_keys = logging.LogRecord.__dict__.keys() | {
            "message", "asctime", "args", "exc_info", "exc_text", "stack_info",
            "run_id", "agent", "span_id", "trace_id",
        }
        safe_extra = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _stdlib_keys and k.lower() not in _BLOCKED_LOG_KEYS
        }
        if safe_extra:
            log["extra"] = safe_extra

        if record.exc_info:
            log["exception"] = self.formatException(record.exc_info)

        return json.dumps(log, default=str)


def get_logger(name: str) -> logging.Logger:
    """Return a structured-JSON logger. Never log raw article content through it."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_SafeJSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    return logger


def log_with_run_id(logger: logging.Logger, level: int, msg: str, run_id: str, **kwargs: Any) -> None:
    """Convenience wrapper that threads run_id into every log record."""
    extra = {"run_id": run_id, **kwargs}
    logger.log(level, msg, extra=extra)
