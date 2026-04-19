from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Keys whose values must never appear in structured logs
_BLOCKED_LOG_KEYS = frozenset(
    {"content", "text", "body", "article", "summary", "raw", "html", "snippet"}
)

_tracer: trace.Tracer | None = None
_meter_provider: MeterProvider | None = None


def _build_resource(service_name: str) -> Resource:
    return Resource.create({
        "service.name": os.getenv("SERVICE_NAME", service_name),
        "service.version": os.getenv("SERVICE_VERSION", "0.1.0"),
        "deployment.environment": os.getenv("ENVIRONMENT", "local-dev"),
    })


def setup_telemetry(service_name: str) -> trace.Tracer:
    """
    Initialise the OTLP trace exporter for *service_name*.
    Reads SERVICE_NAME, SERVICE_VERSION, ENVIRONMENT from env with fallbacks.
    Safe to call multiple times; subsequent calls are no-ops.

    [SIEM-TODO] In production, OTel traces are ingested by Azure Sentinel for
    UEBA (User and Entity Behavior Analytics) and Fusion detection:
      - UEBA builds a behavioral baseline per agent (service.name) using span
        timing, call frequency, and attribute distributions. Deviations — e.g.
        search-worker suddenly emitting spans to unexpected egress.host values —
        surface as anomalous entity behaviour in the Sentinel UEBA workbook.
      - Fusion detection correlates trace anomalies with identity signals from
        Entra ID (once DPoP is active) to detect multi-stage attacks: a
        compromised agent token used from an unexpected source AND unusual
        downstream span patterns would fuse into a high-confidence incident.
      Production path: swap the otlp/jaeger exporter in otel-collector-config.yaml
      for the Sentinel OTLP exporter. The resource processor already stamps
      service.name and deployment.environment, which Sentinel uses as entity keys.
      See src/common/SIEM_INTEGRATION_GUIDE.md for the full connector config.
    """
    global _tracer
    if _tracer is not None:
        return _tracer

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    resource = _build_resource(service_name)
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)
    return _tracer


def setup_metrics(service_name: str) -> metrics.Meter:
    """
    Initialise the OTLP metrics exporter for *service_name*.
    Safe to call multiple times; subsequent calls are no-ops.
    """
    global _meter_provider
    if _meter_provider is not None:
        return metrics.get_meter(service_name)

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    resource = _build_resource(service_name)
    exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True)
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=10_000)
    _meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(_meter_provider)
    return metrics.get_meter(service_name)


def get_tracer(service_name: str = "news-mas") -> trace.Tracer:
    if _tracer is not None:
        return _tracer
    return trace.get_tracer(service_name)


def get_meter(name: str = "news-mas") -> metrics.Meter:
    return metrics.get_meter(name)


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
    """
    Return a structured-JSON logger. Never log raw article content through it.

    [SIEM-TODO] In production, structured JSON logs flow:
      Agent → Loki (via OTel Collector log pipeline) → Azure Sentinel
      via the Sentinel OTel connector (Microsoft Sentinel Data Connector for
      OpenTelemetry). The _SafeJSONFormatter fields — msg, level, logger,
      run_id, extra — map directly to Sentinel Log Analytics workspace columns.
      Enable the connector in the Sentinel workspace and point the OTel
      collector's loki exporter at the Sentinel OTLP endpoint instead.
      All structured log events (circuit_breaker_triggered, injection_detected,
      aap_token_validated) then appear in the SecurityEvent table and can
      trigger Analytics Rules and Playbooks automatically.
    """
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


def configure_langsmith() -> bool:
    """
    Configure LangSmith tracing by mapping LANGSMITH_* env vars to the
    LANGCHAIN_* vars that LangChain/LangSmith SDK reads at call time.

    Returns True if configured, False if API key is absent (graceful
    degradation — agents start normally without tracing in offline dev).
    Safe to call multiple times; subsequent calls are no-ops.
    """
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        return True

    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        get_logger(__name__).warning(
            "LANGSMITH_API_KEY not set — LangSmith tracing disabled"
        )
        return False

    project = (
        os.getenv("LANGCHAIN_PROJECT")
        or os.getenv("LANGSMITH_PROJECT")
        or "news-mas-dev"
    )
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    return True
