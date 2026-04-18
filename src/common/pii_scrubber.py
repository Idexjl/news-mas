from __future__ import annotations

import logging
from typing import Any

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

logger = logging.getLogger(__name__)

# Entity types to detect — standard PII + healthcare-adjacent PHI identifiers
# that Presidio ships recognisers for out of the box.
_PII_ENTITIES: list[str] = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "CRYPTO",
    "IBAN_CODE",
    "IP_ADDRESS",
    "LOCATION",
    "DATE_TIME",
    "NRP",              # national/religious/political group
    "URL",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_ITIN",
    "US_PASSPORT",
    "US_SSN",
    "MEDICAL_LICENSE",  # PHI — medical professional licence numbers
    "UK_NHS",           # PHI — UK National Health Service numbers
]

_analyzer: AnalyzerEngine | None = None
_anonymizer: AnonymizerEngine | None = None

# Preferred spaCy models in descending quality order.
# Pattern-based recognizers (US_SSN, CREDIT_CARD, etc.) do not need NLP,
# but Presidio still requires a valid NlpEngine to be attached at init time.
# Falling back through smaller models avoids a hard crash when en_core_web_lg
# hasn't been downloaded yet.
_SPACY_MODELS = ("en_core_web_lg", "en_core_web_md", "en_core_web_sm")


def _build_analyzer() -> AnalyzerEngine:
    for model_name in _SPACY_MODELS:
        try:
            cfg = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": model_name}],
            }
            nlp_engine = NlpEngineProvider(nlp_configuration=cfg).create_engine()
            return AnalyzerEngine(
                nlp_engine=nlp_engine,
                supported_languages=["en"],
            )
        except Exception:
            continue
    # Last resort: let Presidio try its own default (requires en_core_web_lg)
    return AnalyzerEngine()


def _get_engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    global _analyzer, _anonymizer
    if _analyzer is None:
        _analyzer = _build_analyzer()
    if _anonymizer is None:
        _anonymizer = AnonymizerEngine()
    return _analyzer, _anonymizer


def detect_pii(text: str, language: str = "en") -> list[dict[str, Any]]:
    """
    Detect PII/PHI entities in *text*.

    Returns a list of dicts with keys: type, start, end, score.
    Raw content is never logged; only entity type and count are emitted.
    """
    analyzer, _ = _get_engines()
    results: list[RecognizerResult] = analyzer.analyze(
        text=text,
        entities=_PII_ENTITIES,
        language=language,
    )
    detections = [
        {"type": r.entity_type, "start": r.start, "end": r.end, "score": r.score}
        for r in results
    ]
    if detections:
        type_counts: dict[str, int] = {}
        for d in detections:
            type_counts[d["type"]] = type_counts.get(d["type"], 0) + 1
        logger.info("PII/PHI detected", extra={"entity_counts": type_counts})
    return detections


def scrub_text(text: str, language: str = "en") -> str:
    """
    Anonymise all detected PII/PHI in *text* by replacing each span with
    a type placeholder, e.g. <EMAIL_ADDRESS>.
    """
    analyzer, anonymizer = _get_engines()
    results = analyzer.analyze(
        text=text,
        entities=_PII_ENTITIES,
        language=language,
    )
    if not results:
        return text

    operators = {
        entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
        for entity in _PII_ENTITIES
    }
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators,
    )
    type_counts: dict[str, int] = {}
    for r in results:
        type_counts[r.entity_type] = type_counts.get(r.entity_type, 0) + 1
    logger.info("PII/PHI scrubbed", extra={"entity_counts": type_counts})
    return anonymized.text
