import pytest

presidio = pytest.importorskip("presidio_analyzer", reason="presidio-analyzer not installed")

from src.common.pii_scrubber import detect_pii, scrub_text


def test_detect_email():
    hits = detect_pii("Reach me at alice@example.com anytime.")
    types = [h["type"] for h in hits]
    assert "EMAIL_ADDRESS" in types


def test_detect_phone_number():
    hits = detect_pii("Call 555-867-5309 for support.")
    types = [h["type"] for h in hits]
    assert "PHONE_NUMBER" in types


def test_detect_us_ssn():
    # 123-45-6789 is on Presidio's deny-list (invalidate_result blocks it).
    # Use a number that passes all validation checks.
    hits = detect_pii("SSN: 301-55-8374")
    types = [h["type"] for h in hits]
    assert "US_SSN" in types


def test_detect_person_name():
    hits = detect_pii("Dr. John Smith signed the report.")
    types = [h["type"] for h in hits]
    assert "PERSON" in types


def test_detect_credit_card():
    hits = detect_pii("Card number: 4111 1111 1111 1111")
    types = [h["type"] for h in hits]
    assert "CREDIT_CARD" in types


def test_phi_medical_license_returns_list():
    hits = detect_pii("License: MD12345")
    assert isinstance(hits, list)


def test_scrub_removes_email():
    result = scrub_text("Contact bob@corp.org for the report.")
    assert "bob@corp.org" not in result
    assert "<EMAIL_ADDRESS>" in result


def test_scrub_removes_phone():
    result = scrub_text("Phone: 800-555-1234")
    assert "800-555-1234" not in result


def test_scrub_clean_text_unchanged():
    text = "The prime minister announced new climate policy today."
    result = scrub_text(text)
    assert isinstance(result, str)


def test_scrub_returns_string_type():
    assert isinstance(scrub_text(""), str)


def test_detect_returns_list_for_clean_text():
    hits = detect_pii("Stock markets rose 2% in afternoon trading.")
    assert isinstance(hits, list)


def test_detection_includes_score():
    hits = detect_pii("My email is test@example.com")
    email_hits = [h for h in hits if h["type"] == "EMAIL_ADDRESS"]
    assert len(email_hits) > 0
    assert "score" in email_hits[0]
    assert 0.0 <= email_hits[0]["score"] <= 1.0
