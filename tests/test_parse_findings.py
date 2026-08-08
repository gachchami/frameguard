from frameguard.parse_findings import parse_model_findings


def test_parses_fenced_top_level_array_and_expands_zero_interval() -> None:
    raw = """
    ```json
    [
      {
        "type": "api_key",
        "value": "sk-demo-12345678",
        "modality": "visual",
        "start_seconds": 0.0,
        "end_seconds": 0.0,
        "confidence": 1.0
      }
    ]
    ```
    """
    findings = parse_model_findings(raw, 5.0)
    assert len(findings) == 1
    assert findings[0].start_seconds == 0.0
    assert findings[0].end_seconds == 5.0


def test_clamps_interval_to_chunk_duration() -> None:
    raw = """
    [{
      "type": "account_id",
      "value": "CUST-493821",
      "modality": "both",
      "start_seconds": 0,
      "end_seconds": 14.8,
      "confidence": 1
    }]
    """
    findings = parse_model_findings(raw, 5.0)
    assert findings[0].end_seconds == 5.0


def test_ignores_commentary_after_complete_json() -> None:
    raw = """
    {"findings": [{
      "type": "face",
      "value": "person",
      "start_seconds": 1,
      "end_seconds": 2
    }]}
    I found one item in the clip.
    """
    findings = parse_model_findings(raw, 5.0)
    assert len(findings) == 1
    assert findings[0].value == "person"


def test_uses_first_complete_json_when_model_repeats_response() -> None:
    raw = """
    {"findings": [{
      "type": "face",
      "value": "first response",
      "start_seconds": 0,
      "end_seconds": 1
    }]}
    {"findings": [{
      "type": "face",
      "value": "duplicate response",
      "start_seconds": 0,
      "end_seconds": 1
    }]}
    """
    findings = parse_model_findings(raw, 5.0)
    assert [finding.value for finding in findings] == ["first response"]


def test_braces_inside_json_string_do_not_confuse_extraction() -> None:
    raw = '{"type": "text", "value": "token {redacted}", "end_seconds": 1} trailing'
    findings = parse_model_findings(raw, 5.0)
    assert findings[0].value == "token {redacted}"


def test_rejects_echoed_response_schema_example() -> None:
    raw = """
    {"findings": [{
      "type": "api_key|password|email|phone|other",
      "value": "exact sensitive value",
      "start_seconds": 0,
      "end_seconds": 10,
      "visual_location": "short location description or null"
    }]}
    """
    assert parse_model_findings(raw, 10.0) == []
