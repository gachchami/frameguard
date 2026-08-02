from frameguard.parse_findings import parse_model_findings


def test_parses_findings_wrapper() -> None:
    raw = """
    {
      "findings": [
        {
          "type": "email",
          "value": "alice@example.com",
          "modality": "visual",
          "start_seconds": 1.0,
          "end_seconds": 3.0,
          "confidence": 0.9,
          "reason": "Personal information"
        }
      ]
    }
    """

    findings = parse_model_findings(raw, 5.0)

    assert len(findings) == 1
    assert findings[0].value == "alice@example.com"
    assert findings[0].start_seconds == 1.0
    assert findings[0].end_seconds == 3.0


def test_parses_top_level_array() -> None:
    raw = """
    [
      {
        "type": "api_key",
        "value": "sk-demo-123",
        "modality": "visual",
        "start_seconds": 0.0,
        "end_seconds": 0.0,
        "confidence": 1.0,
        "reason": "Exposed API key"
      }
    ]
    """

    findings = parse_model_findings(raw, 5.0)

    assert len(findings) == 1
    assert findings[0].value == "sk-demo-123"

    # Invalid 0 -> 0 timestamps become the full chunk.
    assert findings[0].start_seconds == 0.0
    assert findings[0].end_seconds == 5.0


def test_parses_fenced_array() -> None:
    raw = """
    ```json
    [
      {
        "type": "account_id",
        "value": "CUST-493821",
        "modality": "both",
        "start_seconds": 0.0,
        "end_seconds": 14.8,
        "confidence": 1.0,
        "reason": "Private account identifier"
      }
    ]
    ```
    """

    findings = parse_model_findings(raw, 5.0)

    assert len(findings) == 1
    assert findings[0].value == "CUST-493821"

    # Timestamp is clamped to the five-second chunk.
    assert findings[0].start_seconds == 0.0
    assert findings[0].end_seconds == 5.0


def test_parses_single_finding_object() -> None:
    raw = """
    {
      "type": "email",
      "value": "alice@example.com",
      "modality": "visual",
      "start_seconds": 0,
      "end_seconds": 5,
      "confidence": 1
    }
    """

    findings = parse_model_findings(raw, 5.0)

    assert len(findings) == 1
    assert findings[0].value == "alice@example.com"
