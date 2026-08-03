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
