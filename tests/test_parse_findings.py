from frameguard.parse_findings import parse_model_findings


def test_parses_fenced_json_and_clamps_times() -> None:
    raw = """```json
    {"findings": [{
      "type": "phone",
      "value": "+91 98765 43210",
      "modality": "audio",
      "start_seconds": -2,
      "end_seconds": 99,
      "confidence": 1.4,
      "reason": "personal data",
      "visual_location": null
    }]}
    ```"""
    findings = parse_model_findings(raw, clip_duration_seconds=5.0)
    assert len(findings) == 1
    assert findings[0].start_seconds == 0.0
    assert findings[0].end_seconds == 5.0
    assert findings[0].confidence == 1.0
