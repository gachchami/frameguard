from pathlib import Path

from frameguard.observability import RunEventRecorder


def test_run_log_never_contains_secret_values(tmp_path: Path) -> None:
    log_path = tmp_path / "run.jsonl"
    recorder = RunEventRecorder(log_path, run_id="test", level="DEBUG")

    secret = "sk-demo-83hhd8282hd91jd82"
    recorder.info("finding.detected", value=secret, type="api_key")
    recorder.debug("model.debug", raw_text=f'{{"value":"{secret}"}}')

    content = log_path.read_text(encoding="utf-8")
    assert secret not in content
    assert "fingerprint" in content
    assert '"redacted": true' in content
