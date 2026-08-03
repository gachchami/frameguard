from frameguard.deterministic_detectors import detect_pattern_matches


def test_detects_sample_privacy_patterns() -> None:
    text = (
        "API_KEY=sk-demo-83hhd8282hd91jd82 "
        "email alice@example.com internal server 192.168.1.24 "
        "account CUST-493821"
    )
    matches = detect_pattern_matches(text)
    found = {(item.type, item.value) for item in matches}

    assert ("api_key", "sk-demo-83hhd8282hd91jd82") in found
    assert ("email", "alice@example.com") in found
    assert ("ip_address", "192.168.1.24") in found
    assert ("account_id", "CUST-493821") in found
