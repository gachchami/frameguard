from frameguard.minor_pipeline import normalize_face_redaction_mode


def test_face_mode_aliases() -> None:
    assert normalize_face_redaction_mode("off") == "off"
    assert normalize_face_redaction_mode("blur-all") == "all"
    assert normalize_face_redaction_mode("children") == "likely_minors"
    assert normalize_face_redaction_mode("visually apparent children") == "likely_minors"
