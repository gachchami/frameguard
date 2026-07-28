from frameguard.visual_locator import normalize_for_search


def test_normalization_ignores_ocr_spacing_and_punctuation() -> None:
    assert normalize_for_search("alice @ example.com") == normalize_for_search("alice@example.com")
    assert normalize_for_search("+91 98765-43210") == "919876543210"
