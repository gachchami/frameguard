from frameguard.pipeline import merge_findings
from frameguard.schemas import Finding


def test_merges_qwen_and_ocr_and_preserves_both_sources() -> None:
    qwen = Finding(
        id="qwen",
        type="email",
        value="alice@example.com",
        modality="both",
        start_ms=5000,
        end_ms=10000,
        confidence=0.9,
        sources=["qwen"],
    )
    ocr = Finding(
        id="ocr",
        type="email",
        value="alice@example.com",
        modality="visual",
        start_ms=5100,
        end_ms=9900,
        confidence=0.99,
        sources=["ocr_regex"],
    )

    merged = merge_findings([qwen, ocr])
    assert len(merged) == 1
    assert merged[0].modality == "both"
    assert merged[0].action == "blur_and_mute"
    assert merged[0].sources == ["ocr_regex", "qwen"]
