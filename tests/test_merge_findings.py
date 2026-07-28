from frameguard.pipeline import merge_findings
from frameguard.schemas import Finding


def _finding(start: int, end: int) -> Finding:
    return Finding(
        id=f"f-{start}",
        type="email",
        value="alice@example.com",
        modality="visual",
        start_ms=start,
        end_ms=end,
        confidence=0.8,
    )


def test_merges_same_secret_across_chunk_boundary() -> None:
    merged = merge_findings([_finding(3800, 5000), _finding(5000, 6500)])
    assert len(merged) == 1
    assert merged[0].start_ms == 3800
    assert merged[0].end_ms == 6500
