from frameguard.observability import mask_value


def test_generated_face_track_label_is_visible() -> None:
    assert mask_value("face_001", "face") == "face_001"


def test_generated_reference_face_track_label_is_visible() -> None:
    assert mask_value("reference_face_001", "face") == "reference_face_001"
