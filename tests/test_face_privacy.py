from frameguard.observability import mask_value


def test_generated_face_track_label_is_visible() -> None:
    assert mask_value("face_001", "face") == "face_001"
