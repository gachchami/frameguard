# Verify the reference-face overlay

After extracting at the FrameGuard repository root, run:

```bash
git status --short --untracked-files=all

grep -n "Blur only the uploaded reference face" app.py
grep -n "ReferenceFaceMatcher" frameguard/face_reference.py
grep -n "redaction_mode == \"reference\"" frameguard/face_tracking.py
```

Expected new files include:

- `frameguard/face_reference.py`
- `scripts/download_sface_model.py`
- `scripts/check_sface_model.py`
- `scripts/generate_face_test_dataset.py`
- `tests/test_face_privacy.py`
- `OVERLAY_VERSION.txt`

The modified UI file must contain the label:

`Blur only the uploaded reference face`
