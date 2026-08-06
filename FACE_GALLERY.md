# Manual Face Gallery

The manual face gallery provides a user-reviewed face-redaction workflow.

## Workflow

1. Upload a video.
2. Click **Extract face profiles**.
3. FrameGuard runs YuNet face detection and uses SFace embeddings to group
   fragmented tracks that appear to belong to the same person.
4. Review the profile gallery.
5. Select gallery people and choose either:
   - **Blur selected gallery people**; or
   - **Keep selected gallery people visible; blur everyone else**.
6. Optionally upload one or more reference photos and choose whether matching
   people should be blurred or kept visible.
7. Preview uploaded-photo matches, then click **Render manual face selection**.

The final render still runs the normal Qwen video/audio privacy analysis,
OCR/regex scan, and QR detection. The selected face-track findings are added to
those results before the MP4 is rendered.

## Privacy

Face profile crops and SFace embeddings remain in the server-side Gradio
session. They are not written into the JSON audit report or JSONL run log.
The report records profile IDs, associated track IDs, selection actions, and
uploaded-photo match scores.

## Thresholds

- **Face-gallery identity grouping threshold** controls how fragmented YuNet
  tracks are grouped.
  - Raise it when different people are merged.
  - Lower it when one person appears as several profiles.
- **SFace uploaded-photo match threshold** controls uploaded-reference matching.
  Higher values are stricter.

SFace matching assists a user-reviewed workflow; it is not identity
verification.

## Apply

From the repository root:

```bash
unzip -o frameguard_face_gallery_overlay.zip -d .
uv run python -m compileall -q app.py frameguard
uv run pytest -q tests/test_face_gallery.py
./scripts/start.sh
```
