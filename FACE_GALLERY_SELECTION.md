# FrameGuard Gallery Selection and Deduplication Fix

This flat repository overlay addresses:

- face cards that were too small or unclear;
- no visible selected state in the gallery;
- duplicate profiles caused by fragmented face tracks;
- H.264 `mmco: unref short failure` warnings caused by repeated random seeks;
- the Gradio 6 warning that `css` must be passed to `launch()`.

## Behavior

- Click a face card to select or deselect it.
- Selected cards show a green border and a `SELECTED` banner.
- Gallery cards are larger and use a tighter crop.
- Several candidate frames are ranked by source face size, sharpness, and YuNet
  confidence before choosing a profile image.
- Up to three strong SFace embeddings are averaged for each track fragment.
- Non-overlapping compatible fragments are merged into one unique person.
- Simultaneously visible faces are never merged.
- The status reports unique people, raw track fragments, and merged duplicates.

## Apply

```bash
cd /persistent/projects/frameguard
unzip -o frameguard_gallery_selection_dedup_overlay.zip -d .

uv run python -m compileall -q app.py frameguard
uv run pytest -q tests/test_face_gallery.py

./scripts/start.sh
```
