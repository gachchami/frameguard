# FrameGuard face-reference test dataset

This dataset contains **56 short MP4 videos** for testing face detection,
tracking, and reference-only redaction.

## Coverage

- 48 single-face videos:
  - 2 subjects
  - 8 screen regions
  - 3 face sizes
  - varied static, horizontal, vertical, diagonal, orbital, and zoom motion
- 8 two-face videos for testing:
  - blur all faces
  - upload `reference/subject_a.png` and blur only subject A
  - upload `reference/subject_b.png` and blur only subject B

Screen regions:

- top-left, top-center, top-right
- center-left, center, center-right
- bottom-left, bottom-right

Face target heights:

- small: approximately 32 pixels
- medium: approximately 64 pixels
- large: approximately 112 pixels

## Files

- `reference/subject_a.png`
- `reference/subject_b.png`
- `videos/*.mp4`
- `manifest.csv`
- `annotations.json`
- `contact_sheet.jpg`

`annotations.json` records the known synthetic placement box for every face on
every frame. These boxes are useful for evaluating detector recall, tracking
coverage, and redaction coverage.

## Recommended tests

### Blur all faces

1. Upload any video from `videos/`.
2. Enable face redaction.
3. Select **Blur every detected face**.
4. Process the video.

### Blur one uploaded reference face

1. Upload one of the `dual_subjects_*.mp4` videos.
2. Enable face redaction.
3. Select **Blur only the uploaded reference face**.
4. Upload `reference/subject_a.png` or `reference/subject_b.png`.
5. Process the video.
6. Confirm that only the matching subject is blurred.

## Important limitation

These videos are synthetic compositions of two public sample portraits. They
are suitable for functional and regression testing, but they are not a complete
real-world benchmark. Add consented recordings with side profiles, occlusions,
low light, motion blur, masks, glasses, and crowded scenes before treating face
redaction as production-ready.
