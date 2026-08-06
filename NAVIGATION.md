# FrameGuard Navigation

The interface is organized into four top-level sections.

## Sensitive Content

A dedicated workflow for detecting and redacting:

- API keys
- email addresses
- IP addresses
- account identifiers
- phone numbers
- private URLs
- QR codes
- sensitive spoken information

This workflow does not enable face redaction.

## Face Privacy

A separate workflow for detecting people and blurring faces.

It includes:

- reviewed face-gallery selection;
- blur-selected and keep-selected-visible rules;
- uploaded reference-photo matching;
- blur-all and single-reference automatic rules;
- experimental visual child classification.

## Results

Shared protected-video preview, downloads, findings, metrics, and audit
output.

## Settings

Shared Qwen, YuNet, SFace, OCR, tracking, reporting, and diagnostics
controls.

## Apply

```bash
cd /persistent/projects/frameguard
unzip -o frameguard_two_navigation_sections_overlay.zip -d .

uv run python -m compileall -q app.py
./scripts/start.sh
```
