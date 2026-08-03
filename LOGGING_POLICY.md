# FrameGuard instrumentation policy

## INFO

Records pipeline lifecycle, video metadata, chunk boundaries, Qwen request/response summaries, finding counts, merge results, localization totals, rendering totals, report paths, and total duration.

## DEBUG

Adds privacy-safe per-frame OCR/QR counts and per-finding localization diagnostics.

## WARNING

Records recoverable privacy risks such as a visual finding that could not be localized.

## ERROR

Records failed stages, HTTP status, exception type, safe error summaries, and processing duration. It does not include response bodies or local variables.

## Data that is never logged

- Exact finding values
- Raw model responses
- Prompts and message payloads
- Video/audio data URIs
- Authorization headers, credentials, passwords, or tokens

## Correlation without exposure

Every detected value is represented in logs by:

- type
- character length
- a 12-character per-run keyed fingerprint

The fingerprint is stable within one run, allowing Qwen and OCR detections of the same value to be correlated. It changes on the next run and is not intended as a permanent identifier.
