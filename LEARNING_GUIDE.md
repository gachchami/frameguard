# FrameGuard learning order

Read the project in this order:

1. `frameguard/schemas.py` — the data objects passed between modules.
2. `frameguard/prompts.py` — what the multimodal model is asked to detect.
3. `frameguard/multimodal_llm.py` — how a local MP4 becomes a vLLM request.
4. `frameguard/parse_findings.py` — why model output must be validated.
5. `frameguard/video.py` — chunking and time offsets.
6. `frameguard/visual_locator.py` — OCR is used for coordinates, not judgment.
7. `frameguard/redact.py` — visual blur and audio muting.
8. `frameguard/pipeline.py` — orchestration and duplicate merging.
9. `app.py` — the Gradio interface around the pipeline.

## The key design lesson

The model answers the semantic question:

> What information is sensitive?

Deterministic tools answer the editing questions:

> Where is it in the image? During which interval should it be removed?

That division is important. Language models understand context, while OCR,
OpenCV, and FFmpeg provide repeatable coordinates and media transformations.
