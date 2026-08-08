from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Any

import cv2
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .face_gallery import (
    FaceGallerySession,
    analyze_video_with_face_gallery,
    match_uploaded_reference_photos,
    scan_face_gallery,
)
from .face_reference import DEFAULT_COSINE_THRESHOLD
from .minor_pipeline import analyze_video_with_face_policy
from .observability import configure_application_logging, mask_value
from .pipeline import PipelineResult, analyze_video

configure_application_logging()
LOGGER = logging.getLogger("frameguard.web")

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.environ.get("FRAMEGUARD_OUTPUT_DIR", ROOT / "outputs")).resolve()
UPLOAD_DIR = OUTPUT_DIR / "uploads"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _defaults() -> dict[str, Any]:
    return {
        "api_base": os.environ.get("FRAMEGUARD_API_BASE", "http://127.0.0.1:8091/v1"),
        "model": os.environ.get("FRAMEGUARD_MODEL", "/workspace/persistent/Qwen2.5-Omni-3B"),
        "chunk_seconds": float(os.environ.get("FRAMEGUARD_CHUNK_SECONDS", "5")),
        "detector_mode": os.environ.get("FRAMEGUARD_DETECTOR", "qwen"),
        "face_model_path": os.environ.get(
            "FRAMEGUARD_FACE_MODEL", str(ROOT / "models/face_detection_yunet_2023mar.onnx")
        ),
        "face_recognition_model_path": os.environ.get(
            "FRAMEGUARD_FACE_RECOGNITION_MODEL",
            str(ROOT / "models/face_recognition_sface_2021dec.onnx"),
        ),
        "deterministic_sample_interval_ms": 350,
        "face_sample_interval_ms": 200,
        "face_score_threshold": 0.75,
        "face_max_track_gap_ms": 900,
        "face_min_track_observations": 2,
        "reference_match_threshold": DEFAULT_COSINE_THRESHOLD,
        "identity_similarity_threshold": 0.40,
        "run_log_level": "INFO",
        "show_sensitive_values": False,
        "include_raw_model_output": False,
    }


def _config(raw: str) -> dict[str, Any]:
    try:
        supplied = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "Configuration must be valid JSON") from exc
    if not isinstance(supplied, dict):
        raise HTTPException(400, "Configuration must be a JSON object")
    result = _defaults()
    result.update(supplied)
    return result


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"
    message: str = "Waiting to start"
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class RuntimeState:
    jobs: dict[str, Job] = field(default_factory=dict)
    galleries: dict[str, FaceGallerySession] = field(default_factory=dict)
    assets: dict[str, Path] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


STATE = RuntimeState()

app = FastAPI(title="FrameGuard API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _save_upload(upload: UploadFile, category: str) -> Path:
    suffix = Path(upload.filename or "upload").suffix.lower()
    allowed = {"video": {".mp4", ".mov", ".m4v"}, "image": {".jpg", ".jpeg", ".png", ".webp"}}
    if suffix not in allowed[category]:
        raise HTTPException(400, f"Unsupported {category} file type: {suffix or 'unknown'}")
    target = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with target.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return target


def _new_job(kind: str) -> Job:
    job = Job(id=uuid.uuid4().hex, kind=kind)
    with STATE.lock:
        STATE.jobs[job.id] = job
    return job


def _run_job(job_id: str, work: Callable[[], dict[str, Any]]) -> None:
    with STATE.lock:
        job = STATE.jobs[job_id]
        job.status = "running"
        job.message = "Analyzing and rendering your video"
    try:
        result = work()
    except Exception as exc:
        LOGGER.exception("Job %s failed", job_id)
        with STATE.lock:
            job.status = "failed"
            job.message = "Processing failed"
            job.error = str(exc)
        return
    with STATE.lock:
        job.status = "completed"
        job.message = "Protected video is ready"
        job.result = result


def _pipeline_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "api_base": str(config["api_base"]),
        "model": str(config["model"]),
        "api_key": os.environ.get("FRAMEGUARD_API_KEY", "EMPTY"),
        "chunk_seconds": float(config["chunk_seconds"]),
        "output_dir": OUTPUT_DIR,
        "detector_mode": str(config["detector_mode"]),
        "deterministic_sample_interval_ms": int(config["deterministic_sample_interval_ms"]),
        "run_log_level": str(config["run_log_level"]),
        "include_sensitive_values_in_report": bool(config["show_sensitive_values"]),
        "include_raw_model_output": bool(config["include_raw_model_output"]),
    }


def _face_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "face_model_path": str(config["face_model_path"]),
        "face_sample_interval_ms": int(config["face_sample_interval_ms"]),
        "face_score_threshold": float(config["face_score_threshold"]),
        "face_max_track_gap_ms": int(config["face_max_track_gap_ms"]),
        "face_min_track_observations": int(config["face_min_track_observations"]),
    }


def _result_payload(result: PipelineResult, show_sensitive_values: bool) -> dict[str, Any]:
    asset_ids: dict[str, str] = {}
    for name, path in {
        "video": result.output_video,
        "report": result.report_path,
        "log": result.log_path,
    }.items():
        asset_id = uuid.uuid4().hex
        STATE.assets[asset_id] = path.resolve()
        asset_ids[name] = asset_id
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    findings = []
    for item in result.findings:
        payload = asdict(item)
        payload["value"] = (
            item.value
            if show_sensitive_values
            else mask_value(item.value, item.type)
        )
        findings.append(payload)
    return {
        "run_id": result.run_id,
        "assets": {name: f"/api/assets/{asset_id}" for name, asset_id in asset_ids.items()},
        "findings": findings,
        "metrics": result.metrics,
        "report_preview": {
            key: report.get(key)
            for key in (
                "privacy",
                "configuration",
                "metrics",
                "face_gallery",
                "child_classification",
            )
            if report.get(key) is not None
        },
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def config() -> dict[str, Any]:
    return _defaults()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with STATE.lock:
        job = STATE.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return asdict(job)


@app.get("/api/assets/{asset_id}")
def get_asset(asset_id: str) -> FileResponse:
    path = STATE.assets.get(asset_id)
    if path is None or not path.is_file():
        raise HTTPException(404, "Asset not found")
    return FileResponse(path, filename=path.name)


@app.post("/api/jobs/sensitive", status_code=202)
def sensitive_job(
    background: BackgroundTasks,
    video: Annotated[UploadFile, File()],
    config_json: Annotated[str, Form()] = "{}",
) -> dict[str, str]:
    path, cfg, job = _save_upload(video, "video"), _config(config_json), _new_job("sensitive")

    def work() -> dict[str, Any]:
        result = analyze_video(
            path,
            deterministic_ocr=bool(cfg.get("deterministic_ocr", True)),
            detect_qr_codes=bool(cfg.get("detect_qr_codes", True)),
            redact_faces=False,
            **_pipeline_kwargs(cfg),
        )
        return _result_payload(result, bool(cfg["show_sensitive_values"]))

    background.add_task(_run_job, job.id, work)
    return {"job_id": job.id}


@app.post("/api/jobs/automatic", status_code=202)
def automatic_job(
    background: BackgroundTasks,
    video: Annotated[UploadFile, File()],
    reference_face: Annotated[UploadFile | None, File()] = None,
    config_json: Annotated[str, Form()] = "{}",
) -> dict[str, str]:
    path, cfg, job = _save_upload(video, "video"), _config(config_json), _new_job("automatic")
    reference_path = _save_upload(reference_face, "image") if reference_face else None
    mode = str(cfg.get("face_redaction_mode", "all"))
    if mode == "reference" and reference_path is None:
        raise HTTPException(400, "A reference face is required for reference mode")

    def work() -> dict[str, Any]:
        common = {
            **_pipeline_kwargs(cfg),
            **_face_kwargs(cfg),
            "deterministic_ocr": False,
            "detect_qr_codes": False,
        }
        if mode == "likely_minors":
            result = analyze_video_with_face_policy(
                path,
                face_redaction_mode=mode,
                child_minimum_confidence=float(cfg.get("child_minimum_confidence", 0.70)),
                child_minimum_usable_timestamps=int(cfg.get("child_minimum_usable_timestamps", 3)),
                child_consensus_fraction=float(cfg.get("child_consensus_fraction", 0.70)),
                child_max_samples_per_track=int(cfg.get("child_max_samples_per_track", 5)),
                child_continue_on_error=bool(cfg.get("child_continue_on_error", True)),
                child_blur_uncertain=bool(cfg.get("child_blur_uncertain", False)),
                **common,
            )
        else:
            result = analyze_video(
                path,
                redact_faces=True,
                face_redaction_mode=mode,
                reference_face_path=reference_path,
                face_recognition_model_path=str(cfg["face_recognition_model_path"]),
                reference_match_threshold=float(cfg["reference_match_threshold"]),
                **common,
            )
        return _result_payload(result, bool(cfg["show_sensitive_values"]))

    background.add_task(_run_job, job.id, work)
    return {"job_id": job.id}


@app.post("/api/galleries", status_code=202)
def create_gallery(
    background: BackgroundTasks,
    video: Annotated[UploadFile, File()],
    config_json: Annotated[str, Form()] = "{}",
) -> dict[str, str]:
    path, cfg, job = _save_upload(video, "video"), _config(config_json), _new_job("gallery")

    def work() -> dict[str, Any]:
        session = scan_face_gallery(
            path,
            face_recognition_model_path=str(cfg["face_recognition_model_path"]),
            identity_similarity_threshold=float(cfg["identity_similarity_threshold"]),
            **_face_kwargs(cfg),
        )
        with STATE.lock:
            STATE.galleries[session.session_id] = session
        return {
            "gallery_id": session.session_id,
            "summary": session.public_summary(),
            "profiles": [
                {
                    **profile.public_summary(),
                    "portrait_url": (
                        f"/api/galleries/{session.session_id}/portraits/"
                        f"{profile.person_id}"
                    ),
                    "preview_urls": [
                        f"/api/galleries/{session.session_id}/portraits/"
                        f"{profile.person_id}/{index}"
                        for index in range(len(profile.preview_rgbs))
                    ],
                }
                for profile in session.profiles
            ],
        }

    background.add_task(_run_job, job.id, work)
    return {"job_id": job.id}


@app.get("/api/galleries/{gallery_id}/portraits/{person_id}")
def gallery_portrait(gallery_id: str, person_id: str) -> Response:
    session = STATE.galleries.get(gallery_id)
    if session is None:
        raise HTTPException(404, "Gallery not found")
    profile = next((item for item in session.profiles if item.person_id == person_id), None)
    if profile is None:
        raise HTTPException(404, "Profile not found")
    bgr = cv2.cvtColor(profile.portrait_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise HTTPException(500, "Could not encode portrait")
    return Response(
        encoded.tobytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/galleries/{gallery_id}/portraits/{person_id}/{preview_index}")
def gallery_portrait_preview(
    gallery_id: str,
    person_id: str,
    preview_index: int,
) -> Response:
    session = STATE.galleries.get(gallery_id)
    if session is None:
        raise HTTPException(404, "Gallery not found")
    profile = next((item for item in session.profiles if item.person_id == person_id), None)
    if profile is None or not 0 <= preview_index < len(profile.preview_rgbs):
        raise HTTPException(404, "Profile preview not found")
    bgr = cv2.cvtColor(profile.preview_rgbs[preview_index], cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise HTTPException(500, "Could not encode portrait preview")
    return Response(
        encoded.tobytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/galleries/{gallery_id}/matches")
def preview_matches(
    gallery_id: str,
    photos: Annotated[list[UploadFile], File()],
    config_json: Annotated[str, Form()] = "{}",
) -> dict[str, Any]:
    session, cfg = STATE.galleries.get(gallery_id), _config(config_json)
    if session is None:
        raise HTTPException(404, "Gallery not found")
    paths = [_save_upload(photo, "image") for photo in photos]
    matches = match_uploaded_reference_photos(
        session,
        paths,
        face_model_path=str(cfg["face_model_path"]),
        face_recognition_model_path=str(cfg["face_recognition_model_path"]),
        threshold=float(cfg["reference_match_threshold"]),
    )
    return {"matches": matches}


@app.post("/api/galleries/{gallery_id}/render", status_code=202)
def render_gallery(
    gallery_id: str,
    background: BackgroundTasks,
    photos: Annotated[list[UploadFile] | None, File()] = None,
    config_json: Annotated[str, Form()] = "{}",
) -> dict[str, str]:
    session, cfg = STATE.galleries.get(gallery_id), _config(config_json)
    if session is None:
        raise HTTPException(404, "Gallery not found")
    selected_ids = set(cfg.get("selected_person_ids", []))
    selected_labels = [
        profile.label
        for profile in session.profiles
        if profile.person_id in selected_ids
    ]
    photo_paths = [_save_upload(photo, "image") for photo in (photos or [])]
    if not selected_labels and not photo_paths:
        raise HTTPException(400, "Select at least one person or upload a reference photo")
    job = _new_job("gallery_render")

    def work() -> dict[str, Any]:
        result = analyze_video_with_face_gallery(
            session.video_path,
            session=session,
            selected_labels=selected_labels,
            gallery_action=str(cfg.get("gallery_action", "blur_selected")),
            uploaded_photos=photo_paths,
            uploaded_photo_action=str(cfg.get("uploaded_photo_action", "blur")),
            face_model_path=str(cfg["face_model_path"]),
            face_recognition_model_path=str(cfg["face_recognition_model_path"]),
            reference_match_threshold=float(cfg["reference_match_threshold"]),
            deterministic_ocr=False,
            detect_qr_codes=False,
            **_pipeline_kwargs(cfg),
        )
        return _result_payload(result, bool(cfg["show_sensitive_values"]))

    background.add_task(_run_job, job.id, work)
    return {"job_id": job.id}


FRONTEND_DIST = ROOT / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
