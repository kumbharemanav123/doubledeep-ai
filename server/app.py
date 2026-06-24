from __future__ import annotations

import base64
import hashlib
import math
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
import onnxruntime as ort
import uvicorn
import yt_dlp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model.onnx"
MAX_VIDEO_SECONDS = 5 * 60
FRAME_LIMIT = 18
IMAGE_SIZE = 224

app = FastAPI(title="DoubleDeep local analyzer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

session: ort.InferenceSession | None = None


class AnalyzeUrlRequest(BaseModel):
    url: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def get_session() -> ort.InferenceSession:
    global session
    if session is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=500, detail="model.onnx was not found.")
        session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return session


def validate_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Enter a full http or https URL.")
    return url


def download_video(url: str, workdir: Path) -> Path:
    output = str(workdir / "%(id)s.%(ext)s")

    def reject_long(info: dict[str, Any], *, incomplete: bool) -> str | None:
        duration = info.get("duration")
        if duration and duration > MAX_VIDEO_SECONDS + 1:
            return "Videos must be 5 minutes or shorter."
        return None

    options = {
        "format": "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/best[height<=720]/best",
        "outtmpl": output,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "match_filter": reject_long,
        "merge_output_format": "mp4",
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as error:
        raise HTTPException(status_code=422, detail=f"Could not download this video: {error}") from error

    candidates = sorted(workdir.glob("*"), key=lambda path: path.stat().st_size, reverse=True)
    if not candidates:
        raise HTTPException(status_code=422, detail="No video file was produced by yt-dlp.")
    return candidates[0]


def build_view_rects(width: int, height: int) -> list[tuple[int, int, int, int]]:
    views = [(0, 0, width, height)]
    if height > width * 1.1:
        side = width
        for y in [0, max((height - side) // 2, 0), max(height - side, 0)]:
            views.append((0, y, side, side))
    elif width > height * 1.1:
        side = height
        for x in [0, max((width - side) // 2, 0), max(width - side, 0)]:
            views.append((x, 0, side, side))
    else:
        side = max(32, int(min(width, height) * 0.84))
        views.extend(
            [
                (0, 0, side, side),
                (max((width - side) // 2, 0), max((height - side) // 2, 0), side, side),
                (max(width - side, 0), max(height - side, 0), side, side),
            ]
        )
    return views


def frame_to_tensor(frame_rgb: np.ndarray, include_crops: bool) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    rects = build_view_rects(width, height) if include_crops else [(0, 0, width, height)]
    views = []
    for x, y, w, h in rects:
        crop = frame_rgb[y : y + h, x : x + w]
        resized = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        values = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        views.append(np.transpose(values, (2, 0, 1)))
    return np.stack(views, axis=0).astype(np.float32)


def aggregate_probabilities(probabilities: list[float]) -> dict[str, float | bool]:
    values = sorted(probabilities)
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    mean = float(np.mean(probabilities))
    disagreement = float(np.std(probabilities))
    ai_support = sum(value >= 0.8 for value in probabilities) / len(probabilities)
    real_support = sum(value <= 0.2 for value in probabilities) / len(probabilities)
    strong_conflict = max(probabilities) >= 0.95 and min(probabilities) <= 0.05
    probability = (0.56 * median) + (0.24 * mean) + (0.20 * probabilities[0])
    if ai_support > 0.5:
        probability = max(probability, 0.64 + (0.28 * ai_support))
    elif strong_conflict:
        probability = 0.5 + (0.08 * (ai_support - real_support))
    elif real_support > 0.5:
        probability = min(probability, 0.36 - (0.24 * real_support))
    probability = clamp(probability, 0.01, 0.99)
    support = max(ai_support, real_support)
    confidence = 0.52 + (0.30 * abs(probability - 0.5) * 2) + (0.18 * support) - (0.34 * disagreement)
    if strong_conflict and ai_support <= 0.5:
        confidence = min(confidence, 0.44)
    return {
        "probability": probability,
        "confidence": clamp(confidence, 0.25, 0.96),
        "disagreement": disagreement,
        "support": support,
        "strong_conflict": strong_conflict,
    }


def classify(probability: float, confidence: float) -> tuple[str, str]:
    if confidence < 0.48 or 0.43 <= probability <= 0.57:
        return "inconclusive", "Inconclusive"
    if probability > 0.57:
        return "likely_ai", "Likely AI-generated / deepfake"
    return "likely_real", "Likely camera-captured"


def frame_difference(first: np.ndarray | None, second: np.ndarray) -> float:
    if first is None:
        return 0.0
    small_a = cv2.resize(first, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    small_b = cv2.resize(second, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    return clamp(float(np.mean(np.abs(small_a - small_b)) / 70.0))


def texture_complexity(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    diff = np.abs(gray[:, 4:] - gray[:, :-4])
    return clamp(float(np.mean(diff) / 48.0))


def thumbnail(frame_rgb: np.ndarray) -> str:
    preview = cv2.resize(frame_rgb, (220, 124), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def analyze_video_file(video_path: Path, source_url: str) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=422, detail="OpenCV could not open the downloaded video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_total / fps if frame_total else 0
    if duration <= 0:
        raise HTTPException(status_code=422, detail="The video duration could not be read.")
    if duration > MAX_VIDEO_SECONDS + 1:
        raise HTTPException(status_code=422, detail="Videos must be 5 minutes or shorter.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = max(6, min(FRAME_LIMIT, math.ceil(duration / 10)))
    times = [0.35 + ((duration - 0.7) * index / max(frame_count - 1, 1)) for index in range(frame_count)]
    model = get_session()
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    frame_reports: list[dict[str, Any]] = []
    probabilities: list[float] = []
    differences: list[float] = []
    previous: np.ndarray | None = None

    for index, time_value in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(time_value, 0) * 1000)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = frame_to_tensor(frame_rgb, include_crops=(index == 0))
        logits = model.run([output_name], {input_name: tensor})[0].reshape(-1)
        frame_probs = [sigmoid(float(value)) for value in logits]
        probability = float(aggregate_probabilities(frame_probs)["probability"])
        motion = frame_difference(previous, frame_rgb)
        if previous is not None:
            differences.append(motion)
        previous = frame_rgb
        probabilities.append(probability)
        frame_reports.append(
            {
                "time": time_value,
                "probability": probability,
                "motion": motion,
                "texture": texture_complexity(frame_rgb),
                "thumbnail": thumbnail(frame_rgb),
            }
        )
    cap.release()
    if not probabilities:
        raise HTTPException(status_code=422, detail="No frames could be sampled from this video.")

    assessment = aggregate_probabilities(probabilities)
    motion_mean = float(np.mean(differences)) if differences else 0.0
    motion_instability = clamp((float(np.std(differences)) if differences else 0.0) * 2.2)
    frozen_segments = sum(value < 0.018 for value in differences) / max(len(differences), 1)
    probability_jump = max([abs(probabilities[i] - probabilities[i - 1]) for i in range(1, len(probabilities))] or [0.0])
    temporal_risk = clamp((0.38 * motion_instability) + (0.32 * frozen_segments) + (0.30 * probability_jump))
    adjusted_probability = clamp((0.78 * float(assessment["probability"])) + (0.22 * temporal_risk), 0.01, 0.99)
    confidence = clamp(float(assessment["confidence"]) + (0.12 * abs(temporal_risk - 0.5)) - (0.18 * float(assessment["disagreement"])), 0.25, 0.95)
    verdict, label = classify(adjusted_probability, confidence)
    digest = hashlib.sha256(source_url.encode("utf-8")).hexdigest()

    return {
        "scan_id": str(uuid.uuid4()),
        "created_at": now_iso(),
        "media_type": "video",
        "file_name": source_url,
        "source_url": source_url,
        "verdict": verdict,
        "label": label,
        "ai_probability": adjusted_probability,
        "confidence": confidence,
        "engine": "DoubleDeep local URL backend",
        "model": "SigLIP ONNX frame detector plus server-side yt-dlp extraction",
        "runtime": "Python / ONNX Runtime CPU / OpenCV",
        "video": {
            "duration_seconds": duration,
            "width": width,
            "height": height,
            "sampled_frames": len(frame_reports),
        },
        "evidence": [
            {
                "direction": "ai" if adjusted_probability >= 0.5 else "real",
                "title": "Server-side frame analysis",
                "detail": f"{len(frame_reports)} sampled frames produced a {round(float(assessment['probability']) * 100)}% synthetic-frame likelihood before temporal adjustment.",
            },
            {
                "direction": "ai" if temporal_risk > 0.55 else "neutral",
                "title": "Temporal forensic signals",
                "detail": f"Motion instability {round(motion_instability * 100)}%, near-freeze segments {round(frozen_segments * 100)}%, frame-score jump {round(probability_jump * 100)}%.",
            },
            {
                "direction": "neutral",
                "title": "YouTube extraction",
                "detail": "The local backend resolved the URL with yt-dlp and sampled decoded frames on this computer.",
            },
        ],
        "signals": [
            {"label": "AI/deepfake likelihood", "value": adjusted_probability},
            {"label": "Camera likelihood", "value": 1 - adjusted_probability},
            {"label": "Decision confidence", "value": confidence},
            {"label": "Frame agreement", "value": float(assessment["support"])},
            {"label": "Temporal risk", "value": temporal_risk},
            {"label": "Motion instability", "value": motion_instability},
            {"label": "Average motion", "value": motion_mean},
        ],
        "frames": frame_reports,
        "sha256": digest,
        "limitations": "Server-side extraction can still fail if the platform blocks automated downloads, requires login, or changes its delivery format.",
    }


@app.post("/api/analyze-url")
def analyze_url(payload: AnalyzeUrlRequest) -> dict[str, Any]:
    url = validate_url(payload.url.strip())
    with tempfile.TemporaryDirectory(prefix="doubledeep-") as temp_dir:
        video_path = download_video(url, Path(temp_dir))
        return analyze_video_file(video_path, url)


app.mount("/", StaticFiles(directory=str(ROOT), html=True), name="site")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8787, reload=False)
