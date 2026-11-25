"""Optional text and face detection. Install with `pip install reelint[vision]`.

Everything here degrades gracefully: if the libraries aren't installed,
`available()` is False and callers report the features as unknown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

MIN_TEXT_CONFIDENCE = 0.6


@dataclass(frozen=True)
class TextBox:
    text: str
    confidence: float
    height: float  # fraction of the frame height
    center_y: float  # 0 = top, 1 = bottom


@dataclass(frozen=True)
class Face:
    size: float  # face height as a fraction of the frame height
    center_y: float


def missing() -> list[str]:
    """Names of the optional packages that aren't installed."""
    out = []
    try:
        import cv2  # noqa: F401
    except ImportError:
        out.append("opencv-python-headless")
    try:
        import rapidocr  # noqa: F401
    except ImportError:
        out.append("rapidocr")
    return out


def available() -> bool:
    return not missing()


@lru_cache(maxsize=1)
def _ocr():
    from rapidocr import RapidOCR
    return RapidOCR(params={"Global.log_level": "critical"})


@lru_cache(maxsize=1)
def _faces():
    import cv2
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        raise RuntimeError(f"OpenCV face model not found at {path}; install opencv-python-headless<5")
    return detector


def read_text(rgb: np.ndarray) -> list[TextBox]:
    """Text found in an RGB uint8 frame."""
    result = _ocr()(rgb[:, :, ::-1].copy())  # RapidOCR expects OpenCV's BGR order
    if result.boxes is None or not result.txts:
        return []
    height = rgb.shape[0]
    boxes = []
    for text, score, box in zip(result.txts, result.scores, result.boxes):
        text = text.strip()
        if float(score) < MIN_TEXT_CONFIDENCE or not re.search(r"\w", text):
            continue
        ys = np.asarray(box)[:, 1]
        boxes.append(TextBox(text=text, confidence=float(score),
                             height=float(ys.max() - ys.min()) / height,
                             center_y=float(ys.mean()) / height))
    return boxes


def find_faces(rgb: np.ndarray) -> list[Face]:
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height = rgb.shape[0]
    found = _faces().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                      minSize=(max(24, height // 20),) * 2)
    return [Face(size=float(h) / height, center_y=float(y + h / 2) / height) for (x, y, w, h) in found]


def normalise(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.lower()).strip()


def first(items: list, default: Optional[float] = None):
    return items[0] if items else default
