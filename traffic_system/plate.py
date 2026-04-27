from __future__ import annotations

import importlib.util
import os
import re
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np

from traffic_system.settings import OCR_LABELS, OCR_LOCAL_UTILS, OCR_MODEL_JSON, OCR_MODEL_WEIGHTS, WPOD_JSON, WPOD_WEIGHTS


@dataclass
class PlateExtractionResult:
    plate_image: np.ndarray | None
    plate_text: str | None
    source: str


def _extract_plate_candidate_contour(vehicle_crop: np.ndarray | None) -> np.ndarray | None:
    if vehicle_crop is None or vehicle_crop.size == 0:
        return None

    gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(filtered, 30, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    best_crop = None
    best_area = 0
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) != 4:
            continue

        x, y, w, h = cv2.boundingRect(polygon)
        if h == 0:
            continue

        aspect_ratio = w / float(h)
        area = w * h
        if 2.0 <= aspect_ratio <= 6.5 and area > best_area:
            best_area = area
            best_crop = vehicle_crop[max(y, 0) : y + h, max(x, 0) : x + w].copy()

    return best_crop


def _extract_plate_candidates(vehicle_crop: np.ndarray | None) -> list[np.ndarray]:
    if vehicle_crop is None or vehicle_crop.size == 0:
        return []

    candidates: list[np.ndarray] = []
    contour_crop = _extract_plate_candidate_contour(vehicle_crop)
    if contour_crop is not None:
        candidates.append(contour_crop)

    height, width = vehicle_crop.shape[:2]
    lower_band = vehicle_crop[int(height * 0.45) : height, :]
    if lower_band.size > 0:
        gray = cv2.cvtColor(lower_band, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 50, 50)
        edges = cv2.Canny(filtered, 60, 180)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(max(1, h))
            area = w * h
            if 2.0 <= aspect_ratio <= 6.8 and area >= 700:
                crop = lower_band[max(0, y - 4) : y + h + 4, max(0, x - 4) : x + w + 4].copy()
                if crop.size > 0:
                    candidates.append(crop)

    center_crop = vehicle_crop[int(height * 0.5) : int(height * 0.9), int(width * 0.15) : int(width * 0.9)]
    if center_crop.size > 0:
        candidates.append(center_crop.copy())

    # Search several likely license-plate windows across the lower half of the vehicle.
    lower_start = int(height * 0.48)
    lower_end = int(height * 0.92)
    left_positions = [0.05, 0.18, 0.32, 0.48]
    widths = [0.22, 0.28, 0.34, 0.40]
    heights = [0.12, 0.16, 0.20]
    top_positions = [0.02, 0.12, 0.22, 0.32]
    lower_region = vehicle_crop[lower_start:lower_end, :]
    if lower_region.size > 0:
        region_h, region_w = lower_region.shape[:2]
        for left_ratio in left_positions:
            for width_ratio in widths:
                for height_ratio in heights:
                    for top_ratio in top_positions:
                        x = int(region_w * left_ratio)
                        y = int(region_h * top_ratio)
                        w = int(region_w * width_ratio)
                        h = int(region_h * height_ratio)
                        if x + w > region_w or y + h > region_h:
                            continue
                        aspect_ratio = w / float(max(1, h))
                        if not 2.0 <= aspect_ratio <= 6.8:
                            continue
                        crop = lower_region[y : y + h, x : x + w].copy()
                        if crop.size > 0 and crop.shape[0] >= 18 and crop.shape[1] >= 50:
                            candidates.append(crop)

    unique: list[np.ndarray] = []
    seen_shapes: set[tuple[int, int]] = set()
    for candidate in candidates:
        key = candidate.shape[:2]
        if key in seen_shapes:
            continue
        seen_shapes.add(key)
        unique.append(candidate)
    return unique


def _recognize_plate_text_tesseract(plate_image: np.ndarray | None) -> str | None:
    if plate_image is None or shutil.which("tesseract") is None:
        return None

    try:
        import pytesseract
    except ImportError:
        return None

    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh, config="--psm 7")
    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    return text or None


def _sanitize_plate_text(text: str | None) -> str | None:
    if not text:
        return None
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    return cleaned if len(cleaned) >= 5 else None


def _score_plate_text(text: str | None) -> int:
    cleaned = _sanitize_plate_text(text)
    if not cleaned:
        return 0
    score = len(cleaned)
    if any(char.isalpha() for char in cleaned):
        score += 2
    if any(char.isdigit() for char in cleaned):
        score += 2
    return score


def _sort_contours(contours: list[np.ndarray]) -> list[np.ndarray]:
    return sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])


def _segment_characters(plate_image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (333, 75))
    gray = cv2.bilateralFilter(gray, 7, 35, 35)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    characters: list[np.ndarray] = []
    plate_height = thresh.shape[0]

    for contour in _sort_contours(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if h < plate_height * 0.4 or h > plate_height * 0.96:
            continue
        if w < 6 or w > 90:
            continue

        ratio = w / float(h)
        if not 0.15 <= ratio <= 1.1:
            continue

        char = thresh[y : y + h, x : x + w]
        if char.size == 0:
            continue

        char = cv2.copyMakeBorder(char, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=0)
        characters.append(char)

    return characters


class PlateModelBackend:
    def __init__(self) -> None:
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

        from tensorflow.keras.models import model_from_json

        self.local_utils = self._load_local_utils()
        self.wpod_model = model_from_json(WPOD_JSON.read_text())
        self.wpod_model.load_weights(str(WPOD_WEIGHTS))
        self.ocr_model = model_from_json(OCR_MODEL_JSON.read_text())
        self.ocr_model.load_weights(str(OCR_MODEL_WEIGHTS))
        self.labels = np.load(OCR_LABELS, allow_pickle=True).astype(str)

    def _load_local_utils(self) -> ModuleType:
        spec = importlib.util.spec_from_file_location("traffic_system_plate_local_utils", OCR_LOCAL_UTILS)
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to load plate local_utils module.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def extract_plate(self, vehicle_crop: np.ndarray) -> np.ndarray | None:
        rgb_image = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
        _, plate_images, _, _ = self.local_utils.detect_lp(self.wpod_model, rgb_image, max_dim=608, lp_threshold=0.5)
        if not plate_images:
            return None
        plate_rgb = plate_images[0]
        if plate_rgb is None or plate_rgb.size == 0:
            return None
        plate_rgb = np.clip(plate_rgb * 255.0, 0, 255).astype(np.uint8) if plate_rgb.max() <= 1.0 else plate_rgb.astype(np.uint8)
        return cv2.cvtColor(plate_rgb, cv2.COLOR_RGB2BGR)

    def recognize_text(self, plate_image: np.ndarray) -> str | None:
        grayscale = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        variants = [
            plate_image,
            cv2.cvtColor(cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7), cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(cv2.equalizeHist(grayscale), cv2.COLOR_GRAY2BGR),
        ]

        best_text = None
        best_score = 0
        for variant in variants:
            characters = _segment_characters(variant)
            if not characters:
                continue

            predictions: list[str] = []
            for character in characters:
                resized = cv2.resize(character, (80, 80))
                normalized = resized.astype("float32") / 255.0
                stacked = np.stack((normalized,) * 3, axis=-1)
                scores = self.ocr_model.predict(stacked[np.newaxis, :], verbose=0)
                label_index = int(np.argmax(scores))
                predictions.append(self.labels[label_index])

            text = _sanitize_plate_text("".join(predictions))
            score = _score_plate_text(text)
            if text and score > best_score:
                best_text = text
                best_score = score
        return best_text


@lru_cache(maxsize=1)
def _get_plate_backend() -> PlateModelBackend | None:
    required = [WPOD_JSON, WPOD_WEIGHTS, OCR_MODEL_JSON, OCR_MODEL_WEIGHTS, OCR_LABELS, OCR_LOCAL_UTILS]
    if not all(path.exists() for path in required):
        return None

    try:
        return PlateModelBackend()
    except Exception:
        return None


def extract_plate_details(vehicle_crop: np.ndarray | None) -> PlateExtractionResult:
    if vehicle_crop is None or vehicle_crop.size == 0:
        return PlateExtractionResult(plate_image=None, plate_text=None, source="none")

    backend = _get_plate_backend()
    if backend is not None:
        try:
            plate_image = backend.extract_plate(vehicle_crop)
            if plate_image is not None:
                plate_text = backend.recognize_text(plate_image)
                if plate_text:
                    return PlateExtractionResult(plate_image=plate_image, plate_text=plate_text, source="wpod-ocr")
                candidates = [plate_image, *_extract_plate_candidates(vehicle_crop)]
                best_candidate = None
                best_text = None
                best_score = 0
                for candidate in candidates:
                    candidate_text = backend.recognize_text(candidate)
                    score = _score_plate_text(candidate_text)
                    if score > best_score:
                        best_candidate = candidate
                        best_text = candidate_text
                        best_score = score
                if best_candidate is not None and best_text:
                    return PlateExtractionResult(plate_image=best_candidate, plate_text=best_text, source="wpod-ocr-fallback")
        except Exception:
            pass

    best_tesseract_candidate = None
    best_tesseract_text = None
    best_tesseract_score = 0
    for candidate in _extract_plate_candidates(vehicle_crop):
        plate_text = _sanitize_plate_text(_recognize_plate_text_tesseract(candidate))
        score = _score_plate_text(plate_text)
        if score > best_tesseract_score:
            best_tesseract_candidate = candidate
            best_tesseract_text = plate_text
            best_tesseract_score = score
    if best_tesseract_candidate is not None and best_tesseract_text:
        return PlateExtractionResult(plate_image=best_tesseract_candidate, plate_text=best_tesseract_text, source="contour+tesseract")

    candidates = _extract_plate_candidates(vehicle_crop)
    plate_image = candidates[0] if candidates else None
    return PlateExtractionResult(plate_image=plate_image, plate_text=None, source="contour-only")


def extract_plate_candidate(vehicle_crop: np.ndarray | None) -> np.ndarray | None:
    return extract_plate_details(vehicle_crop).plate_image


def recognize_plate_text(plate_image: np.ndarray | None) -> str | None:
    backend = _get_plate_backend()
    if backend is not None and plate_image is not None:
        try:
            text = backend.recognize_text(plate_image)
            if text:
                return text
        except Exception:
            pass
    return _recognize_plate_text_tesseract(plate_image)


def plate_support_status() -> str:
    if _get_plate_backend() is not None:
        return "Plate OCR available via local WPOD-NET and character-recognition models."

    if shutil.which("tesseract") is not None:
        try:
            import pytesseract  # noqa: F401
            return "Plate OCR available via contour extraction and Tesseract fallback."
        except ImportError:
            pass

    return "Plate OCR limited: model backend unavailable, using contour-based plate extraction."
