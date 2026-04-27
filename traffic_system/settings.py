from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "app_output"
OUTPUT_ROOT.mkdir(exist_ok=True)

COCO_CFG = ROOT / "cfg" / "yolov3.cfg"
COCO_WEIGHTS = ROOT / "yolov3.weights"
COCO_NAMES = ROOT / "coco.names"

HELMET_CFG = ROOT / "yolov3-obj.cfg"
HELMET_WEIGHTS = ROOT / "yolov3-obj_2400.weights"
HELMET_NAMES = ROOT / "obj.names"

PLATE_CFG = ROOT / "obj.cfg"
PLATE_WEIGHTS = ROOT / "obj_60000.weights"
PLATE_NAMES = ROOT / "obj3.names"

OCR_SOURCE_ROOT = ROOT / "_ocr_source_fresh"
WPOD_JSON = OCR_SOURCE_ROOT / "wpod-net.json"
WPOD_WEIGHTS = OCR_SOURCE_ROOT / "wpod-net.h5"
OCR_MODEL_JSON = OCR_SOURCE_ROOT / "MobileNets_character_recognition.json"
OCR_MODEL_WEIGHTS = OCR_SOURCE_ROOT / "License_character_recognition_weight.h5"
OCR_LABELS = OCR_SOURCE_ROOT / "license_character_classes.npy"
OCR_LOCAL_UTILS = OCR_SOURCE_ROOT / "local_utils.py"

DETECTOR_SOURCE_ROOT = ROOT / "_detector_source"
YOLOX_ONNX = DETECTOR_SOURCE_ROOT / "object_detection_yolox_2022nov.onnx"

SAMPLES_ROOT = ROOT / "samples"
SAMPLE_IMAGES_DIR = SAMPLES_ROOT / "images"
SAMPLE_VIDEOS_DIR = SAMPLES_ROOT / "videos"
SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SAMPLE_IMAGE = ROOT / "0.jpg"
DEFAULT_SAMPLE_VIDEO = ROOT / "om1.mp4"

SUPPORTED_IMAGE_TYPES = [("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
SUPPORTED_VIDEO_TYPES = [("Video files", "*.mp4 *.avi *.mov *.mkv")]

TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""
