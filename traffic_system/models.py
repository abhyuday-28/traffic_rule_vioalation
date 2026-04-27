from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]


class YoloDetector:
    def __init__(self, cfg_path: Path, weights_path: Path, labels_path: Path) -> None:
        missing = [str(path.name) for path in (cfg_path, weights_path, labels_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing model files: {', '.join(missing)}")

        self.labels = labels_path.read_text().strip().splitlines()
        self.net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
        self.output_layers = self._resolve_output_layers()

    def _resolve_output_layers(self) -> list[str]:
        layer_names = self.net.getLayerNames()
        unconnected = self.net.getUnconnectedOutLayers()
        flat_indices = np.array(unconnected).flatten().tolist()
        return [layer_names[index - 1] for index in flat_indices]

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        labels_filter: set[str] | None = None,
    ) -> list[Detection]:
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (608, 608), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence < confidence_threshold:
                    continue

                label = self.labels[class_id]
                if labels_filter and label not in labels_filter:
                    continue

                box = detection[0:4] * np.array([width, height, width, height])
                center_x, center_y, box_width, box_height = box.astype("int")
                left = int(center_x - (box_width / 2))
                top = int(center_y - (box_height / 2))

                boxes.append([left, top, int(box_width), int(box_height)])
                confidences.append(confidence)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        detections: list[Detection] = []
        if len(indices) == 0:
            return detections

        for index in np.array(indices).flatten():
            left, top, box_width, box_height = boxes[index]
            detections.append(
                Detection(
                    label=self.labels[class_ids[index]],
                    confidence=confidences[index],
                    box=(left, top, box_width, box_height),
                )
            )
        return detections


class YoloXDetector:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path.name}")

        self.net = cv2.dnn.readNet(str(model_path))
        self.input_size = (640, 640)
        self.strides = [8, 16, 32]
        self.labels = COCO_CLASSES
        self._generate_anchors()

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.28,
        nms_threshold: float = 0.3,
        labels_filter: set[str] | None = None,
    ) -> list[Detection]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        letterboxed, scale = self._letterbox(rgb, self.input_size)
        blob = np.transpose(letterboxed, (2, 0, 1))[np.newaxis, :, :, :]
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        return self._postprocess(outputs[0], frame.shape[:2], scale, confidence_threshold, nms_threshold, labels_filter)

    def _letterbox(self, image: np.ndarray, target_size: tuple[int, int]) -> tuple[np.ndarray, float]:
        padded = np.ones((target_size[0], target_size[1], 3), dtype=np.float32) * 114.0
        ratio = min(target_size[0] / image.shape[0], target_size[1] / image.shape[1])
        resized = cv2.resize(
            image,
            (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        return padded, ratio

    def _generate_anchors(self) -> None:
        grids = []
        expanded_strides = []
        hsizes = [self.input_size[0] // stride for stride in self.strides]
        wsizes = [self.input_size[1] // stride for stride in self.strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, self.strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        self.grids = np.concatenate(grids, axis=1)
        self.expanded_strides = np.concatenate(expanded_strides, axis=1)

    def _postprocess(
        self,
        outputs: np.ndarray,
        original_shape: tuple[int, int],
        scale: float,
        confidence_threshold: float,
        nms_threshold: float,
        labels_filter: set[str] | None,
    ) -> list[Detection]:
        dets = outputs[0]
        dets[:, :2] = (dets[:, :2] + self.grids) * self.expanded_strides
        dets[:, 2:4] = np.exp(dets[:, 2:4]) * self.expanded_strides

        boxes = dets[:, :4]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0

        scores = dets[:, 4:5] * dets[:, 5:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.amax(scores, axis=1)

        filtered_boxes: list[list[float]] = []
        filtered_scores: list[float] = []
        filtered_class_ids: list[int] = []
        for box, score, class_id in zip(boxes_xyxy, confidences, class_ids):
            label = self._normalize_label(self.labels[int(class_id)])
            if score < confidence_threshold:
                continue
            if labels_filter and label not in labels_filter:
                continue
            filtered_boxes.append(box.tolist())
            filtered_scores.append(float(score))
            filtered_class_ids.append(int(class_id))

        if not filtered_boxes:
            return []

        indices = cv2.dnn.NMSBoxes(filtered_boxes, filtered_scores, confidence_threshold, nms_threshold)
        detections: list[Detection] = []
        if len(indices) == 0:
            return detections

        height, width = original_shape
        for index in np.array(indices).flatten():
            x1, y1, x2, y2 = np.array(filtered_boxes[index], dtype=np.float32) / scale
            x1 = float(np.clip(x1, 0, width - 1))
            y1 = float(np.clip(y1, 0, height - 1))
            x2 = float(np.clip(x2, 0, width - 1))
            y2 = float(np.clip(y2, 0, height - 1))
            label = self._normalize_label(self.labels[filtered_class_ids[index]])
            detections.append(
                Detection(
                    label=label,
                    confidence=filtered_scores[index],
                    box=(int(x1), int(y1), max(1, int(x2 - x1)), max(1, int(y2 - y1))),
                )
            )
        return detections

    def _normalize_label(self, label: str) -> str:
        return "motorbike" if label == "motorcycle" else label
