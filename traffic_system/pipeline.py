from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from traffic_system.models import Detection, YoloDetector, YoloXDetector
from traffic_system.plate import extract_plate_details, plate_support_status
from traffic_system.settings import COCO_CFG, COCO_NAMES, COCO_WEIGHTS, HELMET_CFG, HELMET_NAMES, HELMET_WEIGHTS, OUTPUT_ROOT, PLATE_WEIGHTS, YOLOX_ONNX


PERSON_LABEL = "person"
MOTORBIKE_LABEL = "motorbike"
VEHICLE_LABELS = {"car", "bus", "truck", "motorbike"}
TRAFFIC_LIGHT_LABEL = "traffic light"


@dataclass
class EngineConfig:
    detect_no_helmet: bool = True
    detect_triple_riding: bool = True
    detect_red_light: bool = True
    stop_line_ratio: float = 0.68
    confidence_threshold: float = 0.28
    nms_threshold: float = 0.30


@dataclass
class ViolationRecord:
    violation_type: str
    plate_text: str
    confidence: float
    evidence_path: Path
    plate_path: Path | None
    notes: str
    frame_index: int
    helmet_box: tuple[int, int, int, int] | None = None
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


@dataclass
class AnalysisResult:
    annotated_frame: np.ndarray
    violations: list[ViolationRecord]
    messages: list[str]
    traffic_light_state: str


class TrafficViolationEngine:
    def __init__(self) -> None:
        self.detector_name = "YOLOv3"
        if YOLOX_ONNX.exists():
            self.detector = YoloXDetector(YOLOX_ONNX)
            self.detector_name = "YOLOX"
        else:
            self.detector = YoloDetector(COCO_CFG, COCO_WEIGHTS, COCO_NAMES)
        self.plate_status = plate_support_status()
        self.helmet_model_available = HELMET_CFG.exists() and HELMET_WEIGHTS.exists() and HELMET_NAMES.exists()
        self.plate_model_available = PLATE_WEIGHTS.exists()
        self.helmet_detector = YoloDetector(HELMET_CFG, HELMET_WEIGHTS, HELMET_NAMES) if self.helmet_model_available else None
        cascade_root = Path(cv2.data.haarcascades)
        self.face_cascade = cv2.CascadeClassifier(str(cascade_root / "haarcascade_frontalface_default.xml"))
        self.alt_face_cascade = cv2.CascadeClassifier(str(cascade_root / "haarcascade_frontalface_alt2.xml"))
        self.profile_face_cascade = cv2.CascadeClassifier(str(cascade_root / "haarcascade_profileface.xml"))
        self.run_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(exist_ok=True)
        self.recent_events: dict[tuple[str, int, int], int] = {}

    def capability_messages(self) -> list[str]:
        messages = [self.plate_status]
        if self.helmet_model_available:
            messages.append("Custom helmet detector is active.")
        else:
            messages.append("Helmet model not found. Using a rider-face heuristic for no-helmet detection.")
        messages.append(f"Primary traffic detector: {self.detector_name}.")
        if not self.plate_model_available:
            messages.append("Legacy YOLO plate detector not found. Using the rebuilt OCR pipeline instead.")
        return messages

    def analyze_frame(self, frame: np.ndarray, frame_index: int, config: EngineConfig) -> AnalysisResult:
        self._analysis_frame = frame
        detections = self.detector.detect(
            frame,
            confidence_threshold=config.confidence_threshold,
            nms_threshold=config.nms_threshold,
            labels_filter={PERSON_LABEL, MOTORBIKE_LABEL, TRAFFIC_LIGHT_LABEL, *VEHICLE_LABELS},
        )
        helmet_detections = self._detect_helmets(frame)
        annotated = frame.copy()
        self._draw_detections(annotated, detections)
        self._draw_detections(annotated, helmet_detections, color=(255, 215, 0))

        stop_line_y = int(frame.shape[0] * config.stop_line_ratio)
        cv2.line(annotated, (0, stop_line_y), (frame.shape[1], stop_line_y), (30, 144, 255), 2)
        cv2.putText(
            annotated,
            "Stop line",
            (12, max(25, stop_line_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (30, 144, 255),
            2,
        )

        messages = self.capability_messages()
        violations: list[ViolationRecord] = []
        traffic_light_state = self._detect_traffic_light_state(frame, detections)

        if config.detect_triple_riding:
            violations.extend(self._detect_triple_riding(frame, annotated, detections, frame_index))

        if config.detect_red_light and traffic_light_state == "red":
            violations.extend(self._detect_red_light_jump(frame, annotated, detections, stop_line_y, frame_index))

        if config.detect_no_helmet:
            violations.extend(self._detect_no_helmet(frame, annotated, detections, helmet_detections, frame_index))
            if self.helmet_model_available:
                messages.append("No-helmet detection is using the custom helmet model with heuristic backup.")
            else:
                messages.append("No-helmet detection is using a face-on-rider heuristic because custom helmet weights are missing.")

        cv2.putText(
            annotated,
            f"Traffic light: {traffic_light_state}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255) if traffic_light_state == "red" else (0, 200, 0),
            2,
        )
        result = AnalysisResult(annotated_frame=annotated, violations=violations, messages=messages, traffic_light_state=traffic_light_state)
        self._analysis_frame = None
        return result

    def _draw_detections(self, frame: np.ndarray, detections: list[Detection], color: tuple[int, int, int] = (34, 139, 34)) -> None:
        for detection in detections:
            x, y, w, h = detection.box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{detection.label}: {detection.confidence:.2f}",
                (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

    def _detect_triple_riding(
        self,
        frame: np.ndarray,
        annotated: np.ndarray,
        detections: list[Detection],
        frame_index: int,
    ) -> list[ViolationRecord]:
        persons = [item for item in detections if item.label == PERSON_LABEL]
        motorbikes = [item for item in detections if item.label == MOTORBIKE_LABEL]
        violations: list[ViolationRecord] = []
        rider_groups = self._group_riders_by_bike(motorbikes, persons)

        for bike in motorbikes:
            bx, by, bw, bh = bike.box
            riders = rider_groups.get(id(bike), [])
            visible_head_boxes = self._find_visible_heads_for_bike(frame, bike)
            effective_rider_count = max(len(riders), len(visible_head_boxes))

            if effective_rider_count < 3:
                continue

            rough_x = int(bx / 40)
            rough_y = int(by / 40)
            cache_key = ("triple_riding", rough_x, rough_y)
            last_seen = self.recent_events.get(cache_key, -9999)
            if frame_index - last_seen < 20:
                continue
            self.recent_events[cache_key] = frame_index

            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 0, 255), 3)
            cv2.putText(
                annotated,
                "Triple riding",
                (bx, max(22, by - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            violations.append(
                self._create_violation(
                    frame=frame,
                    vehicle_box=bike.box,
                    violation_type="Triple Riding",
                    confidence=min(0.99, bike.confidence),
                    notes=f"Detected {effective_rider_count} riders on one motorbike. Person detections={len(riders)}, visible heads={len(visible_head_boxes)}.",
                    frame_index=frame_index,
                )
            )
        return violations

    def _detect_red_light_jump(
        self,
        frame: np.ndarray,
        annotated: np.ndarray,
        detections: list[Detection],
        stop_line_y: int,
        frame_index: int,
    ) -> list[ViolationRecord]:
        violations: list[ViolationRecord] = []
        for detection in detections:
            if detection.label not in VEHICLE_LABELS:
                continue

            x, y, w, h = detection.box
            vehicle_bottom = y + h
            if not (vehicle_bottom > stop_line_y and y < stop_line_y + 40):
                continue

            rough_x = int(x / 50)
            cache_key = ("red_light_jump", rough_x, int(stop_line_y / 50))
            last_seen = self.recent_events.get(cache_key, -9999)
            if frame_index - last_seen < 20:
                continue
            self.recent_events[cache_key] = frame_index

            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 69, 255), 3)
            cv2.putText(
                annotated,
                "Red light jump",
                (x, max(22, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 69, 255),
                2,
            )
            violations.append(
                self._create_violation(
                    frame=frame,
                    vehicle_box=detection.box,
                    violation_type="Red Light Jumping",
                    confidence=detection.confidence,
                    notes="Vehicle crossed the configured stop line while the detected traffic light was red.",
                    frame_index=frame_index,
                )
            )
        return violations

    def _detect_no_helmet(
        self,
        frame: np.ndarray,
        annotated: np.ndarray,
        detections: list[Detection],
        helmet_detections: list[Detection],
        frame_index: int,
    ) -> list[ViolationRecord]:
        persons = [item for item in detections if item.label == PERSON_LABEL]
        motorbikes = [item for item in detections if item.label == MOTORBIKE_LABEL]
        violations: list[ViolationRecord] = []
        rider_groups = self._group_riders_by_bike(motorbikes, persons)

        for bike in motorbikes:
            bx, by, bw, bh = bike.box
            riders = rider_groups.get(id(bike), [])
            visible_head_boxes = self._find_visible_heads_for_bike(frame, bike)
            if not riders and not visible_head_boxes:
                continue

            matched_helmet_count = 0
            unmatched_head_boxes: list[tuple[int, int, int, int]] = []
            best_unmatched_confidence = 0.0
            best_unmatched_notes = "At least one rider assigned to this bike does not have a matched helmet."

            for rider in riders:
                rx, ry, rw, rh = self._clamp_box(frame, rider.box)
                head_h = max(20, int(rh * 0.38))
                head_box = (rx, ry, rw, head_h)
                head_roi = frame[ry : ry + head_h, rx : rx + rw]
                if head_roi.size == 0:
                    continue

                matched_helmet = self._find_matching_helmet(frame, head_box, helmet_detections)
                head_visible, visibility_confidence, visibility_notes = self._head_region_is_visible(head_roi)
                if matched_helmet is not None:
                    hx, hy, hw, hh = matched_helmet.box
                    cv2.rectangle(annotated, (hx, hy), (hx + hw, hy + hh), (0, 215, 255), 3)
                    cv2.putText(
                        annotated,
                        f"Helmet: {matched_helmet.confidence:.2f}",
                        (hx, max(22, hy - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 215, 255),
                        2,
                    )
                    matched_helmet_count += 1
                    continue

                if head_visible:
                    unmatched_head_boxes.append(head_box)
                    best_unmatched_confidence = max(best_unmatched_confidence, visibility_confidence, rider.confidence)
                    best_unmatched_notes = f"No matched helmet for a visible rider head. {visibility_notes}"
                    continue

                no_helmet, helmet_confidence, notes = self._head_region_looks_unhelmeted(head_roi)
                if not no_helmet:
                    continue

                unmatched_head_boxes.append(head_box)
                best_unmatched_confidence = max(best_unmatched_confidence, helmet_confidence, rider.confidence)
                best_unmatched_notes = notes

            for head_box in visible_head_boxes:
                if any(self._intersection_over_smallest(head_box, existing_box) > 0.55 for existing_box in unmatched_head_boxes):
                    continue

                matched_helmet = self._find_matching_helmet(frame, head_box, helmet_detections)
                if matched_helmet is not None:
                    matched_helmet_count += 1
                    hx, hy, hw, hh = matched_helmet.box
                    cv2.rectangle(annotated, (hx, hy), (hx + hw, hy + hh), (0, 215, 255), 3)
                    cv2.putText(
                        annotated,
                        f"Helmet: {matched_helmet.confidence:.2f}",
                        (hx, max(22, hy - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 215, 255),
                        2,
                    )
                    continue

                unmatched_head_boxes.append(head_box)
                best_unmatched_confidence = max(best_unmatched_confidence, 0.7)
                best_unmatched_notes = "Visible rider head on the bike has no matched helmet."

            if not unmatched_head_boxes:
                continue

            rough_x = int(bx / 60)
            rough_y = int(by / 60)
            cache_key = ("no_helmet", rough_x, rough_y)
            last_seen = self.recent_events.get(cache_key, -9999)
            if frame_index - last_seen < 20:
                continue
            self.recent_events[cache_key] = frame_index

            missing_helmet_count = max(1, len(unmatched_head_boxes))
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (255, 0, 0), 3)
            for hx, hy, hw, hh in unmatched_head_boxes:
                cv2.rectangle(annotated, (hx, hy), (hx + hw, hy + hh), (255, 215, 0), 3)

            cv2.putText(
                annotated,
                f"No Helmet: riders={len(riders)} helmets={matched_helmet_count}",
                (bx, max(22, by - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 215, 0),
                2,
            )
            violations.append(
                self._create_violation(
                    frame=frame,
                    vehicle_box=bike.box,
                    violation_type="No Helmet",
                    confidence=max(0.65, best_unmatched_confidence, bike.confidence),
                    notes=(
                        f"Rule-based helmet check: bike has {max(len(riders), len(visible_head_boxes))} rider(s), "
                        f"{matched_helmet_count} matched helmet(s), and {missing_helmet_count} rider(s) without helmets. "
                        f"{best_unmatched_notes}"
                    ),
                    frame_index=frame_index,
                    helmet_box=unmatched_head_boxes[0],
                )
            )

        return violations

    def _detect_helmets(self, frame: np.ndarray) -> list[Detection]:
        if self.helmet_detector is None:
            return []
        return self.helmet_detector.detect(
            frame,
            confidence_threshold=0.25,
            nms_threshold=0.25,
            labels_filter={"Helmet"},
        )

    def _detect_traffic_light_state(self, frame: np.ndarray, detections: list[Detection]) -> str:
        traffic_lights = [item for item in detections if item.label == TRAFFIC_LIGHT_LABEL]
        if not traffic_lights:
            return "unknown"

        light = max(traffic_lights, key=lambda item: item.confidence)
        x, y, w, h = self._clamp_box(frame, light.box)
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_mask_1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        red_mask_2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
        green_mask = cv2.inRange(hsv, (36, 40, 40), (90, 255, 255))
        yellow_mask = cv2.inRange(hsv, (15, 70, 70), (35, 255, 255))

        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
        height = roi.shape[0]
        if height < 6:
            return "unknown"

        top = slice(0, max(1, height // 3))
        middle = slice(max(1, height // 3), max(2, (2 * height) // 3))
        bottom = slice(max(2, (2 * height) // 3), height)

        red_score = int(np.count_nonzero(red_mask[top, :]))
        yellow_score = int(np.count_nonzero(yellow_mask[middle, :]))
        green_score = int(np.count_nonzero(green_mask[bottom, :]))

        scores = {"red": red_score, "yellow": yellow_score, "green": green_score}
        best_state = max(scores, key=scores.get)
        sorted_scores = sorted(scores.values(), reverse=True)
        best_score = sorted_scores[0]
        second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0

        if best_score < 8:
            return "unknown"
        if best_score < second_score * 1.2:
            return "unknown"
        return best_state

    def _create_violation(
        self,
        frame: np.ndarray,
        vehicle_box: tuple[int, int, int, int],
        violation_type: str,
        confidence: float,
        notes: str,
        frame_index: int,
        helmet_box: tuple[int, int, int, int] | None = None,
    ) -> ViolationRecord:
        x, y, w, h = self._expand_vehicle_box(frame, vehicle_box)
        vehicle_crop = frame[y : y + h, x : x + w].copy()
        plate_result = extract_plate_details(vehicle_crop)
        plate_crop = plate_result.plate_image
        plate_text = plate_result.plate_text or "Plate not confidently read"

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        evidence_path = self.run_dir / f"{violation_type.lower().replace(' ', '_')}_{stamp}.jpg"
        cv2.imwrite(str(evidence_path), vehicle_crop)

        plate_path = None
        if plate_crop is not None and plate_crop.size > 0:
            plate_path = self.run_dir / f"plate_{stamp}.jpg"
            cv2.imwrite(str(plate_path), plate_crop)

        return ViolationRecord(
            violation_type=violation_type,
            plate_text=plate_text,
            confidence=confidence,
            evidence_path=evidence_path,
            plate_path=plate_path,
            notes=f"{notes} Plate source: {plate_result.source}.",
            frame_index=frame_index,
            helmet_box=helmet_box,
        )

    def _expand_vehicle_box(
        self,
        frame: np.ndarray,
        box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        x, y, w, h = box
        pad_x = int(w * 0.14)
        pad_top = int(h * 0.08)
        pad_bottom = int(h * 0.22)
        x = max(0, x - pad_x)
        y = max(0, y - pad_top)
        max_x = min(frame.shape[1], x + w + pad_x * 2)
        max_y = min(frame.shape[0], y + h + pad_bottom + pad_top)
        return x, y, max(1, max_x - x), max(1, max_y - y)

    def _group_riders_by_bike(
        self,
        bikes: list[Detection],
        persons: list[Detection],
    ) -> dict[int, list[Detection]]:
        grouped: dict[int, list[Detection]] = {id(bike): [] for bike in bikes}
        assignments: dict[int, tuple[Detection, float]] = {}

        for person in persons:
            best_bike = None
            best_score = 0.0
            for bike in bikes:
                score = self._score_rider_for_bike(bike, person)
                if score > best_score:
                    best_score = score
                    best_bike = bike

            if best_bike is not None and best_score >= 0.48:
                assignments[id(person)] = (best_bike, best_score)

        for person in persons:
            assignment = assignments.get(id(person))
            if assignment is None:
                continue
            bike, _ = assignment
            grouped[id(bike)].append(person)

        for bike in bikes:
            unique_riders: list[Detection] = []
            for rider in sorted(grouped[id(bike)], key=lambda item: item.confidence, reverse=True):
                if any(self._intersection_over_smallest(rider.box, existing.box) > 0.65 for existing in unique_riders):
                    continue
                unique_riders.append(rider)
            grouped[id(bike)] = unique_riders

        return grouped

    def _find_riders_for_bike(self, bike: Detection, persons: list[Detection]) -> list[Detection]:
        grouped = self._group_riders_by_bike([bike], persons)
        return grouped.get(id(bike), [])

    def _score_rider_for_bike(self, bike: Detection, person: Detection) -> float:
        bx, by, bw, bh = bike.box
        bike_center_x = bx + bw / 2
        bike_left = bx - bw * 0.5
        bike_right = bx + bw * 1.5
        bike_top = by - bh * 1.0
        bike_bottom = by + bh * 1.4
        px, py, pw, ph = person.box
        person_center_x = px + pw / 2
        person_bottom = py + ph
        person_right = px + pw

        horizontal_overlap = self._range_overlap_ratio(px, person_right, bike_left, bike_right)
        center_distance_ratio = abs(person_center_x - bike_center_x) / float(max(1, bw))
        inside_lane = bike_left <= person_center_x <= bike_right
        vertically_close = bike_top <= person_bottom <= bike_bottom
        torso_above_bike = py <= by + bh * 0.65
        proportion_ok = ph >= bh * 0.35
        feet_near_bike = abs(person_bottom - (by + bh)) <= max(50.0, bh * 0.65)

        if not vertically_close or not torso_above_bike or not proportion_ok:
            return 0.0

        score = 0.0
        if inside_lane:
            score += 0.35
        if feet_near_bike:
            score += 0.2
        score += min(0.25, horizontal_overlap * 0.6)
        score += max(0.0, 0.15 - center_distance_ratio * 0.12)
        score += min(0.05, person.confidence * 0.05)
        score += min(0.05, bike.confidence * 0.05)
        return score

    def _find_matching_helmet(
        self,
        frame: np.ndarray,
        head_box: tuple[int, int, int, int],
        helmets: list[Detection],
    ) -> Detection | None:
        hx, hy, hw, hh = head_box
        best_match = None
        best_overlap = 0.0
        head_center_x = hx + hw / 2
        head_center_y = hy + hh / 2
        for helmet in helmets:
            x, y, w, h = helmet.box
            overlap = self._intersection_over_smallest(head_box, helmet.box)
            helmet_center_x = x + w / 2
            helmet_center_y = y + h / 2
            inside_head = hx - hw * 0.15 <= helmet_center_x <= hx + hw * 1.15 and hy - hh * 0.2 <= helmet_center_y <= hy + hh * 1.2
            center_distance = np.hypot(helmet_center_x - head_center_x, helmet_center_y - head_center_y)
            normalized_distance = center_distance / float(max(1.0, max(hw, hh)))
            plausible_size = 0.45 <= (w / float(max(1, hw))) <= 1.7 and 0.45 <= (h / float(max(1, hh))) <= 1.9
            helmet_like, helmet_visual_score = self._helmet_region_looks_realistic(frame, (x, y, w, h))

            if plausible_size and helmet_like and (inside_head or overlap > 0.18):
                score = max(overlap, helmet.confidence, helmet_visual_score) - normalized_distance * 0.12
                if score > best_overlap:
                    best_overlap = score
                    best_match = helmet
        return best_match

    def _intersection_over_smallest(
        self,
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        intersection = inter_w * inter_h
        if intersection == 0:
            return 0.0
        smallest = min(aw * ah, bw * bh)
        return intersection / float(max(1, smallest))

    def _head_region_looks_unhelmeted(self, head_roi: np.ndarray) -> tuple[bool, float, str]:
        gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)

        total_faces = self._count_face_hits(gray)

        skin_mask_1 = cv2.inRange(hsv, (0, 30, 60), (20, 180, 255))
        skin_mask_2 = cv2.inRange(hsv, (160, 30, 60), (180, 180, 255))
        skin_mask = cv2.bitwise_or(skin_mask_1, skin_mask_2)
        skin_ratio = float(np.count_nonzero(skin_mask)) / float(skin_mask.size)

        bright_ratio = float(np.count_nonzero(gray > 110)) / float(gray.size)
        edge_ratio = float(np.count_nonzero(cv2.Canny(gray, 70, 140))) / float(gray.size)
        face_score = 0.45 if total_faces > 0 else 0.0
        skin_score = min(0.35, skin_ratio * 1.4)
        bright_score = min(0.1, bright_ratio * 0.2)
        edge_score = min(0.1, edge_ratio * 0.8)
        confidence = min(0.99, face_score + skin_score + bright_score + edge_score)

        if total_faces > 0 and skin_ratio > 0.08:
            return (
                True,
                max(confidence, 0.7),
                "Heuristic detected a visible rider face and exposed skin in the head region, suggesting no helmet.",
            )

        if skin_ratio > 0.18 and bright_ratio > 0.35 and edge_ratio > 0.04:
            return (
                True,
                max(confidence, 0.62),
                "Heuristic detected a large exposed-skin region around the rider's head, suggesting no helmet.",
            )

        return False, confidence, ""

    def _find_visible_heads_for_bike(
        self,
        frame: np.ndarray,
        bike: Detection,
    ) -> list[tuple[int, int, int, int]]:
        bx, by, bw, bh = bike.box
        roi_x = max(0, int(bx - bw * 0.2))
        roi_y = max(0, int(by - bh * 1.4))
        roi_w = min(frame.shape[1] - roi_x, int(bw * 1.4))
        roi_h = min(frame.shape[0] - roi_y, int(bh * 1.2))
        if roi_w <= 0 or roi_h <= 0:
            return []

        roi = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        head_boxes: list[tuple[int, int, int, int]] = []
        for cascade in (self.face_cascade, self.alt_face_cascade, self.profile_face_cascade):
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=3,
                minSize=(18, 18),
            )
            for fx, fy, fw, fh in faces:
                self._append_head_box(head_boxes, bx, by, bw, bh, roi_x, roi_y, (int(fx), int(fy), int(fw), int(fh)))

        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(18, int(bw * 0.18)),
            param1=80,
            param2=18,
            minRadius=max(10, int(min(bw, bh) * 0.06)),
            maxRadius=max(18, int(min(bw, bh) * 0.18)),
        )
        if circles is not None:
            for cx, cy, radius in np.round(circles[0]).astype(int):
                self._append_head_box(
                    head_boxes,
                    bx,
                    by,
                    bw,
                    bh,
                    roi_x,
                    roi_y,
                    (int(cx - radius), int(cy - radius), int(radius * 2), int(radius * 2)),
                )

        return head_boxes

    def _head_region_is_visible(self, head_roi: np.ndarray) -> tuple[bool, float, str]:
        gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)

        total_faces = self._count_face_hits(gray)
        skin_mask_1 = cv2.inRange(hsv, (0, 30, 60), (20, 180, 255))
        skin_mask_2 = cv2.inRange(hsv, (160, 30, 60), (180, 180, 255))
        skin_mask = cv2.bitwise_or(skin_mask_1, skin_mask_2)
        skin_ratio = float(np.count_nonzero(skin_mask)) / float(max(1, skin_mask.size))
        edge_ratio = float(np.count_nonzero(cv2.Canny(gray, 70, 140))) / float(max(1, gray.size))
        contrast = float(np.std(gray)) / 255.0

        confidence = 0.0
        if total_faces > 0:
            confidence += 0.45
        confidence += min(0.3, skin_ratio * 1.4)
        confidence += min(0.15, edge_ratio * 1.2)
        confidence += min(0.1, contrast * 0.5)
        confidence = min(0.99, confidence)

        visible = total_faces > 0 or (edge_ratio > 0.08 and contrast > 0.16)
        notes = f"Head visibility score={confidence:.2f}, skin ratio={skin_ratio:.2f}."
        return visible, confidence, notes

    def _helmet_region_looks_realistic(
        self,
        frame: np.ndarray,
        helmet_box: tuple[int, int, int, int],
    ) -> tuple[bool, float]:
        x, y, w, h = helmet_box
        if w <= 0 or h <= 0:
            return False, 0.0

        x, y, w, h = self._clamp_box(frame, helmet_box)
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            return False, 0.0

        score = 0.0
        aspect_ratio = w / float(max(1, h))
        if 0.7 <= aspect_ratio <= 1.8:
            score += 0.2
        if h >= 16 and w >= 16:
            score += 0.15

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        saturation = float(np.mean(hsv[:, :, 1])) / 255.0
        brightness = float(np.mean(hsv[:, :, 2])) / 255.0
        edge_ratio = float(np.count_nonzero(cv2.Canny(gray, 80, 160))) / float(max(1, gray.size))
        skin_mask_1 = cv2.inRange(hsv, (0, 30, 60), (20, 180, 255))
        skin_mask_2 = cv2.inRange(hsv, (160, 30, 60), (180, 180, 255))
        skin_ratio = float(np.count_nonzero(cv2.bitwise_or(skin_mask_1, skin_mask_2))) / float(max(1, gray.size))

        if saturation > 0.16:
            score += 0.2
        if 0.15 <= brightness <= 0.85:
            score += 0.1
        if edge_ratio > 0.04:
            score += 0.1
        if skin_ratio < 0.18:
            score += 0.2

        return score >= 0.45, min(0.99, score)

    def _append_head_box(
        self,
        head_boxes: list[tuple[int, int, int, int]],
        bx: int,
        by: int,
        bw: int,
        bh: int,
        roi_x: int,
        roi_y: int,
        local_box: tuple[int, int, int, int],
    ) -> None:
        fx, fy, fw, fh = local_box
        head_box = (roi_x + fx, roi_y + fy, fw, fh)
        head_center_x = head_box[0] + head_box[2] / 2
        head_bottom = head_box[1] + head_box[3]
        valid_position = bx - bw * 0.2 <= head_center_x <= bx + bw * 1.2 and head_bottom <= by + bh * 0.62
        plausible_size = bw * 0.08 <= head_box[2] <= bw * 0.5 and bh * 0.08 <= head_box[3] <= bh * 0.55
        if not valid_position or not plausible_size:
            return
        if not self._head_candidate_looks_real(head_box):
            return
        if any(self._intersection_over_smallest(head_box, existing_box) > 0.45 for existing_box in head_boxes):
            return
        head_boxes.append(head_box)

    def _count_face_hits(self, gray: np.ndarray) -> int:
        total = 0
        for cascade in (self.face_cascade, self.alt_face_cascade, self.profile_face_cascade):
            total += len(
                cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.08,
                    minNeighbors=3,
                    minSize=(18, 18),
                )
            )
        return total

    def _head_candidate_looks_real(self, head_box: tuple[int, int, int, int]) -> bool:
        x, y, w, h = head_box
        if w < 14 or h < 14:
            return False
        frame = getattr(self, "_analysis_frame", None)
        if frame is None:
            return True
        x, y, w, h = self._clamp_box(frame, head_box)
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            return False
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        edge_ratio = float(np.count_nonzero(cv2.Canny(gray, 70, 140))) / float(max(1, gray.size))
        contrast = float(np.std(gray)) / 255.0
        skin_mask_1 = cv2.inRange(hsv, (0, 30, 60), (20, 180, 255))
        skin_mask_2 = cv2.inRange(hsv, (160, 30, 60), (180, 180, 255))
        skin_ratio = float(np.count_nonzero(cv2.bitwise_or(skin_mask_1, skin_mask_2))) / float(max(1, gray.size))
        face_hits = self._count_face_hits(gray)
        if face_hits > 0:
            return True
        return edge_ratio > 0.09 and contrast > 0.12 and skin_ratio < 0.55

    def _range_overlap_ratio(self, start_a: float, end_a: float, start_b: float, end_b: float) -> float:
        overlap = max(0.0, min(end_a, end_b) - max(start_a, start_b))
        span = max(1.0, min(end_a - start_a, end_b - start_b))
        return overlap / span

    def _clamp_box(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, w, h = box
        x = max(0, x)
        y = max(0, y)
        w = max(1, min(w, frame.shape[1] - x))
        h = max(1, min(h, frame.shape[0] - y))
        return x, y, w, h
