from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackedDetections:
    boxes: np.ndarray
    track_ids: np.ndarray
    confidences: np.ndarray


@dataclass
class ClassDetections:
    boxes: np.ndarray
    class_ids: np.ndarray
    confidences: np.ndarray


class PlayerDetector:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.35,
        tracker: str = "bytetrack.yaml",
        player_class_ids: set[int] | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.confidence = confidence
        self.tracker = tracker
        self.player_class_ids = player_class_ids
        self._fallback_tracker = SimpleIoUTracker()
        try:
            import lap  # noqa: F401

            self._use_ultralytics_tracker = True
        except ModuleNotFoundError:
            self._use_ultralytics_tracker = False

    def detect(self, frame: np.ndarray) -> np.ndarray:
        result = self.model.predict(frame, conf=self.confidence, verbose=False)[0]
        boxes = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            if not self._is_player_class(class_id):
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            boxes.append([x1, y1, x2, y2])
        return np.array(boxes, dtype=np.float32).reshape(-1, 4)

    def detect_classes(self, frame: np.ndarray, class_ids: set[int] | None = None) -> ClassDetections:
        result = self.model.predict(frame, conf=self.confidence, verbose=False)[0]
        boxes = []
        detected_class_ids = []
        confidences = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_ids is not None and class_id not in class_ids:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            boxes.append([x1, y1, x2, y2])
            detected_class_ids.append(class_id)
            confidences.append(float(box.conf[0]))
        return ClassDetections(
            boxes=np.array(boxes, dtype=np.float32).reshape(-1, 4),
            class_ids=np.array(detected_class_ids, dtype=np.int32),
            confidences=np.array(confidences, dtype=np.float32),
        )

    def class_ids_named(self, names: set[str]) -> set[int]:
        model_names = getattr(self.model, "names", {})
        matches = set()
        for class_id, name in model_names.items():
            normalized = str(name).strip().lower()
            if normalized in names:
                matches.add(int(class_id))
        return matches

    def track(self, frame: np.ndarray) -> TrackedDetections:
        if not self._use_ultralytics_tracker:
            boxes = self.detect(frame)
            return self._fallback_tracker.update(boxes)
        try:
            result = self.model.track(
                frame,
                conf=self.confidence,
                persist=True,
                tracker=self.tracker,
                verbose=False,
            )[0]
        except ModuleNotFoundError as exc:
            import warnings
            warnings.warn(f"ByteTrack unavailable ({exc}); falling back to IoU tracker.")
            self._use_ultralytics_tracker = False
            boxes = self.detect(frame)
            return self._fallback_tracker.update(boxes)
        boxes = []
        track_ids = []
        confidences = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            if not self._is_player_class(class_id):
                continue
            if box.id is None:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            boxes.append([x1, y1, x2, y2])
            track_ids.append(int(box.id[0]))
            confidences.append(float(box.conf[0]))
        return TrackedDetections(
            boxes=np.array(boxes, dtype=np.float32).reshape(-1, 4),
            track_ids=np.array(track_ids, dtype=np.int32),
            confidences=np.array(confidences, dtype=np.float32),
        )

    def _is_player_class(self, class_id: int) -> bool:
        if self.player_class_ids is not None:
            return class_id in self.player_class_ids
        names = getattr(self.model, "names", {})
        name = str(names.get(class_id, "")).lower()
        if name:
            return "person" in name or "player" in name
        return class_id == 0


class SimpleIoUTracker:
    def __init__(self, iou_threshold: float = 0.25, max_missing: int = 20) -> None:
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.next_id = 1
        self.tracks: dict[int, dict[str, np.ndarray | int]] = {}

    def update(self, boxes: np.ndarray) -> TrackedDetections:
        if len(boxes) == 0:
            self._age_unmatched(set(self.tracks))
            return TrackedDetections(
                boxes=boxes,
                track_ids=np.empty((0,), dtype=np.int32),
                confidences=np.empty((0,), dtype=np.float32),
            )

        unmatched_tracks = set(self.tracks)
        assigned_ids: list[int] = []
        assigned_boxes: list[np.ndarray] = []

        for box in boxes:
            best_id = None
            best_iou = 0.0
            for track_id in unmatched_tracks:
                score = box_iou(box, self.tracks[track_id]["box"])  # type: ignore[arg-type]
                if score > best_iou:
                    best_iou = score
                    best_id = track_id

            if best_id is None or best_iou < self.iou_threshold:
                best_id = self.next_id
                self.next_id += 1
            else:
                unmatched_tracks.remove(best_id)

            self.tracks[best_id] = {"box": box.copy(), "missing": 0}
            assigned_ids.append(best_id)
            assigned_boxes.append(box)

        self._age_unmatched(unmatched_tracks)
        return TrackedDetections(
            boxes=np.array(assigned_boxes, dtype=np.float32).reshape(-1, 4),
            track_ids=np.array(assigned_ids, dtype=np.int32),
            confidences=np.ones(len(assigned_ids), dtype=np.float32),
        )

    def _age_unmatched(self, unmatched_tracks: set[int]) -> None:
        for track_id in list(unmatched_tracks):
            self.tracks[track_id]["missing"] = int(self.tracks[track_id]["missing"]) + 1
            if int(self.tracks[track_id]["missing"]) > self.max_missing:
                del self.tracks[track_id]


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def foot_points_from_boxes(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0, 2), dtype=np.float32)
    x = (boxes[:, 0] + boxes[:, 2]) / 2
    y = boxes[:, 3]
    return np.column_stack([x, y]).astype(np.float32)
