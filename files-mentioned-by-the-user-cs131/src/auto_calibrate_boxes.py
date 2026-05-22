from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.calibrate import read_frame, save_calibration_review


# These aliases cover common class names in court-landmark detectors. If the
# Roboflow export uses different names, run with --print-classes and add aliases.
RIGHT_HOOP_CLASS_TO_LANDMARK = {
    "right bottom left paint": ("near elbow / FT line lane", (28.0, 33.0)),
    "right bottom right paint": ("near lane baseline corner", (47.0, 33.0)),
    "right top left paint": ("far elbow / FT line lane", (28.0, 17.0)),
    "right top right paint": ("far lane baseline corner", (47.0, 17.0)),
    "paint corner bottom left": ("near lane baseline corner", (47.0, 33.0)),
    "paint corner top left": ("far lane baseline corner", (47.0, 17.0)),
    "paint corner bottom right": ("near elbow / FT line lane", (28.0, 33.0)),
    "paint corner top right": ("far elbow / FT line lane", (28.0, 17.0)),
    "paint center": ("FT line center", (28.0, 25.0)),
    "middle": ("top of 3PT arc", (18.0, 25.0)),
    "corner three bottom": ("near 3PT corner/arc break", (32.80223491591336, 47.0)),
    "corner three top": ("far 3PT corner/arc break", (32.80223491591336, 3.0)),
    "court corner bottom right": ("baseline near corner", (47.0, 50.0)),
    "court corner top right": ("baseline far corner", (47.0, 0.0)),
    "court corner bottom left": ("midcourt near sideline", (0.0, 50.0)),
    "court corner top left": ("midcourt far sideline", (0.0, 0.0)),
}


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


def swap_left_right_name(class_name: str) -> str:
    if class_name.startswith("left "):
        return "right " + class_name[len("left ") :]
    if class_name.startswith("right "):
        return "left " + class_name[len("right ") :]
    return class_name


def mirror_for_left_hoop(mapping: dict[str, tuple[str, tuple[float, float]]]) -> dict[str, tuple[str, tuple[float, float]]]:
    mirrored = {}
    for class_name, (label, point) in mapping.items():
        mirrored[swap_left_right_name(class_name)] = (label, (47.0 - point[0], point[1]))
    return mirrored


def box_center(box: np.ndarray) -> tuple[int, int]:
    return int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)


def detect_landmark_boxes(model_path: str, frame: np.ndarray, confidence: float, print_classes: bool) -> list[tuple[str, tuple[int, int], float]]:
    from ultralytics import YOLO

    model = YOLO(model_path)
    result = model.predict(frame, conf=confidence, verbose=False)[0]
    names = getattr(model, "names", {})
    if print_classes:
        print("Model classes:")
        for class_id, name in names.items():
            print(f"{class_id}: {name}")
    detections: list[tuple[str, tuple[int, int], float]] = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = normalize_name(str(names.get(class_id, class_id)))
        detections.append((class_name, box_center(box.xyxy[0].cpu().numpy()), float(box.conf[0])))
    return detections


def choose_frame_with_landmarks(
    video_path: Path,
    model_path: str,
    mapping: dict[str, tuple[str, tuple[float, float]]],
    confidence: float,
    frame_number: int,
    scan_frames: int,
    scan_step: int,
) -> tuple[int, np.ndarray, list[tuple[str, tuple[int, int], float]]]:
    best: tuple[int, int, np.ndarray, list[tuple[str, tuple[int, int], float]]] | None = None
    for offset in range(0, max(scan_frames, 1), max(scan_step, 1)):
        current_frame_number = frame_number + offset
        frame = read_frame(video_path, current_frame_number)
        detections = detect_landmark_boxes(model_path, frame, confidence, print_classes=False)
        mapped_classes = {class_name for class_name, _center, _score in detections if class_name in mapping}
        score = len(mapped_classes)
        if best is None or score > best[1]:
            best = (current_frame_number, score, frame, detections)
        if score >= 4:
            return current_frame_number, frame, detections

    assert best is not None
    return best[0], best[2], best[3]


def auto_calibrate_boxes(
    video_path: Path,
    output_path: Path,
    model_path: str,
    hoop_side: str,
    frame_number: int,
    confidence: float,
    print_classes: bool,
    scan_frames: int,
    scan_step: int,
) -> None:
    mapping = RIGHT_HOOP_CLASS_TO_LANDMARK if hoop_side == "right" else mirror_for_left_hoop(RIGHT_HOOP_CLASS_TO_LANDMARK)
    frame_number, frame, detections = choose_frame_with_landmarks(
        video_path=video_path,
        model_path=model_path,
        mapping=mapping,
        confidence=confidence,
        frame_number=frame_number,
        scan_frames=scan_frames,
        scan_step=scan_step,
    )
    if print_classes:
        detect_landmark_boxes(model_path, frame, confidence, print_classes=True)

    best_by_class: dict[str, tuple[tuple[int, int], float]] = {}
    for class_name, center, score in detections:
        if class_name not in mapping:
            continue
        previous = best_by_class.get(class_name)
        if previous is None or score > previous[1]:
            best_by_class[class_name] = (center, score)

    labels: list[str] = []
    frame_points: list[tuple[int, int]] = []
    court_points_ft: list[tuple[float, float]] = []
    for class_name, (center, _score) in best_by_class.items():
        label, court_point = mapping[class_name]
        labels.append(label)
        frame_points.append(center)
        court_points_ft.append(court_point)

    if len(frame_points) < 4:
        detected_names = sorted({name for name, _center, _score in detections})
        raise ValueError(
            f"Need at least 4 mapped court landmarks; found {len(frame_points)}. "
            f"Detected classes: {detected_names}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "video": str(video_path),
        "frame_number": frame_number,
        "hoop_side": hoop_side,
        "labels": labels,
        "frame_points": frame_points,
        "court_points_ft": court_points_ft,
        "source": "auto_calibrate_boxes",
        "model": model_path,
    }
    output_path.write_text(json.dumps(data, indent=2))
    save_calibration_review(output_path, frame, frame_points, np.array(court_points_ft, dtype=np.float32), hoop_side=hoop_side)
    print(f"Saved box-based auto calibration to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--hoop-side", choices=["left", "right"], default="right")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--print-classes", action="store_true")
    parser.add_argument("--scan-frames", type=int, default=1)
    parser.add_argument("--scan-step", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    auto_calibrate_boxes(
        video_path=args.video,
        output_path=args.output,
        model_path=args.model,
        hoop_side=args.hoop_side,
        frame_number=args.frame,
        confidence=args.confidence,
        print_classes=args.print_classes,
        scan_frames=args.scan_frames,
        scan_step=args.scan_step,
    )
