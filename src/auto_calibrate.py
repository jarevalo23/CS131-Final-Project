from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.calibrate import read_frame, save_calibration_review
from src.court_landmarks import landmark_court_points, landmark_labels


def predict_landmarks(
    model_path: str,
    frame: np.ndarray,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    from ultralytics import YOLO

    model = YOLO(model_path)
    result = model.predict(frame, conf=confidence, verbose=False)[0]
    if result.keypoints is None or len(result.keypoints) == 0:
        raise ValueError("No court landmark keypoints detected.")
    box_confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(result.keypoints))
    best_index = int(np.argmax(box_confidences))
    keypoints_xy = result.keypoints.xy[best_index].cpu().numpy()
    if result.keypoints.conf is not None:
        keypoint_confidences = result.keypoints.conf[best_index].cpu().numpy()
    else:
        keypoint_confidences = np.ones(len(keypoints_xy), dtype=np.float32)
    return keypoints_xy, keypoint_confidences


def auto_calibrate(
    video_path: Path,
    output_path: Path,
    model_path: str,
    hoop_side: str,
    frame_number: int,
    confidence: float,
    keypoint_confidence: float,
) -> None:
    frame = read_frame(video_path, frame_number)
    keypoints_xy, keypoint_confidences = predict_landmarks(model_path, frame, confidence)
    labels = landmark_labels(hoop_side)
    court_points = landmark_court_points(hoop_side)

    frame_points: list[tuple[int, int]] = []
    selected_labels: list[str] = []
    selected_court_points: list[tuple[float, float]] = []
    for index, (point, score) in enumerate(zip(keypoints_xy, keypoint_confidences)):
        if index >= len(labels) or score < keypoint_confidence:
            continue
        frame_points.append((int(point[0]), int(point[1])))
        selected_labels.append(labels[index])
        selected_court_points.append(court_points[index])

    if len(frame_points) < 4:
        raise ValueError(f"Need at least 4 detected landmarks; found {len(frame_points)}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "video": str(video_path),
        "frame_number": frame_number,
        "hoop_side": hoop_side,
        "labels": selected_labels,
        "frame_points": frame_points,
        "court_points_ft": selected_court_points,
        "source": "auto_calibrate",
        "model": model_path,
    }
    output_path.write_text(json.dumps(data, indent=2))
    save_calibration_review(
        output_path,
        frame,
        frame_points,
        np.array(selected_court_points, dtype=np.float32),
        hoop_side=hoop_side,
    )
    print(f"Saved auto calibration to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--hoop-side", choices=["left", "right"], default="right")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--keypoint-confidence", type=float, default=0.25)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    auto_calibrate(
        video_path=args.video,
        output_path=args.output,
        model_path=args.model,
        hoop_side=args.hoop_side,
        frame_number=args.frame,
        confidence=args.confidence,
        keypoint_confidence=args.keypoint_confidence,
    )
