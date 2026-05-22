from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from src.calibrate import read_frame
from src.court_landmarks import landmark_court_points, landmark_labels


def normalized_point(point: list[float], width: int, height: int) -> tuple[float, float]:
    return float(point[0]) / width, float(point[1]) / height


def export_one(calibration_path: Path, output_dir: Path) -> bool:
    data = json.loads(calibration_path.read_text())
    video_path = Path(data["video"])
    if not video_path.exists():
        video_path = Path.cwd() / video_path
    hoop_side = data.get("hoop_side", "right")
    labels = landmark_labels(hoop_side)
    label_to_index = {label: index for index, label in enumerate(labels)}

    frame = read_frame(video_path, int(data.get("frame_number", 0)))
    height, width = frame.shape[:2]

    keypoints = [[0.0, 0.0, 0.0] for _ in labels]
    for label, point in zip(data["labels"], data["frame_points"]):
        if label not in label_to_index:
            continue
        x, y = normalized_point(point, width, height)
        keypoints[label_to_index[label]] = [x, y, 2.0]

    if sum(1 for _x, _y, visible in keypoints if visible > 0) < 4:
        return False

    split = "train"
    image_dir = output_dir / "images" / split
    label_dir = output_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    stem = calibration_path.stem
    image_path = image_dir / f"{stem}.jpg"
    label_path = label_dir / f"{stem}.txt"
    cv2.imwrite(str(image_path), frame)

    bbox = [0.5, 0.5, 1.0, 1.0]
    flattened_keypoints = [value for keypoint in keypoints for value in keypoint]
    values = [0, *bbox, *flattened_keypoints]
    label_path.write_text(" ".join(f"{value:.8f}" if isinstance(value, float) else str(value) for value in values) + "\n")
    return True


def write_yaml(output_dir: Path, hoop_side: str) -> None:
    labels = landmark_labels(hoop_side)
    yaml = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/train",
        "",
        "names:",
        "  0: court",
        f"kpt_shape: [{len(labels)}, 3]",
        "",
        "landmark_names:",
        *[f"  {index}: {label}" for index, label in enumerate(labels)],
        "",
        "court_points_ft:",
        *[f"  {index}: [{point[0]}, {point[1]}]" for index, point in enumerate(landmark_court_points(hoop_side))],
    ]
    (output_dir / "data.yaml").write_text("\n".join(yaml) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrations", type=Path, default=Path("calibrations"))
    parser.add_argument("--output", type=Path, default=Path("court_landmark_dataset"))
    parser.add_argument("--hoop-side", choices=["left", "right"], default="right")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.clean and args.output.exists():
        shutil.rmtree(args.output)
    exported = 0
    for calibration_path in sorted(args.calibrations.glob("*.json")):
        try:
            if export_one(calibration_path, args.output):
                exported += 1
        except Exception as exc:
            print(f"Skipped {calibration_path}: {exc}")
    write_yaml(args.output, args.hoop_side)
    print(f"Exported {exported} calibration frame(s) to {args.output}")
    print(f"Dataset YAML: {args.output / 'data.yaml'}")

