from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.court import (
    DEFAULT_COURT_SIZE,
    NBA_CORNER_THREE_Y_OFFSET_FT,
    NBA_HALF_COURT_LENGTH_FT,
    NBA_HOOP_FROM_BASELINE_FT,
    NBA_THREE_POINT_RADIUS_FT,
    top_down_court,
)
from src.homography import estimate_homography


THREE_POINT_TOP_FROM_BASELINE_FT = NBA_HOOP_FROM_BASELINE_FT + NBA_THREE_POINT_RADIUS_FT
THREE_POINT_ARC_START_FROM_BASELINE_FT = NBA_HOOP_FROM_BASELINE_FT + (
    NBA_THREE_POINT_RADIUS_FT**2 - NBA_CORNER_THREE_Y_OFFSET_FT**2
) ** 0.5
RIGHT_TOP_OF_ARC_X_FT = NBA_HALF_COURT_LENGTH_FT - THREE_POINT_TOP_FROM_BASELINE_FT
LEFT_TOP_OF_ARC_X_FT = THREE_POINT_TOP_FROM_BASELINE_FT
RIGHT_CORNER_THREE_X_FT = NBA_HALF_COURT_LENGTH_FT - THREE_POINT_ARC_START_FROM_BASELINE_FT
LEFT_CORNER_THREE_X_FT = THREE_POINT_ARC_START_FROM_BASELINE_FT


EXTENDED_LANDMARKS_RIGHT = [
    ("midcourt near sideline", (0.0, 50.0)),
    ("baseline near corner", (47.0, 50.0)),
    ("baseline far corner", (47.0, 0.0)),
    ("midcourt far sideline", (0.0, 0.0)),
    ("near lane baseline corner", (47.0, 33.0)),
    ("far lane baseline corner", (47.0, 17.0)),
    ("near elbow / FT line lane", (28.0, 33.0)),
    ("far elbow / FT line lane", (28.0, 17.0)),
    ("FT line center", (28.0, 25.0)),
    ("top of 3PT arc", (RIGHT_TOP_OF_ARC_X_FT, 25.0)),
    ("near 3PT corner/arc break", (RIGHT_CORNER_THREE_X_FT, 47.0)),
    ("far 3PT corner/arc break", (RIGHT_CORNER_THREE_X_FT, 3.0)),
]

EXTENDED_LANDMARKS_LEFT = [
    ("midcourt near sideline", (47.0, 50.0)),
    ("baseline near corner", (0.0, 50.0)),
    ("baseline far corner", (0.0, 0.0)),
    ("midcourt far sideline", (47.0, 0.0)),
    ("near lane baseline corner", (0.0, 33.0)),
    ("far lane baseline corner", (0.0, 17.0)),
    ("near elbow / FT line lane", (19.0, 33.0)),
    ("far elbow / FT line lane", (19.0, 17.0)),
    ("FT line center", (19.0, 25.0)),
    ("top of 3PT arc", (LEFT_TOP_OF_ARC_X_FT, 25.0)),
    ("near 3PT corner/arc break", (LEFT_CORNER_THREE_X_FT, 47.0)),
    ("far 3PT corner/arc break", (LEFT_CORNER_THREE_X_FT, 3.0)),
]


def court_points_for_hoop_side(hoop_side: str) -> np.ndarray:
    return np.array([point for _label, point in landmarks_for_hoop_side(hoop_side)], dtype=np.float32)


def landmarks_for_hoop_side(hoop_side: str) -> list[tuple[str, tuple[float, float]]]:
    if hoop_side == "right":
        return EXTENDED_LANDMARKS_RIGHT
    if hoop_side == "left":
        return EXTENDED_LANDMARKS_LEFT
    raise ValueError(f"Unsupported hoop side: {hoop_side}")


def draw_marker(frame: np.ndarray, point: tuple[int, int], label: str) -> None:
    x, y = point
    cv2.drawMarker(frame, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 28, 2)
    cv2.circle(frame, (x, y), 14, (0, 0, 255), 2)
    cv2.putText(
        frame,
        label,
        (x + 12, y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def draw_instructions(frame: np.ndarray, labels: list[str], next_index: int) -> None:
    overlay = frame.copy()
    box_height = min(310, 82 + len(labels) * 20)
    cv2.rectangle(overlay, (12, 12), (760, box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    lines = [
        "Click visible landmarks in order. Press k to skip hidden points.",
        *[f"{index + 1}. {label}" for index, label in enumerate(labels)],
        "Press s to save (min 4), r reset, u undo, q quit.",
    ]
    for i, line in enumerate(lines):
        color = (255, 255, 255)
        if 1 <= i <= len(labels) and i - 1 == next_index:
            color = (0, 255, 255)
        cv2.putText(frame, line, (28, 38 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


def read_frame(video_path: Path, frame_number: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_number} from {video_path}")
    return frame


def save_calibration(
    output_path: Path,
    video_path: Path,
    frame_number: int,
    labels: list[str],
    frame_points: list[tuple[int, int]],
    court_points_ft: np.ndarray,
    hoop_side: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "video": str(video_path),
        "frame_number": frame_number,
        "hoop_side": hoop_side,
        "labels": labels,
        "frame_points": frame_points,
        "court_points_ft": court_points_ft.tolist(),
    }
    output_path.write_text(json.dumps(data, indent=2))


def save_calibration_review(
    output_path: Path,
    base_frame: np.ndarray,
    points: list[tuple[int, int]],
    court_points_ft: np.ndarray,
    hoop_side: str = "left",
) -> None:
    frame_points = np.array(points, dtype=np.float32)
    homography = estimate_homography(frame_points, court_points_ft)
    inverse_homography = np.linalg.inv(homography)
    projected_court_points = cv2.perspectiveTransform(frame_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
    expected_court_points = np.column_stack(
        [
            court_points_ft[:, 0] / 47.0 * DEFAULT_COURT_SIZE[0],
            court_points_ft[:, 1] / 50.0 * DEFAULT_COURT_SIZE[1],
        ]
    )
    errors = np.linalg.norm(projected_court_points - expected_court_points, axis=1)

    court = top_down_court(DEFAULT_COURT_SIZE, hoop_side=hoop_side)
    warped_court = cv2.warpPerspective(
        court,
        inverse_homography,
        (base_frame.shape[1], base_frame.shape[0]),
        flags=cv2.INTER_LINEAR,
    )
    mask = cv2.cvtColor(warped_court, cv2.COLOR_BGR2GRAY) > 0
    review = base_frame.copy()
    review[mask] = cv2.addWeighted(review[mask], 0.55, warped_court[mask], 0.45, 0)

    for index, point in enumerate(points):
        draw_marker(review, point, str(index + 1))

    review_path = output_path.with_name(f"{output_path.stem}_review.jpg")
    cv2.imwrite(str(review_path), review)
    report_path = output_path.with_name(f"{output_path.stem}_errors.txt")
    report_lines = [
        f"mean_error_px={float(errors.mean()):.2f}",
        f"max_error_px={float(errors.max()):.2f}",
        "",
    ]
    for index, error in enumerate(errors):
        report_lines.append(f"{index + 1}: error_px={float(error):.2f}, frame_point={points[index]}, court_ft={court_points_ft[index].tolist()}")
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Saved homography review to {review_path}")
    print(f"Saved calibration error report to {report_path}")
    print(report_lines[0])
    print(report_lines[1])


def interactive_calibration(video_path: Path, output_path: Path, frame_number: int, hoop_side: str) -> None:
    base_frame = read_frame(video_path, frame_number)
    landmarks = landmarks_for_hoop_side(hoop_side)
    labels = [label for label, _point in landmarks]
    clicked_points: list[tuple[int, int]] = []
    clicked_labels: list[str] = []
    clicked_court_points: list[tuple[float, float]] = []
    landmark_index = 0

    window_name = "Court calibration"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal landmark_index
        if event == cv2.EVENT_LBUTTONDOWN and landmark_index < len(landmarks):
            label, court_point = landmarks[landmark_index]
            clicked_points.append((x, y))
            clicked_labels.append(label)
            clicked_court_points.append(court_point)
            landmark_index += 1

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        frame = base_frame.copy()
        draw_instructions(frame, labels, landmark_index)
        for index, point in enumerate(clicked_points):
            draw_marker(frame, point, str(index + 1))
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            clicked_points.clear()
            clicked_labels.clear()
            clicked_court_points.clear()
            landmark_index = 0
        if key == ord("u"):
            if clicked_points:
                clicked_points.pop()
                clicked_labels.pop()
                clicked_court_points.pop()
                landmark_index = max(0, landmark_index - 1)
        if key == ord("k"):
            landmark_index = min(len(landmarks), landmark_index + 1)
        if key == ord("s"):
            if len(clicked_points) < 4:
                print(f"Need at least 4 points before saving; currently have {len(clicked_points)}.")
                continue
            court_points_ft = np.array(clicked_court_points, dtype=np.float32)
            save_calibration(output_path, video_path, frame_number, clicked_labels, clicked_points, court_points_ft, hoop_side)
            marked_path = output_path.with_suffix(".jpg")
            marked = base_frame.copy()
            draw_instructions(marked, labels, landmark_index)
            for index, point in enumerate(clicked_points):
                draw_marker(marked, point, str(index + 1))
            cv2.imwrite(str(marked_path), marked)
            save_calibration_review(output_path, base_frame, clicked_points, court_points_ft, hoop_side=hoop_side)
            print(f"Saved calibration to {output_path}")
            print(f"Saved marked frame to {marked_path}")
            break

    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", default=Path("calibrations/court_points.json"), type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--hoop-side", choices=["left", "right"], default="right")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    interactive_calibration(args.video, args.output, args.frame, args.hoop_side)
