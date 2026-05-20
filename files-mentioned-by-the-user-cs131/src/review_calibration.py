from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.calibrate import draw_marker, read_frame
from src.court import DEFAULT_COURT_SIZE, top_down_court
from src.homography import estimate_homography, load_calibration, load_hoop_side


def save_review(video_path: Path, calibration_path: Path, output_path: Path) -> None:
    frame = read_frame(video_path, 0)
    frame_points, court_points_ft = load_calibration(calibration_path)
    hoop_side = load_hoop_side(calibration_path)
    homography = estimate_homography(frame_points, court_points_ft)
    inverse_homography = np.linalg.inv(homography)

    court = top_down_court(DEFAULT_COURT_SIZE, hoop_side=hoop_side)
    warped_court = cv2.warpPerspective(
        court,
        inverse_homography,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
    )
    mask = cv2.cvtColor(warped_court, cv2.COLOR_BGR2GRAY) > 0
    review = frame.copy()
    review[mask] = cv2.addWeighted(review[mask], 0.55, warped_court[mask], 0.45, 0)

    for index, point in enumerate(frame_points):
        draw_marker(review, tuple(point.astype(int)), str(index + 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), review)
    print(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = args.output or args.calibration.with_name(f"{args.calibration.stem}_review.jpg")
    save_review(args.video, args.calibration, output)
