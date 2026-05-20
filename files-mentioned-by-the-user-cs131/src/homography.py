from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.court import DEFAULT_COURT_SIZE, court_feet_to_pixels


# Replace these with points clicked from your actual calibration frame.
# Each source point is a pixel coordinate in the broadcast frame.
DEFAULT_FRAME_POINTS = np.array(
    [
        [100, 650],
        [1180, 650],
        [850, 250],
        [430, 250],
    ],
    dtype=np.float32,
)

# Matching destination points on a normalized half-court in feet.
DEFAULT_COURT_POINTS_FT = np.array(
    [
        [0, 0],
        [47, 0],
        [47, 50],
        [0, 50],
    ],
    dtype=np.float32,
)


def estimate_homography(
    frame_points: np.ndarray = DEFAULT_FRAME_POINTS,
    court_points_ft: np.ndarray = DEFAULT_COURT_POINTS_FT,
    court_size: tuple[int, int] = DEFAULT_COURT_SIZE,
    use_ransac: bool = True,
) -> np.ndarray:
    destination = court_feet_to_pixels(court_points_ft, court_size)
    method = cv2.RANSAC if use_ransac and len(frame_points) >= 5 else 0
    matrix, status = cv2.findHomography(frame_points, destination, method, 4.0)
    if matrix is None or status is None:
        raise ValueError("Could not estimate homography from the provided points.")
    return matrix


def load_calibration(calibration_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with calibration_path.open() as file:
        data = json.load(file)
    frame_points = np.array(data["frame_points"], dtype=np.float32)
    court_points_ft = np.array(data["court_points_ft"], dtype=np.float32)
    return frame_points, court_points_ft


def load_hoop_side(calibration_path: Path) -> str:
    with calibration_path.open() as file:
        data = json.load(file)
    return data.get("hoop_side", "right")


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    projected = cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), homography)
    return projected.reshape(-1, 2)
