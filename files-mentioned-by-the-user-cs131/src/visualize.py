from __future__ import annotations

import cv2
import numpy as np

from src.court import DEFAULT_COURT_SIZE, top_down_court


def in_court_mask(
    projected_points: np.ndarray,
    court_size: tuple[int, int] = DEFAULT_COURT_SIZE,
    margin: int = 8,
) -> np.ndarray:
    if len(projected_points) == 0:
        return np.array([], dtype=bool)
    width, height = court_size
    return (
        (projected_points[:, 0] >= margin)
        & (projected_points[:, 0] < width - margin)
        & (projected_points[:, 1] >= margin)
        & (projected_points[:, 1] < height - margin)
    )


def orient_display_points(
    projected_points: np.ndarray,
    court_size: tuple[int, int] = DEFAULT_COURT_SIZE,
    flip_x: bool = False,
    flip_y: bool = False,
) -> np.ndarray:
    oriented = projected_points.copy()
    if flip_x and len(oriented) > 0:
        oriented[:, 0] = court_size[0] - 1 - oriented[:, 0]
    if flip_y and len(oriented) > 0:
        oriented[:, 1] = court_size[1] - 1 - oriented[:, 1]
    return oriented


def draw_camera_view(
    frame: np.ndarray,
    boxes: np.ndarray,
    foot_points: np.ndarray,
    track_ids: np.ndarray | None = None,
    ball_boxes: np.ndarray | None = None,
    ball_points: np.ndarray | None = None,
) -> np.ndarray:
    annotated = frame.copy()
    if track_ids is None:
        track_ids = np.full(len(boxes), -1, dtype=np.int32)
    for box, foot, track_id in zip(boxes, foot_points, track_ids):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(annotated, tuple(foot.astype(int)), 4, (0, 0, 255), -1)
        if track_id >= 0:
            cv2.putText(
                annotated,
                f"ID {track_id}",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
    if ball_boxes is not None:
        if ball_points is None:
            ball_points = np.empty((0, 2), dtype=np.float32)
        for box, point in zip(ball_boxes, ball_points):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 255), 2)
            cv2.circle(annotated, tuple(point.astype(int)), 5, (0, 220, 255), -1)
            cv2.putText(
                annotated,
                "ball",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
    return annotated


def draw_top_down(
    projected_points: np.ndarray,
    court_size: tuple[int, int] = DEFAULT_COURT_SIZE,
    hoop_side: str = "left",
    track_ids: np.ndarray | None = None,
    ball_points: np.ndarray | None = None,
) -> np.ndarray:
    court = top_down_court(court_size, hoop_side=hoop_side)
    if track_ids is None:
        track_ids = np.full(len(projected_points), -1, dtype=np.int32)
    for point, track_id in zip(projected_points, track_ids):
        x, y = point.astype(int)
        if 0 <= x < court_size[0] and 0 <= y < court_size[1]:
            cv2.circle(court, (x, y), 8, (30, 80, 220), -1)
            if track_id >= 0:
                cv2.putText(
                    court,
                    str(track_id),
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (30, 80, 220),
                    2,
                    cv2.LINE_AA,
                )
    if ball_points is not None:
        for point in ball_points:
            x, y = point.astype(int)
            if 0 <= x < court_size[0] and 0 <= y < court_size[1]:
                cv2.circle(court, (x, y), 7, (0, 220, 255), -1)
                cv2.circle(court, (x, y), 12, (0, 120, 220), 2)
    return court


def combine_views(camera: np.ndarray, court: np.ndarray) -> np.ndarray:
    camera_h = camera.shape[0]
    court_scaled = cv2.resize(court, (int(court.shape[1] * camera_h / court.shape[0]), camera_h))
    return np.hstack([camera, court_scaled])
