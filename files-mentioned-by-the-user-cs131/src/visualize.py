from __future__ import annotations

import cv2
import numpy as np

from src.court import DEFAULT_COURT_SIZE, top_down_court


TEAM_COLORS = {
    -1: (160, 160, 160),
    0: (255, 80, 30),
    1: (30, 80, 230),
}


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
    team_ids: np.ndarray | None = None,
    ball_boxes: np.ndarray | None = None,
    ball_points: np.ndarray | None = None,
    ball_radius: int = 7,
) -> np.ndarray:
    annotated = frame.copy()
    if track_ids is None:
        track_ids = np.full(len(boxes), -1, dtype=np.int32)
    if team_ids is None:
        team_ids = np.full(len(boxes), -1, dtype=np.int32)
    for box, foot, track_id, team_id in zip(boxes, foot_points, track_ids, team_ids):
        color = TEAM_COLORS.get(int(team_id), TEAM_COLORS[-1])
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, tuple(foot.astype(int)), 4, (0, 0, 255), -1)
        if track_id >= 0:
            label = f"ID {track_id}"
            if team_id >= 0:
                label += f" T{team_id}"
            cv2.putText(
                annotated,
                label,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
    if ball_boxes is not None:
        if ball_points is None:
            ball_points = np.empty((0, 2), dtype=np.float32)
        for box, point in zip(ball_boxes, ball_points):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 255), 3)
            cv2.circle(annotated, tuple(point.astype(int)), ball_radius + 4, (0, 90, 220), 3)
            cv2.circle(annotated, tuple(point.astype(int)), ball_radius, (0, 220, 255), -1)
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
    team_ids: np.ndarray | None = None,
    ball_points: np.ndarray | None = None,
    ball_trail: list[np.ndarray] | None = None,
    padding: int = 0,
    ball_radius: int = 10,
) -> np.ndarray:
    base_court = top_down_court(court_size, hoop_side=hoop_side)
    if padding > 0:
        court = np.full(
            (court_size[1] + 2 * padding, court_size[0] + 2 * padding, 3),
            base_court[0, 0].tolist(),
            dtype=np.uint8,
        )
        court[padding : padding + court_size[1], padding : padding + court_size[0]] = base_court
    else:
        court = base_court
    draw_width = court_size[0] + 2 * padding
    draw_height = court_size[1] + 2 * padding
    if track_ids is None:
        track_ids = np.full(len(projected_points), -1, dtype=np.int32)
    if team_ids is None:
        team_ids = np.full(len(projected_points), -1, dtype=np.int32)
    for point, track_id, team_id in zip(projected_points, track_ids, team_ids):
        color = TEAM_COLORS.get(int(team_id), TEAM_COLORS[-1])
        x, y = (point + padding).astype(int)
        if 0 <= x < draw_width and 0 <= y < draw_height:
            cv2.circle(court, (x, y), 8, color, -1)
            if track_id >= 0:
                cv2.putText(
                    court,
                    str(track_id),
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
    if ball_trail is not None and len(ball_trail) >= 2:
        trail_points = [(point + padding).astype(int) for point in ball_trail]
        for index in range(1, len(trail_points)):
            alpha = index / max(1, len(trail_points) - 1)
            color = (0, int(120 + 100 * alpha), 255)
            thickness = max(1, int(2 + 3 * alpha))
            cv2.line(court, tuple(trail_points[index - 1]), tuple(trail_points[index]), color, thickness)
    if ball_points is not None:
        for point in ball_points:
            x, y = (point + padding).astype(int)
            if 0 <= x < draw_width and 0 <= y < draw_height:
                cv2.circle(court, (x, y), ball_radius, (0, 220, 255), -1)
                cv2.circle(court, (x, y), ball_radius + 7, (0, 100, 230), 3)
    return court


def combine_views(camera: np.ndarray, court: np.ndarray) -> np.ndarray:
    camera_h = camera.shape[0]
    court_scaled = cv2.resize(court, (int(court.shape[1] * camera_h / court.shape[0]), camera_h))
    return np.hstack([camera, court_scaled])
