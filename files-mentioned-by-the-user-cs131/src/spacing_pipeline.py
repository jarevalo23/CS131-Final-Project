from __future__ import annotations

import csv
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.court import DEFAULT_COURT_SIZE, hoop_pixel_x, hoop_pixel_y
from src.detector import PlayerDetector, foot_points_from_boxes
from src.dynamic_homography import DynamicHomographyTracker
from src.homography import estimate_homography, load_calibration, load_hoop_side, project_points
from src.smoothing import PointSmoother, TrackSmoother
from src.team_classifier import JerseyTeamClassifier
from src.visualize import combine_views, draw_camera_view, draw_top_down, in_court_mask, orient_display_points


def compress_from_center_axis(
    projected_points: np.ndarray,
    court_size: tuple[int, int],
    compression_factor: float,
) -> np.ndarray:
    if compression_factor == 1.0 or len(projected_points) == 0:
        return projected_points
    compressed = projected_points.copy()
    center_y = court_size[1] / 2
    compressed[:, 1] = center_y + (compressed[:, 1] - center_y) * compression_factor
    return compressed


def parse_id_list(value: str | None) -> set[int] | None:
    if value is None or value.strip() == "":
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def write_rows(csv_path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def center_x(box: np.ndarray) -> float:
    return float((box[0] + box[2]) / 2)


def center_y(box: np.ndarray) -> float:
    return float((box[1] + box[3]) / 2)


def box_centers(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0, 2), dtype=np.float32)
    x = (boxes[:, 0] + boxes[:, 2]) / 2
    y = (boxes[:, 1] + boxes[:, 3]) / 2
    return np.column_stack([x, y]).astype(np.float32)


def filter_player_detections(
    boxes: np.ndarray,
    track_ids: np.ndarray,
    confidences: np.ndarray,
    min_box_height: float,
    duplicate_iou_threshold: float,
    max_players: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return boxes, track_ids, confidences

    heights = boxes[:, 3] - boxes[:, 1]
    keep = heights >= min_box_height
    boxes = boxes[keep]
    track_ids = track_ids[keep]
    confidences = confidences[keep]
    if len(boxes) == 0:
        return boxes, track_ids, confidences

    keep_indices = non_max_suppression(boxes, confidences, duplicate_iou_threshold)
    boxes = boxes[keep_indices]
    track_ids = track_ids[keep_indices]
    confidences = confidences[keep_indices]

    if max_players is not None and max_players > 0 and len(boxes) > max_players:
        order = np.argsort(-confidences)[:max_players]
        boxes = boxes[order]
        track_ids = track_ids[order]
        confidences = confidences[order]

    return boxes, track_ids, confidences


def non_max_suppression(boxes: np.ndarray, confidences: np.ndarray, iou_threshold: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int32)
    if iou_threshold <= 0:
        return np.arange(len(boxes), dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(-confidences)
    keep: list[int] = []

    while len(order) > 0:
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        intersections = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        unions = areas[current] + areas[rest] - intersections
        ious = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)
        order = rest[ious <= iou_threshold]

    return np.array(keep, dtype=np.int32)


def detect_ball(
    detector: PlayerDetector,
    frame: np.ndarray,
    ball_class_ids: set[int] | None,
    ball_confidence: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_ids = ball_class_ids or detector.class_ids_named({"ball", "basketball"})
    if not class_ids:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    original_confidence = detector.confidence
    if ball_confidence is not None:
        detector.confidence = ball_confidence
    try:
        detections = detector.detect_classes(frame, class_ids)
    finally:
        detector.confidence = original_confidence
    if len(detections.boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return detections.boxes, box_centers(detections.boxes), detections.confidences


def choose_ball_candidate(
    projected_ball: np.ndarray,
    ball_boxes: np.ndarray,
    ball_points: np.ndarray,
    confidences: np.ndarray,
    previous_position: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(projected_ball) == 0:
        return projected_ball, ball_boxes, ball_points
    if previous_position is None:
        best_index = int(np.argmax(confidences))
    else:
        distances = np.linalg.norm(projected_ball - previous_position, axis=1)
        scores = distances - 160.0 * confidences
        best_index = int(np.argmin(scores))
    return projected_ball[[best_index]], ball_boxes[[best_index]], ball_points[[best_index]]


def infer_hoop_side_from_frame(detector: PlayerDetector, frame: np.ndarray, hoop_class_ids: set[int] | None) -> str | None:
    class_ids = hoop_class_ids or detector.class_ids_named({"hoop", "basket", "rim"})
    if not class_ids:
        return None
    detections = detector.detect_classes(frame, class_ids)
    if len(detections.boxes) == 0:
        return None
    best_index = int(np.argmax(detections.confidences))
    hoop_center_x = center_x(detections.boxes[best_index])
    return "right" if hoop_center_x >= frame.shape[1] / 2 else "left"


def should_flip_axis_from_hoop(
    detector: PlayerDetector,
    frame: np.ndarray,
    projected_points: np.ndarray,
    foot_points: np.ndarray,
    hoop_side: str,
    hoop_class_ids: set[int] | None,
    axis: str,
) -> bool | None:
    class_ids = hoop_class_ids or detector.class_ids_named({"hoop", "basket", "rim"})
    if not class_ids or len(projected_points) == 0:
        return None
    detections = detector.detect_classes(frame, class_ids)
    if len(detections.boxes) == 0:
        return None

    best_index = int(np.argmax(detections.confidences))
    if axis == "x":
        frame_hoop_coordinate = center_x(detections.boxes[best_index])
        display_hoop_coordinate = hoop_pixel_x(DEFAULT_COURT_SIZE, hoop_side)
        frame_index = 0
        court_index = 0
        court_extent = DEFAULT_COURT_SIZE[0]
    elif axis == "y":
        frame_hoop_coordinate = center_y(detections.boxes[best_index])
        display_hoop_coordinate = hoop_pixel_y(DEFAULT_COURT_SIZE)
        frame_index = 1
        court_index = 1
        court_extent = DEFAULT_COURT_SIZE[1]
    else:
        raise ValueError(f"Unsupported axis: {axis}")

    unflipped_score = 0
    flipped_score = 0
    for foot_point, projected_point in zip(foot_points, projected_points):
        frame_sign = np.sign(float(foot_point[frame_index]) - frame_hoop_coordinate)
        if frame_sign == 0:
            continue
        unflipped_sign = np.sign(float(projected_point[court_index]) - display_hoop_coordinate)
        flipped_coordinate = court_extent - 1 - float(projected_point[court_index])
        flipped_sign = np.sign(flipped_coordinate - display_hoop_coordinate)
        if unflipped_sign == frame_sign:
            unflipped_score += 1
        if flipped_sign == frame_sign:
            flipped_score += 1

    if unflipped_score == flipped_score:
        return None
    return flipped_score > unflipped_score


def run_spacing_steps_1_to_4(
    video_path: Path,
    calibration_path: Path,
    output_path: Path,
    csv_path: Path,
    model_path: str = "yolov8n.pt",
    confidence: float = 0.35,
    max_frames: int | None = None,
    offensive_ids: set[int] | None = None,
    player_class_ids: set[int] | None = None,
    compression_factor: float = 1.0,
    court_margin: int = 8,
    keep_outside_court: bool = False,
    flip_display_x: bool = False,
    flip_display_y: bool = False,
    auto_orient_hoop: bool = False,
    hoop_class_ids: set[int] | None = None,
    ball_class_ids: set[int] | None = None,
    detect_ball_enabled: bool = False,
    ball_confidence: float | None = None,
    player_smoothing_alpha: float = 1.0,
    ball_smoothing_alpha: float = 1.0,
    ball_max_jump: float | None = None,
    ball_max_missing: int = 0,
    ball_court_margin: int = 180,
    ball_trail_length: int = 18,
    topdown_padding: int = 120,
    ball_radius: int = 10,
    dynamic_homography: bool = False,
    classify_teams: bool = False,
    team_min_margin: float = 4.0,
    team_lock_min_votes: int = 8,
    team_lock_majority_ratio: float = 0.75,
    show_unlocked_team_votes: bool = False,
    min_player_box_height: float = 35.0,
    duplicate_iou_threshold: float = 0.65,
    max_players: int | None = None,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    detector = PlayerDetector(
        model_name=model_path,
        confidence=confidence,
        player_class_ids=player_class_ids,
    )
    frame_points, court_points_ft = load_calibration(calibration_path)
    homography = estimate_homography(frame_points, court_points_ft)
    homography_tracker = DynamicHomographyTracker(frame_points, court_points_ft) if dynamic_homography else None
    hoop_side = load_hoop_side(calibration_path)
    orientation_checked = False
    player_smoother = TrackSmoother(player_smoothing_alpha)
    ball_smoother = PointSmoother(ball_smoothing_alpha, max_jump=ball_max_jump, max_missing=ball_max_missing)
    ball_trail: deque[np.ndarray] = deque(maxlen=max(1, ball_trail_length))
    topdown_padding = max(topdown_padding, ball_court_margin)
    team_classifier = (
        JerseyTeamClassifier(
            min_margin=team_min_margin,
            lock_min_votes=team_lock_min_votes,
            lock_majority_ratio=team_lock_majority_ratio,
            show_unlocked_votes=show_unlocked_team_votes,
        )
        if classify_teams
        else None
    )

    writer = None
    rows: list[dict[str, float | int]] = []
    frame_count = 0

    while True:
        if max_frames is not None and frame_count >= max_frames:
            break
        ok, frame = capture.read()
        if not ok:
            break
        if homography_tracker is not None:
            homography = homography_tracker.update(frame)

        tracked = detector.track(frame)
        boxes = tracked.boxes
        track_ids = tracked.track_ids
        confidences = tracked.confidences
        boxes, track_ids, confidences = filter_player_detections(
            boxes=boxes,
            track_ids=track_ids,
            confidences=confidences,
            min_box_height=min_player_box_height,
            duplicate_iou_threshold=duplicate_iou_threshold,
            max_players=max_players,
        )
        if offensive_ids is not None and len(track_ids) > 0:
            offense_mask = np.array([track_id in offensive_ids for track_id in track_ids], dtype=bool)
            boxes = boxes[offense_mask]
            track_ids = track_ids[offense_mask]
            confidences = confidences[offense_mask]

        foot_points = foot_points_from_boxes(boxes)
        projected = project_points(foot_points, homography)
        if keep_outside_court:
            keep = np.ones(len(projected), dtype=bool)
        else:
            keep = in_court_mask(projected, margin=court_margin)
        boxes = boxes[keep]
        foot_points = foot_points[keep]
        track_ids = track_ids[keep]
        team_ids = (
            team_classifier.classify(frame, boxes, track_ids)
            if team_classifier is not None
            else np.full(len(boxes), -1, dtype=np.int32)
        )
        projected = compress_from_center_axis(projected[keep], (940, 1000), compression_factor)
        projected = player_smoother.smooth(track_ids, projected)

        if auto_orient_hoop and not orientation_checked:
            inferred_hoop_side = infer_hoop_side_from_frame(detector, frame, hoop_class_ids)
            if inferred_hoop_side is not None:
                hoop_side = inferred_hoop_side
            inferred_flip_x = should_flip_axis_from_hoop(
                detector,
                frame,
                projected,
                foot_points,
                hoop_side,
                hoop_class_ids,
                axis="x",
            )
            inferred_flip_y = should_flip_axis_from_hoop(
                detector,
                frame,
                projected,
                foot_points,
                hoop_side,
                hoop_class_ids,
                axis="y",
            )
            if inferred_flip_x is not None:
                flip_display_x = inferred_flip_x
            elif inferred_hoop_side == "right":
                flip_display_x = True
            if inferred_flip_y is not None:
                flip_display_y = inferred_flip_y
            orientation_checked = True

        display_points = orient_display_points(projected, flip_x=flip_display_x, flip_y=flip_display_y)
        ball_boxes = np.empty((0, 4), dtype=np.float32)
        ball_points = np.empty((0, 2), dtype=np.float32)
        ball_display_points = np.empty((0, 2), dtype=np.float32)
        if detect_ball_enabled:
            ball_boxes, ball_points, ball_confidences = detect_ball(detector, frame, ball_class_ids, ball_confidence)
            projected_ball = np.empty((0, 2), dtype=np.float32)
            ball_keep = np.zeros(len(ball_points), dtype=bool)
            if len(ball_points) > 0:
                projected_ball = project_points(ball_points, homography)
                ball_keep = in_court_mask(projected_ball, margin=-ball_court_margin)
                projected_ball = compress_from_center_axis(projected_ball[ball_keep], (940, 1000), compression_factor)
                ball_boxes = ball_boxes[ball_keep]
                ball_points = ball_points[ball_keep]
                ball_confidences = ball_confidences[ball_keep]
                projected_ball, ball_boxes, ball_points = choose_ball_candidate(
                    projected_ball,
                    ball_boxes,
                    ball_points,
                    ball_confidences,
                    ball_smoother.position,
                )
            else:
                ball_boxes = np.empty((0, 4), dtype=np.float32)
                ball_points = np.empty((0, 2), dtype=np.float32)

            projected_ball = ball_smoother.smooth(projected_ball)
            if len(projected_ball) > 0:
                ball_display_points = orient_display_points(
                    projected_ball,
                    flip_x=flip_display_x,
                    flip_y=flip_display_y,
                )
                ball_trail.append(ball_display_points[0].copy())

        for track_id, team_id, foot_point, projected_point in zip(track_ids, team_ids, foot_points, projected):
            rows.append(
                {
                    "frame": frame_count,
                    "object_type": "player",
                    "track_id": int(track_id),
                    "team_id": int(team_id),
                    "frame_x": float(foot_point[0]),
                    "frame_y": float(foot_point[1]),
                    "court_x": float(projected_point[0]),
                    "court_y": float(projected_point[1]),
                }
            )
        for ball_point, ball_display_point in zip(ball_points, ball_display_points):
            rows.append(
                {
                    "frame": frame_count,
                    "object_type": "ball",
                    "track_id": -1,
                    "team_id": -1,
                    "frame_x": float(ball_point[0]),
                    "frame_y": float(ball_point[1]),
                    "court_x": float(ball_display_point[0]),
                    "court_y": float(ball_display_point[1]),
                }
            )

        camera_view = draw_camera_view(
            frame,
            boxes,
            foot_points,
            track_ids=track_ids,
            team_ids=team_ids,
            ball_boxes=ball_boxes,
            ball_points=ball_points,
            ball_radius=ball_radius,
        )
        court_view = draw_top_down(
            display_points,
            hoop_side=hoop_side,
            track_ids=track_ids,
            team_ids=team_ids,
            ball_points=ball_display_points,
            ball_trail=list(ball_trail),
            padding=topdown_padding,
            ball_radius=ball_radius,
        )
        combined = combine_views(camera_view, court_view)

        if writer is None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (combined.shape[1], combined.shape[0]),
            )
        writer.write(combined)
        frame_count += 1

    if writer is not None:
        writer.release()
    capture.release()
    write_rows(csv_path, rows)
