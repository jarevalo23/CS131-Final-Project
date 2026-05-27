from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.detector import PlayerDetector, foot_points_from_boxes
from src.homography import estimate_homography, load_calibration, load_hoop_side, project_points
from src.spacing_pipeline import parse_id_list, run_spacing_steps_1_to_4
from src.visualize import combine_views, draw_camera_view, draw_top_down, in_court_mask, orient_display_points


def run(
    video_path: Path,
    output_path: Path,
    max_frames: int | None = None,
    calibration_path: Path | None = None,
    hoop_side: str = "right",
    court_margin: int = 8,
    flip_display_x: bool = False,
    flip_display_y: bool = False,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    ok, frame = capture.read()
    if not ok:
        raise ValueError("Video opened but no frames could be read.")

    detector = PlayerDetector()
    if calibration_path is not None:
        frame_points, court_points_ft = load_calibration(calibration_path)
        hoop_side = load_hoop_side(calibration_path)
        homography = estimate_homography(frame_points, court_points_ft)
    else:
        homography = estimate_homography()

    boxes = detector.detect(frame)
    foot_points = foot_points_from_boxes(boxes)
    projected = project_points(foot_points, homography)
    keep = in_court_mask(projected, margin=court_margin)
    display_points = orient_display_points(projected[keep], flip_x=flip_display_x, flip_y=flip_display_y)
    combined = combine_views(
        draw_camera_view(frame, boxes[keep], foot_points[keep]),
        draw_top_down(display_points, hoop_side=hoop_side),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (combined.shape[1], combined.shape[0]),
    )
    writer.write(combined)

    frame_count = 1
    while True:
        if max_frames is not None and frame_count >= max_frames:
            break
        ok, frame = capture.read()
        if not ok:
            break

        boxes = detector.detect(frame)
        foot_points = foot_points_from_boxes(boxes)
        projected = project_points(foot_points, homography)
        keep = in_court_mask(projected, margin=court_margin)
        display_points = orient_display_points(projected[keep], flip_x=flip_display_x, flip_y=flip_display_y)
        combined = combine_views(
            draw_camera_view(frame, boxes[keep], foot_points[keep]),
            draw_top_down(display_points, hoop_side=hoop_side),
        )
        writer.write(combined)
        frame_count += 1

    writer.release()
    capture.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", default=Path("outputs/annotated.mp4"), type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--court-margin", type=int, default=8)
    parser.add_argument("--keep-outside-court", action="store_true")
    parser.add_argument("--hoop-side", choices=["left", "right"], default="right")
    parser.add_argument("--flip-display-x", action="store_true")
    parser.add_argument("--flip-display-y", action="store_true")
    parser.add_argument("--article-steps-1-4", action="store_true")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--offense-ids")
    parser.add_argument("--player-class-ids")
    parser.add_argument("--hoop-class-ids")
    parser.add_argument("--ball-class-ids")
    parser.add_argument("--detect-ball", action="store_true")
    parser.add_argument("--ball-confidence", type=float, default=0.25)
    parser.add_argument("--auto-orient-hoop", action="store_true")
    parser.add_argument("--compression-factor", type=float, default=1.0)
    parser.add_argument("--player-smoothing-alpha", type=float, default=1.0)
    parser.add_argument("--ball-smoothing-alpha", type=float, default=1.0)
    parser.add_argument("--ball-max-jump", type=float)
    parser.add_argument("--ball-max-missing", type=int, default=0)
    parser.add_argument("--ball-court-margin", type=int, default=180)
    parser.add_argument("--ball-trail-length", type=int, default=18)
    parser.add_argument("--topdown-padding", type=int, default=120)
    parser.add_argument("--ball-radius", type=int, default=10)
    parser.add_argument("--dynamic-homography", action="store_true")
    parser.add_argument("--classify-teams", action="store_true")
    parser.add_argument("--team-min-margin", type=float, default=4.0)
    parser.add_argument("--team-lock-min-votes", type=int, default=8)
    parser.add_argument("--team-lock-majority-ratio", type=float, default=0.75)
    parser.add_argument("--show-unlocked-team-votes", action="store_true")
    parser.add_argument("--min-player-box-height", type=float, default=35.0)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.65)
    parser.add_argument("--max-players", type=int)
    parser.add_argument("--overlay-metrics", action="store_true")
    parser.add_argument("--no-overlay-smoothing", action="store_true")
    parser.add_argument("--overlay-smoothing-alpha", type=float, default=0.4)
    parser.add_argument("--no-overlay-persistence", action="store_true")
    parser.add_argument("--overlay-min-offense", type=int, default=4)
    parser.add_argument("--hide-ball-overlay", action="store_true")
    parser.add_argument("--csv-output", type=Path, default=Path("outputs/tracks.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.article_steps_1_4:
        if args.calibration is None:
            raise ValueError("--calibration is required for --article-steps-1-4")
        run_spacing_steps_1_to_4(
            video_path=args.video,
            calibration_path=args.calibration,
            output_path=args.output,
            csv_path=args.csv_output,
            model_path=args.model,
            confidence=args.confidence,
            max_frames=args.max_frames,
            offensive_ids=parse_id_list(args.offense_ids),
            player_class_ids=parse_id_list(args.player_class_ids),
            compression_factor=args.compression_factor,
            court_margin=args.court_margin,
            keep_outside_court=args.keep_outside_court,
            flip_display_x=args.flip_display_x,
            flip_display_y=args.flip_display_y,
            auto_orient_hoop=args.auto_orient_hoop,
            hoop_class_ids=parse_id_list(args.hoop_class_ids),
            ball_class_ids=parse_id_list(args.ball_class_ids),
            detect_ball_enabled=args.detect_ball,
            ball_confidence=args.ball_confidence,
            player_smoothing_alpha=args.player_smoothing_alpha,
            ball_smoothing_alpha=args.ball_smoothing_alpha,
            ball_max_jump=args.ball_max_jump,
            ball_max_missing=args.ball_max_missing,
            ball_court_margin=args.ball_court_margin,
            ball_trail_length=args.ball_trail_length,
            topdown_padding=args.topdown_padding,
            ball_radius=args.ball_radius,
            dynamic_homography=args.dynamic_homography,
            classify_teams=args.classify_teams,
            team_min_margin=args.team_min_margin,
            team_lock_min_votes=args.team_lock_min_votes,
            team_lock_majority_ratio=args.team_lock_majority_ratio,
            show_unlocked_team_votes=args.show_unlocked_team_votes,
            min_player_box_height=args.min_player_box_height,
            duplicate_iou_threshold=args.duplicate_iou_threshold,
            max_players=args.max_players,
            overlay_metrics=args.overlay_metrics,
            overlay_smoothing=not args.no_overlay_smoothing,
            overlay_smoothing_alpha=args.overlay_smoothing_alpha,
            overlay_persistence=not args.no_overlay_persistence,
            overlay_min_offense=args.overlay_min_offense,
            hide_ball_overlay=args.hide_ball_overlay,
        )
    else:
        run(
            args.video,
            args.output,
            args.max_frames,
            args.calibration,
            hoop_side=args.hoop_side,
            court_margin=args.court_margin,
            flip_display_x=args.flip_display_x,
            flip_display_y=args.flip_display_y,
        )
