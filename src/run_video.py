from __future__ import annotations

import argparse
from pathlib import Path

from src.auto_calibrate_boxes import auto_calibrate_boxes
from src.spacing_pipeline import parse_id_list, run_spacing_steps_1_to_4


DEFAULT_PLAYER_MODEL = "runs/detect/runs/basketball_players_yolov8n/weights/best.pt"
DEFAULT_COURT_MODEL = "runs/detect/runs/court/nba_court_yolov8n/weights/best.pt"


def default_output_path(video_path: Path) -> Path:
    return Path("outputs") / f"{video_path.stem}_auto_pipeline.mp4"


def default_csv_path(video_path: Path) -> Path:
    return Path("outputs") / f"{video_path.stem}_auto_pipeline.csv"


def default_calibration_path(video_path: Path) -> Path:
    return Path("calibrations") / f"{video_path.stem}_auto_pipeline.json"


def run_video(args: argparse.Namespace) -> None:
    video_path = args.video
    calibration_path = args.calibration or default_calibration_path(video_path)
    output_path = args.output or default_output_path(video_path)
    csv_path = args.csv_output or default_csv_path(video_path)

    if args.calibration is None:
        auto_calibrate_boxes(
            video_path=video_path,
            output_path=calibration_path,
            model_path=args.court_model,
            hoop_side=args.hoop_side,
            frame_number=args.calibration_frame,
            confidence=args.court_confidence,
            print_classes=False,
            scan_frames=args.scan_frames,
            scan_step=args.scan_step,
            paint_only=args.paint_only,
            min_point_distance=args.min_point_distance,
        )
    else:
        print(f"Using existing calibration: {calibration_path}")

    run_spacing_steps_1_to_4(
        video_path=video_path,
        calibration_path=calibration_path,
        output_path=output_path,
        csv_path=csv_path,
        model_path=args.player_model,
        confidence=args.player_confidence,
        max_frames=args.max_frames,
        player_class_ids=parse_id_list(args.player_class_ids),
        court_margin=args.court_margin,
        keep_outside_court=True,
        hoop_class_ids=parse_id_list(args.hoop_class_ids),
        ball_class_ids=parse_id_list(args.ball_class_ids),
        detect_ball_enabled=True,
        ball_confidence=args.ball_confidence,
        player_smoothing_alpha=args.player_smoothing_alpha,
        ball_smoothing_alpha=args.ball_smoothing_alpha,
        ball_max_jump=args.ball_max_jump,
        ball_max_missing=args.ball_max_missing,
        ball_court_margin=args.ball_court_margin,
        ball_trail_length=args.ball_trail_length,
        topdown_padding=args.topdown_padding,
        ball_radius=args.ball_radius,
        classify_teams=True,
        team_lock_min_votes=args.team_lock_min_votes,
        team_lock_majority_ratio=args.team_lock_majority_ratio,
        min_player_box_height=args.min_player_box_height,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
        max_players=args.max_players,
    )

    print(f"Saved calibration to {calibration_path}")
    print(f"Saved video to {output_path}")
    print(f"Saved CSV to {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-calibrate a basketball video and render player/team/ball court visualization."
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--player-model", default=DEFAULT_PLAYER_MODEL)
    parser.add_argument("--court-model", default=DEFAULT_COURT_MODEL)
    parser.add_argument("--max-frames", type=int)

    parser.add_argument("--hoop-side", choices=["left", "right"], default="right")
    parser.add_argument("--calibration-frame", type=int, default=0)
    parser.add_argument("--court-confidence", type=float, default=0.05)
    parser.add_argument("--scan-frames", type=int, default=240)
    parser.add_argument("--scan-step", type=int, default=5)
    parser.add_argument("--paint-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-point-distance", type=float, default=35.0)

    parser.add_argument("--player-confidence", type=float, default=0.35)
    parser.add_argument("--player-class-ids", default="3")
    parser.add_argument("--hoop-class-ids", default="1")
    parser.add_argument("--ball-class-ids", default="0")
    parser.add_argument("--court-margin", type=int, default=8)
    parser.add_argument("--player-smoothing-alpha", type=float, default=0.35)
    parser.add_argument("--max-players", type=int, default=10)
    parser.add_argument("--min-player-box-height", type=float, default=45.0)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.60)

    parser.add_argument("--team-lock-min-votes", type=int, default=8)
    parser.add_argument("--team-lock-majority-ratio", type=float, default=0.75)

    parser.add_argument("--ball-confidence", type=float, default=0.12)
    parser.add_argument("--ball-smoothing-alpha", type=float, default=0.08)
    parser.add_argument("--ball-max-jump", type=float, default=320.0)
    parser.add_argument("--ball-max-missing", type=int, default=24)
    parser.add_argument("--ball-court-margin", type=int, default=260)
    parser.add_argument("--ball-trail-length", type=int, default=28)
    parser.add_argument("--topdown-padding", type=int, default=260)
    parser.add_argument("--ball-radius", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    run_video(parse_args())
