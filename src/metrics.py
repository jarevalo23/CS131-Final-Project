"""Spacing / coverage metrics from a tracks CSV (Step 8).

Reads ``outputs/<clip>_tracks.csv`` (produced by ``src.main`` /
``run_spacing_steps_1_to_4``) and computes per-frame spacing metrics:

  * offensive and defensive convex-hull area
  * nearest-defender distance per offensive player (aggregated + per-player file)
  * defensive compactness (mean distance of defenders to their centroid)
  * corner occupancy (offensive players in the baseline corners)
  * double-teams (offensive players with >= 2 defenders within a radius)

Offense vs. defense is set manually with ``--offense-team`` for now (team ids
are jersey clusters 0/1; possession-based labelling comes later). Distances are
in feet, converted from the court-pixel coordinates in the CSV using the
``src.court`` constants.

Court coordinate convention (matches the calibration landmarks / homography):
  x_ft: 0 = midcourt line, 47 = baseline (hoop end);  hoop at (41.75, 25)
  y_ft: 0 and 50 = the two sidelines (court is 50 ft wide)
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.court import DEFAULT_COURT_SIZE, NBA_COURT_WIDTH_FT, NBA_HALF_COURT_LENGTH_FT


# Court pixels per foot. court_feet_to_pixels maps x_ft -> x_ft/47*940 and
# y_ft -> y_ft/50*1000, i.e. 20 px/ft on both axes for the default court size.
PX_PER_FT_X = DEFAULT_COURT_SIZE[0] / NBA_HALF_COURT_LENGTH_FT
PX_PER_FT_Y = DEFAULT_COURT_SIZE[1] / NBA_COURT_WIDTH_FT


@dataclass(frozen=True)
class Player:
    track_id: int
    team_id: int
    x_ft: float
    y_ft: float


@dataclass(frozen=True)
class BallPossession:
    frame: int
    possessor_track_id: int   # -1 means airborne / no possessor detected
    airborne: bool


def load_player_frames(csv_path: Path) -> dict[int, list[Player]]:
    """frame -> list of every player row (all teams, including unlocked -1)."""
    frames: dict[int, list[Player]] = defaultdict(list)
    with csv_path.open() as file:
        for row in csv.DictReader(file):
            if row.get("object_type") != "player":
                continue
            try:
                team = int(row["team_id"])
            except (KeyError, ValueError):
                continue
            frames[int(row["frame"])].append(
                Player(
                    track_id=int(row["track_id"]),
                    team_id=team,
                    x_ft=float(row["court_x"]) / PX_PER_FT_X,
                    y_ft=float(row["court_y"]) / PX_PER_FT_Y,
                )
            )
    return frames


def apply_same_player_guard(players: list[Player], radius_ft: float = 1.0) -> tuple[list[Player], int]:
    """Drop near-duplicate detections (one physical player split into two rows).

    Two rows in the same frame whose court positions are within ``radius_ft`` are
    treated as the same player. Within a duplicate group we keep the
    highest-priority row: a locked team (team_id != -1) is preferred over -1,
    then the lower track_id (older, more established track). Returns the kept
    rows and the number dropped.
    """
    order = sorted(players, key=lambda p: (p.team_id == -1, p.track_id))
    r2 = radius_ft * radius_ft
    kept: list[Player] = []
    for p in order:
        if any((p.x_ft - k.x_ft) ** 2 + (p.y_ft - k.y_ft) ** 2 <= r2 for k in kept):
            continue
        kept.append(p)
    return kept, len(players) - len(kept)


def guard_frames(
    frames: dict[int, list[Player]], radius_ft: float
) -> tuple[dict[int, list[Player]], int, int]:
    """Apply the same-player guard to every frame. Returns (frames, frames_affected, rows_dropped)."""
    out: dict[int, list[Player]] = {}
    frames_affected = 0
    rows_dropped = 0
    for frame, players in frames.items():
        kept, dropped = apply_same_player_guard(players, radius_ft)
        out[frame] = kept
        if dropped > 0:
            frames_affected += 1
            rows_dropped += dropped
    return out, frames_affected, rows_dropped


def convex_hull_area_ft2(points: np.ndarray) -> float | None:
    """2D convex-hull area in ft^2. None if < 3 points; 0.0 if degenerate/collinear."""
    if len(points) < 3:
        return None
    from scipy.spatial import ConvexHull

    try:
        return float(ConvexHull(points).volume)  # for 2D, .volume is the area
    except Exception:
        # Degenerate (collinear / coincident) point set -> no enclosed area.
        return 0.0


def mean_distance_to_centroid_ft(points: np.ndarray) -> float | None:
    if len(points) == 0:
        return None
    centroid = points.mean(axis=0)
    return float(np.linalg.norm(points - centroid, axis=1).mean())


def nearest_defender(off_xy: np.ndarray, def_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each offensive point, distance (ft) and index of the nearest defender."""
    if len(off_xy) == 0 or len(def_xy) == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.int64)
    diff = off_xy[:, None, :] - def_xy[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    idx = np.argmin(dists, axis=1)
    return dists[np.arange(len(off_xy)), idx], idx


def fmt(value: float | None, ndigits: int = 2) -> str:
    return "" if value is None else f"{value:.{ndigits}f}"


def compute_metrics(
    frames: dict[int, list[Player]],
    offense_team: int,
    defense_team: int,
    double_team_radius_ft: float,
    corner_depth_ft: float,
    corner_width_ft: float,
) -> tuple[list[dict], list[dict]]:
    frame_rows: list[dict] = []
    nearest_rows: list[dict] = []

    for frame in sorted(frames):
        players = frames[frame]
        offense = [p for p in players if p.team_id == offense_team]
        defense = [p for p in players if p.team_id == defense_team]
        off_xy = np.array([[p.x_ft, p.y_ft] for p in offense], dtype=np.float64).reshape(-1, 2)
        def_xy = np.array([[p.x_ft, p.y_ft] for p in defense], dtype=np.float64).reshape(-1, 2)

        nd_dist, nd_idx = nearest_defender(off_xy, def_xy)

        # double-teams: offensive players with >= 2 defenders within the radius
        double_ids: list[int] = []
        if len(off_xy) and len(def_xy):
            diff = off_xy[:, None, :] - def_xy[None, :, :]
            within = np.linalg.norm(diff, axis=2) <= double_team_radius_ft
            counts = within.sum(axis=1)
            double_ids = [offense[i].track_id for i in range(len(offense)) if counts[i] >= 2]

        # corner occupancy: near baseline (x >= 47 - depth) and near a sideline
        corner_left = corner_right = 0
        for p in offense:
            if p.x_ft >= NBA_HALF_COURT_LENGTH_FT - corner_depth_ft:
                if p.y_ft <= corner_width_ft:
                    corner_left += 1
                elif p.y_ft >= NBA_COURT_WIDTH_FT - corner_width_ft:
                    corner_right += 1

        frame_rows.append(
            {
                "frame": frame,
                "offense_team": offense_team,
                "defense_team": defense_team,
                "n_offense": len(offense),
                "n_defense": len(defense),
                "offense_hull_area_ft2": fmt(convex_hull_area_ft2(off_xy)),
                "defense_hull_area_ft2": fmt(convex_hull_area_ft2(def_xy)),
                "defense_compactness_ft": fmt(mean_distance_to_centroid_ft(def_xy)),
                "mean_nearest_def_ft": fmt(float(nd_dist.mean()) if len(nd_dist) else None),
                "min_nearest_def_ft": fmt(float(nd_dist.min()) if len(nd_dist) else None),
                "max_nearest_def_ft": fmt(float(nd_dist.max()) if len(nd_dist) else None),
                "corner_occ_left": corner_left,
                "corner_occ_right": corner_right,
                "corner_occ_total": corner_left + corner_right,
                "n_double_teamed": len(double_ids),
                "double_teamed_off_ids": ";".join(str(i) for i in double_ids),
            }
        )

        for i, p in enumerate(offense):
            nearest_rows.append(
                {
                    "frame": frame,
                    "off_track_id": p.track_id,
                    "off_x_ft": f"{p.x_ft:.2f}",
                    "off_y_ft": f"{p.y_ft:.2f}",
                    "nearest_def_track_id": defense[int(nd_idx[i])].track_id if len(def_xy) else "",
                    "nearest_def_dist_ft": f"{nd_dist[i]:.2f}" if len(nd_dist) else "",
                }
            )

    return frame_rows, nearest_rows


def load_possession(csv_path: Path) -> list[BallPossession]:
    """Load per-frame ball possession from a tracks CSV."""
    rows: list[BallPossession] = []
    with csv_path.open() as file:
        for row in csv.DictReader(file):
            if row.get("object_type") != "ball":
                continue
            airborne_raw = row.get("ball_airborne", "True")
            airborne = str(airborne_raw).strip().lower() not in ("false", "0", "")
            possessor_raw = row.get("possessing_ball", "-1")
            try:
                possessor = int(possessor_raw)
            except (ValueError, TypeError):
                possessor = -1
            rows.append(BallPossession(
                frame=int(row["frame"]),
                possessor_track_id=possessor,
                airborne=airborne,
            ))
    return rows


def compute_speed_distance(
    frames: dict[int, list[Player]],
    fps: float,
    window: int = 5,
) -> list[dict]:
    """Compute per-player speed (mph) and cumulative distance (ft) from court coords.

    Uses a sliding window of `window` frames to smooth speed estimates and
    reduce noise from jitter in the projected positions.  Distance is the
    sum of frame-to-frame displacements in court feet.

    Speed is only computed when a player is present in at least `window`
    consecutive frames; otherwise the speed cell is left empty.
    """
    # Build per-track position history: {track_id: [(frame, x_ft, y_ft), ...]}
    track_history: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for frame in sorted(frames):
        for p in frames[frame]:
            track_history[p.track_id].append((frame, p.x_ft, p.y_ft))

    rows: list[dict] = []
    for track_id, history in sorted(track_history.items()):
        if len(history) < 2:
            continue
        # Team id for this track (use the most common non-(-1) value seen).
        team_counts: Counter = Counter()
        for frame_num in sorted(frames):
            for p in frames[frame_num]:
                if p.track_id == track_id and p.team_id >= 0:
                    team_counts[p.team_id] += 1
        team_id = team_counts.most_common(1)[0][0] if team_counts else -1

        # Frame-to-frame displacement in feet.
        total_dist_ft = 0.0
        speeds_mph: list[float] = []
        for i in range(1, len(history)):
            f0, x0, y0 = history[i - 1]
            f1, x1, y1 = history[i]
            frame_gap = max(1, f1 - f0)
            dist_ft = float(np.hypot(x1 - x0, y1 - y0))
            total_dist_ft += dist_ft

            # Only compute speed over contiguous window frames.
            if i >= window:
                fw, xw, yw = history[i - window]
                window_frames = max(1, f1 - fw)
                window_dist_ft = float(np.hypot(x1 - xw, y1 - yw))
                time_s = window_frames / fps
                # ft/s → mph: 1 ft/s = 0.681818 mph
                speeds_mph.append(window_dist_ft / time_s * 0.681818)

        avg_speed = float(np.mean(speeds_mph)) if speeds_mph else None
        max_speed = float(np.max(speeds_mph)) if speeds_mph else None
        rows.append({
            "track_id": track_id,
            "team_id": team_id,
            "frames_tracked": len(history),
            "total_distance_ft": f"{total_dist_ft:.1f}",
            "avg_speed_mph": f"{avg_speed:.2f}" if avg_speed is not None else "",
            "max_speed_mph": f"{max_speed:.2f}" if max_speed is not None else "",
        })
    return rows


def compute_passes(
    possession_rows: list[BallPossession],
    frames: dict[int, list[Player]],
) -> list[dict]:
    """Detect passes and turnovers from possession change events.

    A pass is a possession change between two players on the same team.
    A turnover is a possession change between players on different teams.
    Airborne frames (ball in flight) are skipped — we look for the frame
    where a new player takes possession.

    NOTE: accuracy depends on team classification quality.  If team_id
    values are unreliable for a clip, the pass/turnover counts will be
    noisy.  The raw possession_track_id is always recorded regardless.
    """
    # Build frame -> track_id -> team_id lookup.
    team_by_track: dict[int, int] = {}
    for player_list in frames.values():
        for p in player_list:
            if p.team_id >= 0:
                team_by_track[p.track_id] = p.team_id

    events: list[dict] = []
    prev_possessor = -1
    prev_team = -1

    for bp in sorted(possession_rows, key=lambda r: r.frame):
        if bp.airborne or bp.possessor_track_id < 0:
            continue
        curr_possessor = bp.possessor_track_id
        curr_team = team_by_track.get(curr_possessor, -1)

        if curr_possessor != prev_possessor and prev_possessor >= 0:
            if curr_team < 0 or prev_team < 0:
                event_type = "unknown"
            elif curr_team == prev_team:
                event_type = "pass"
            else:
                event_type = "turnover"
            events.append({
                "frame": bp.frame,
                "from_track_id": prev_possessor,
                "from_team_id": prev_team,
                "to_track_id": curr_possessor,
                "to_team_id": curr_team,
                "event_type": event_type,
            })

        prev_possessor = curr_possessor
        prev_team = curr_team

    return events


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(frame_rows: list[dict]) -> None:
    def col(name: str) -> list[float]:
        return [float(r[name]) for r in frame_rows if r[name] != ""]

    if not frame_rows:
        print("No frames with players for the selected teams.")
        return
    n = len(frame_rows)
    off_hull = col("offense_hull_area_ft2")
    def_comp = col("defense_compactness_ft")
    min_nd = col("min_nearest_def_ft")
    dt_frames = sum(1 for r in frame_rows if int(r["n_double_teamed"]) > 0)
    corner_frames = sum(1 for r in frame_rows if int(r["corner_occ_total"]) > 0)
    print(f"  frames analyzed: {n}")
    if off_hull:
        print(f"  offense hull area ft^2: mean {np.mean(off_hull):.0f}  (min {np.min(off_hull):.0f}, max {np.max(off_hull):.0f})  [{len(off_hull)} frames w/ >=3 offense]")
    if def_comp:
        print(f"  defense compactness ft: mean {np.mean(def_comp):.2f}  (lower = tighter)")
    if min_nd:
        print(f"  closest defender on any offensive player: mean {np.mean(min_nd):.2f} ft, tightest {np.min(min_nd):.2f} ft")
    print(f"  frames with >=1 double-team: {dt_frames} ({100*dt_frames/n:.1f}%)")
    print(f"  frames with a corner occupied: {corner_frames} ({100*corner_frames/n:.1f}%)")


def default_output(tracks_path: Path, suffix: str) -> Path:
    stem = tracks_path.name
    if stem.endswith("_tracks.csv"):
        stem = stem[: -len("_tracks.csv")]
    else:
        stem = tracks_path.stem
    return tracks_path.parent / f"{stem}_{suffix}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracks", required=True, type=Path, help="Path to <clip>_tracks.csv")
    parser.add_argument("--offense-team", type=int, choices=[0, 1], default=0, help="Which jersey-cluster team is on offense (manual)")
    parser.add_argument("--output", type=Path, help="Frame-level metrics CSV (default: <clip>_metrics.csv)")
    parser.add_argument("--nearest-output", type=Path, help="Per-offensive-player nearest-defender CSV (default: <clip>_nearest_defender.csv)")
    parser.add_argument("--double-team-radius-ft", type=float, default=6.0)
    parser.add_argument("--corner-depth-ft", type=float, default=14.0, help="How far from the baseline (x>=47-depth) still counts as a corner")
    parser.add_argument("--corner-width-ft", type=float, default=4.0, help="How far from a sideline still counts as a corner")
    parser.add_argument("--guard-radius-ft", type=float, default=1.0, help="Same-player guard: drop duplicate rows within this distance")
    parser.add_argument("--no-guard", action="store_true", help="Disable the same-player duplicate guard")
    parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate for speed/distance calculation")
    parser.add_argument("--speed-output", type=Path, help="Per-player speed/distance CSV (default: <clip>_speed.csv)")
    parser.add_argument("--passes-output", type=Path, help="Pass/turnover events CSV (default: <clip>_passes.csv)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    offense_team = args.offense_team
    defense_team = 1 - offense_team
    output = args.output or default_output(args.tracks, "metrics")
    nearest_output = args.nearest_output or default_output(args.tracks, "nearest_defender")
    speed_output = args.speed_output or default_output(args.tracks, "speed")
    passes_output = args.passes_output or default_output(args.tracks, "passes")

    frames = load_player_frames(args.tracks)
    if args.no_guard:
        print("same-player guard: DISABLED")
    else:
        frames, frames_affected, rows_dropped = guard_frames(frames, args.guard_radius_ft)
        print(f"same-player guard: radius {args.guard_radius_ft} ft | {frames_affected} frames affected, {rows_dropped} duplicate rows dropped")

    frame_rows, nearest_rows = compute_metrics(
        frames,
        offense_team=offense_team,
        defense_team=defense_team,
        double_team_radius_ft=args.double_team_radius_ft,
        corner_depth_ft=args.corner_depth_ft,
        corner_width_ft=args.corner_width_ft,
    )
    write_csv(output, frame_rows)
    write_csv(nearest_output, nearest_rows)
    print(f"offense=team {offense_team}, defense=team {defense_team}")
    print_summary(frame_rows)
    print(f"  wrote {output}")
    print(f"  wrote {nearest_output} ({len(nearest_rows)} offensive-player rows)")

    # Speed / distance — fully independent of team classification.
    speed_rows = compute_speed_distance(frames, fps=args.fps)
    write_csv(speed_output, speed_rows)
    print(f"\nspeed / distance (fps={args.fps}):")
    for r in speed_rows:
        print(f"  track {r['track_id']} team {r['team_id']}: "
              f"{r['total_distance_ft']} ft  avg {r['avg_speed_mph']} mph  "
              f"max {r['max_speed_mph']} mph  ({r['frames_tracked']} frames)")
    print(f"  wrote {speed_output}")

    # Passes / turnovers — depends on team classification; flagged in output.
    possession_rows = load_possession(args.tracks)
    if possession_rows:
        pass_rows = compute_passes(possession_rows, frames)
        write_csv(passes_output, pass_rows)
        passes = [e for e in pass_rows if e["event_type"] == "pass"]
        turnovers = [e for e in pass_rows if e["event_type"] == "turnover"]
        unknown = [e for e in pass_rows if e["event_type"] == "unknown"]
        print(f"\npossession events (NOTE: pass/turnover labels depend on team classification accuracy):")
        print(f"  passes: {len(passes)}  turnovers: {len(turnovers)}  unknown team: {len(unknown)}")
        print(f"  wrote {passes_output}")
    else:
        print("\nno possession data in tracks CSV (re-run pipeline to generate possessing_ball column)")


if __name__ == "__main__":
    main()
