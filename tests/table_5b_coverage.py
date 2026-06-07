"""Table 5b: offensive-player coverage fractions (duplicate guard applied).

Buckets every offensive-player-frame by nearest-defender distance:
  tight     < 3 ft
  contested 3-6 ft   (inclusive)
  open      > 6 ft

Re-derives distances from the tracks CSV with the same-player guard applied
(src.metrics, radius 1.0 ft), and cross-checks against the already-guarded
nearest_defender CSV that the pipeline wrote.

Run:  PYTHONPATH=. .venv/bin/python tests/table_5b_coverage.py
"""
from __future__ import annotations

import csv
from pathlib import Path

from src.metrics import load_player_frames, guard_frames, compute_metrics

TRACKS = Path("outputs/ny_hawks_game6_away_tracks.csv")
NEAREST_CSV = Path("outputs/ny_hawks_game6_away_nearest_defender.csv")
OFFENSE_TEAM = 0
GUARD_RADIUS_FT = 1.0


def bucket(dists: list[float]) -> None:
    n = len(dists)
    tight = sum(1 for d in dists if d < 3.0)
    contested = sum(1 for d in dists if 3.0 <= d <= 6.0)
    open_ = sum(1 for d in dists if d > 6.0)
    print(f"  total offensive-player-frames with a nearest defender: {n}")
    print(f"  open      (>6 ft):   {open_:5d}   {100*open_/n:5.1f}%")
    print(f"  contested (3-6 ft):  {contested:5d}   {100*contested/n:5.1f}%")
    print(f"  tight     (<3 ft):   {tight:5d}   {100*tight/n:5.1f}%")


def main() -> None:
    # Re-derive from tracks with the duplicate guard applied.
    frames = load_player_frames(TRACKS)
    guarded, frames_affected, rows_dropped = guard_frames(frames, GUARD_RADIUS_FT)
    print(f"same-player guard: radius {GUARD_RADIUS_FT} ft | "
          f"{frames_affected} frames affected, {rows_dropped} duplicate rows dropped")
    _, nearest_rows = compute_metrics(
        guarded, offense_team=OFFENSE_TEAM, defense_team=1 - OFFENSE_TEAM,
        double_team_radius_ft=6.0, corner_depth_ft=14.0, corner_width_ft=4.0,
    )
    dists = [float(r["nearest_def_dist_ft"]) for r in nearest_rows
             if r["nearest_def_dist_ft"] != ""]
    print("\n[5b] Offensive coverage — re-derived from tracks (guard applied):")
    bucket(dists)

    # Cross-check against the pipeline's already-guarded nearest_defender CSV.
    csv_dists = []
    with NEAREST_CSV.open() as f:
        for row in csv.DictReader(f):
            v = row.get("nearest_def_dist_ft", "")
            if v:
                csv_dists.append(float(v))
    print(f"\n[5b] Cross-check — existing {NEAREST_CSV.name}:")
    bucket(csv_dists)


if __name__ == "__main__":
    main()
