"""Report figure 3: raw vs velocity-smoothed ball court_x (PHI clip).

The tracks CSV stores the raw ball detection in *frame pixels* (frame_x/frame_y)
and the already-smoothed position in court coords (court_x/court_y).  To compare
RAW vs SMOOTHED court_x we re-project the raw frame detections through the same
static homography the pipeline used (run_all.sh does not enable dynamic
homography), then run the real BallSmoother over the window.

Plots over a ~150-frame window where the ball is detected most often:
  * raw detected court_x (scatter, with gaps)
  * velocity-smoothed court_x (line; the smoother fills short gaps)
  * shaded frame ranges with no raw detection

Run:  PYTHONPATH=. .venv/bin/python tests/make_figure_ball.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.homography import estimate_homography, load_calibration, project_points
from src.smoothing import BallSmoother

TRACKS = Path("outputs/ny_phi_2ndq_tracks.csv")
CALIB = Path("calibrations/ny_phi_2ndq_auto_pipeline.json")
FIG = Path("figures")

PX_PER_FT_X = 940.0 / 47.0   # matches src/metrics.py
WIN_START, WIN_END = 200, 350  # 150-frame window (densest ball coverage)

# Pipeline ball-smoother config (run_video.py defaults).
ALPHA, MAX_SPEED, MAX_MISSING = 0.08, 320.0, 24


def load_raw_ball() -> dict[int, tuple[float, float]]:
    """frame -> (frame_x, frame_y) for raw ball detections."""
    out: dict[int, tuple[float, float]] = {}
    with TRACKS.open() as f:
        for row in csv.DictReader(f):
            if row.get("object_type") != "ball":
                continue
            out[int(row["frame"])] = (float(row["frame_x"]), float(row["frame_y"]))
    return out


def main() -> None:
    FIG.mkdir(exist_ok=True)
    raw_ball = load_raw_ball()

    frame_pts, court_pts = load_calibration(CALIB)
    H = estimate_homography(frame_pts, court_pts)

    frames = list(range(WIN_START, WIN_END))
    n_det = sum(1 for fr in frames if fr in raw_ball)
    print(f"window [{WIN_START},{WIN_END}): {n_det}/{len(frames)} frames with a raw ball detection "
          f"({100*n_det/len(frames):.0f}% coverage)")

    smoother = BallSmoother(alpha=ALPHA, max_speed=MAX_SPEED, max_missing=MAX_MISSING)
    raw_f, raw_x = [], []
    sm_f, sm_x = [], []
    for fr in frames:
        if fr in raw_ball:
            fx, fy = raw_ball[fr]
            court_px = project_points(np.array([[fx, fy]], dtype=np.float32), H)
            raw_f.append(fr)
            raw_x.append(float(court_px[0, 0]) / PX_PER_FT_X)
            smoothed = smoother.smooth(court_px)
        else:
            smoothed = smoother.smooth(np.empty((0, 2), dtype=np.float32))
        if len(smoothed) > 0:
            sm_f.append(fr)
            sm_x.append(float(smoothed[0, 0]) / PX_PER_FT_X)

    # Break the smoothed line across long gaps the smoother does not fill.
    sm_x_plot = list(sm_x)
    for i in range(1, len(sm_f)):
        if sm_f[i] - sm_f[i - 1] > 1:
            sm_x_plot[i] = np.nan  # break the line at a non-contiguous jump

    # Shade frame ranges with no raw detection.
    detected = set(raw_f)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    in_gap = False
    gap_start = WIN_START
    first_gap = True
    for fr in range(WIN_START, WIN_END + 1):
        present = fr in detected and fr < WIN_END
        if not present and not in_gap:
            in_gap, gap_start = True, fr
        elif present and in_gap:
            ax.axvspan(gap_start - 0.5, fr - 0.5, color="0.85", zorder=0,
                       label="no raw detection" if first_gap else None)
            first_gap = False
            in_gap = False

    ax.plot(sm_f, sm_x_plot, "-", color="crimson", linewidth=1.8,
            label="velocity-smoothed court_x", zorder=3)
    ax.plot(raw_f, raw_x, "o", color="steelblue", markersize=4,
            label="raw detected court_x", zorder=4)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Ball court_x (ft from midcourt)")
    ax.set_title("Ball Tracking: Raw Detection vs Velocity-Smoothed Court Position (PHI clip)")
    ax.set_xlim(WIN_START, WIN_END)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    out = FIG / "fig_ball_smoothing.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
