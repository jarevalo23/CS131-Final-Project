"""Report figures 1 & 2 from the duplicate-guarded Hawks metrics CSV.

Fig 1: figures/fig_hull_area.png  -- offensive convex-hull area (ft^2) vs frame.
Fig 2: figures/fig_hull_spread.png / figures/fig_hull_collapsed.png
       -- the largest and smallest hull-area frames (n_offense >= 4),
          pulled from the overlay video where the hull is already drawn.

Run from repo root:  PYTHONPATH=. .venv/bin/python tests/make_figures_1_2.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = Path("outputs/ny_hawks_game6_away_metrics.csv")
OVERLAY = Path("outputs/hawks_offense0_overlay.mp4")
FIG = Path("figures")


def load_rows() -> list[dict]:
    with METRICS.open() as f:
        return list(csv.DictReader(f))


def fig1_hull_area(rows: list[dict]) -> None:
    frames, areas = [], []
    for r in rows:
        if r["offense_hull_area_ft2"]:
            frames.append(int(r["frame"]))
            areas.append(float(r["offense_hull_area_ft2"]))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(frames, areas, linewidth=0.9, color="steelblue")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Offensive convex hull area (ft²)")
    ax.set_title("Offensive Convex Hull Area Over Possession")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = FIG / "fig_hull_area.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved {out}  ({len(frames)} frames with a hull)")


def grab_frame(video: Path, frame_no: int, out: Path) -> bool:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"  [ERROR] could not read frame {frame_no} from {video}")
        return False
    cv2.imwrite(str(out), frame)
    print(f"  saved {out}  (frame {frame_no})")
    return True


def fig2_spread_collapsed(rows: list[dict]) -> None:
    cand = [
        (int(r["frame"]), float(r["offense_hull_area_ft2"]))
        for r in rows
        if r["offense_hull_area_ft2"] and int(r["n_offense"]) >= 4
    ]
    if not cand:
        print("  [skip] no frames with hull and n_offense>=4")
        return
    largest = max(cand, key=lambda t: t[1])
    smallest = min(cand, key=lambda t: t[1])
    print(f"\n  LARGEST  hull: frame {largest[0]}  area {largest[1]:.1f} ft²  -> fig_hull_spread.png")
    print(f"  SMALLEST hull: frame {smallest[0]}  area {smallest[1]:.1f} ft²  -> fig_hull_collapsed.png")
    grab_frame(OVERLAY, largest[0], FIG / "fig_hull_spread.png")
    grab_frame(OVERLAY, smallest[0], FIG / "fig_hull_collapsed.png")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    rows = load_rows()
    print("[Fig 1] hull area over time")
    fig1_hull_area(rows)
    print("[Fig 2] hull spread / collapsed frames")
    fig2_spread_collapsed(rows)


if __name__ == "__main__":
    main()
