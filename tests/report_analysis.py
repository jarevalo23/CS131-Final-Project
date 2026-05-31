"""Generate all analytics outputs and figures for the final report.

Produces:
  tests/report_figures/nearest_defender_dist.png  -- histogram of defender distances
  tests/report_figures/hull_area_over_time.png     -- offensive spacing over time
  tests/report_figures/hawks/frame_NNNNN.jpg       -- annotated frame screenshots
  tests/report_figures/phi/frame_NNNNN.jpg         -- annotated frame screenshots
  tests/report_figures/calib_hawks.jpg             -- calibration review image
  tests/report_figures/calib_phi.jpg               -- calibration review image
  (also prints the spacing analytics summary table to stdout)

Usage:
    python3 -m tests.report_analysis
    python3 -m tests.report_analysis --skip-frames   # skip slow frame extraction
"""

from __future__ import annotations

import argparse
import csv
import shutil
import statistics
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Paths ──────────────────────────────────────────────────────────────────────

CLIPS = {
    "Hawks G6": {
        "metrics":          Path("outputs/hawks_metrics.csv"),
        "nearest":          Path("outputs/hawks_nearest_defender.csv"),
        "speed":            Path("outputs/hawks_speed.csv"),
        "passes":           Path("outputs/hawks_passes.csv"),
        "overlay_video":    Path("outputs/hawks_overlay_overlay.mp4"),
        "calib_review":     Path("calibrations/ny_hawks_game6_away_auto_pipeline_review.jpg"),
        "color":            "steelblue",
    },
    "NYK vs PHI": {
        "metrics":          Path("outputs/phi_metrics.csv"),
        "nearest":          Path("outputs/phi_nearest_defender.csv"),
        "speed":            Path("outputs/phi_speed.csv"),
        "passes":           Path("outputs/phi_passes.csv"),
        "overlay_video":    Path("outputs/phi_overlay_overlay.mp4"),
        "calib_review":     Path("calibrations/ny_phi_2ndq_auto_pipeline_review.jpg"),
        "color":            "tomato",
    },
}

OUT_DIR = Path("tests/report_figures")


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_col(path: Path, col: str) -> list[float]:
    vals = []
    with path.open() as f:
        for row in csv.DictReader(f):
            raw = row.get(col, "")
            if raw:
                try:
                    vals.append(float(raw))
                except ValueError:
                    pass
    return vals


def load_hull_series(path: Path) -> tuple[list[int], list[float]]:
    frames, vals = [], []
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("offense_hull_area_ft2"):
                frames.append(int(row["frame"]))
                vals.append(float(row["offense_hull_area_ft2"]))
    return frames, vals


def count_events(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            t = row.get("event_type", "unknown")
            counts[t] = counts.get(t, 0) + 1
    return counts


# ── Figure 1: Nearest-defender distance histogram ─────────────────────────────

def plot_nearest_defender_hist() -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, paths in CLIPS.items():
        if not paths["nearest"].exists():
            print(f"  [skip] {paths['nearest']} not found")
            continue
        dists = load_col(paths["nearest"], "nearest_def_dist_ft")
        ax.hist(dists, bins=30, alpha=0.6, label=label, color=paths["color"])

    ax.axvline(4.0, color="black", linestyle="--", linewidth=1,
               label="4 ft — tight coverage threshold")
    ax.set_xlabel("Nearest defender distance (ft)")
    ax.set_ylabel("Frame count")
    ax.set_title("Distribution of Nearest-Defender Distances")
    ax.legend()
    plt.tight_layout()
    out = OUT_DIR / "nearest_defender_dist.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


# ── Figure 2: Offensive hull area over time ────────────────────────────────────

def plot_hull_over_time() -> Path:
    valid = [(label, p) for label, p in CLIPS.items() if p["metrics"].exists()]
    fig, axes = plt.subplots(len(valid), 1, figsize=(9, 4 * len(valid)), sharex=False)
    if len(valid) == 1:
        axes = [axes]

    colors = [p["color"] for _, p in valid]
    for ax, (label, paths), color in zip(axes, valid, colors):
        frames, vals = load_hull_series(paths["metrics"])
        ax.plot(frames, vals, linewidth=0.8, color=color)
        ax.set_title(f"{label} — Offensive Spacing (Convex Hull Area)")
        ax.set_ylabel("Hull area (ft²)")
        ax.set_xlabel("Frame")

    plt.tight_layout()
    out = OUT_DIR / "hull_area_over_time.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


# ── Figure 3: Frame screenshots from overlay videos ───────────────────────────

def extract_frames(video_path: Path, out_dir: Path, n_frames: int = 6) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        print(f"  [skip] {video_path} has 0 frames or could not be opened")
        cap.release()
        return []

    # Evenly spaced frames, skipping the first and last 5%
    start = int(total * 0.05)
    end   = int(total * 0.95)
    step  = max(1, (end - start) // n_frames)
    frame_nums = list(range(start, end, step))[:n_frames]

    saved = []
    for n in frame_nums:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ret, frame = cap.read()
        if ret:
            out = out_dir / f"frame_{n:05d}.jpg"
            cv2.imwrite(str(out), frame)
            saved.append(out)
    cap.release()
    print(f"  extracted {len(saved)} frames from {video_path.name} → {out_dir}")
    return saved


# ── Figure 4: Calibration review images ───────────────────────────────────────

def copy_calib_reviews() -> list[Path]:
    copied = []
    for label, paths in CLIPS.items():
        src = paths["calib_review"]
        if not src.exists():
            print(f"  [skip] calibration review not found: {src}")
            continue
        slug = label.lower().replace(" ", "_").replace("/", "_")
        dst = OUT_DIR / f"calib_{slug}.jpg"
        shutil.copy(src, dst)
        copied.append(dst)
        print(f"  copied {src.name} → {dst}")
    return copied


# ── Analytics summary table (stdout) ──────────────────────────────────────────

def print_summary_table() -> None:
    print("\n" + "=" * 65)
    print("SPACING ANALYTICS SUMMARY")
    print("=" * 65)

    for label, paths in CLIPS.items():
        print(f"\n── {label} ──")

        if not paths["metrics"].exists():
            print("  metrics CSV not found — run src.metrics first")
            continue

        hull    = load_col(paths["metrics"], "offense_hull_area_ft2")
        compact = load_col(paths["metrics"], "defense_compactness_ft")
        corner  = load_col(paths["metrics"], "corner_occ_total")

        double_frames = total_frames = 0
        with paths["metrics"].open() as f:
            for row in csv.DictReader(f):
                total_frames += 1
                if int(row.get("n_double_teamed", 0)) > 0:
                    double_frames += 1

        nd = load_col(paths["nearest"], "nearest_def_dist_ft") if paths["nearest"].exists() else []

        speeds_avg = load_col(paths["speed"], "avg_speed_mph") if paths["speed"].exists() else []
        speeds_max = load_col(paths["speed"], "max_speed_mph") if paths["speed"].exists() else []

        events = count_events(paths["passes"]) if paths["passes"].exists() else {}

        print(f"  Frames analyzed:               {total_frames}")
        if hull:
            print(f"  Offensive hull area (mean):     {statistics.mean(hull):.1f} ft²"
                  f"  (min {min(hull):.0f}, max {max(hull):.0f})")
        if compact:
            print(f"  Defensive compactness (mean):   {statistics.mean(compact):.2f} ft")
        if nd:
            print(f"  Mean nearest-defender dist:     {statistics.mean(nd):.2f} ft"
                  f"  (median {statistics.median(nd):.2f})")
        if total_frames:
            print(f"  % frames with double-team:      {100*double_frames/total_frames:.1f}%")
        if corner:
            print(f"  Corner occupancy (mean):        {statistics.mean(corner):.2f} players/frame")
        if events:
            print(f"  Passes detected:                {events.get('pass', 0)}")
            print(f"  Turnovers detected:             {events.get('turnover', 0)}")
            print(f"  Unknown team events:            {events.get('unknown', 0)}")
        if speeds_avg:
            print(f"  Avg player speed (mean):        {statistics.mean(speeds_avg):.2f} mph")
        if speeds_max:
            print(f"  Max player speed (peak):        {max(speeds_max):.2f} mph")

    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-frames", action="store_true",
                        help="Skip frame extraction (fast mode — plots and stats only)")
    parser.add_argument("--n-frames", type=int, default=6,
                        help="Number of frames to extract per clip (default 6)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Spacing analytics summary")
    print_summary_table()

    print("[2/5] Nearest-defender distance histogram")
    out = plot_nearest_defender_hist()
    print(f"  saved {out}")

    print("\n[3/5] Hull area over time")
    out = plot_hull_over_time()
    print(f"  saved {out}")

    print("\n[4/5] Calibration review images")
    copy_calib_reviews()

    if args.skip_frames:
        print("\n[5/5] Frame extraction skipped (--skip-frames)")
    else:
        print("\n[5/5] Extracting annotated frames from overlay videos")
        for label, paths in CLIPS.items():
            slug = label.lower().replace(" ", "_").replace("/", "_")
            if not paths["overlay_video"].exists():
                print(f"  [skip] {paths['overlay_video']} not found — run pipeline + overlay first")
                continue
            extract_frames(paths["overlay_video"], OUT_DIR / slug, n_frames=args.n_frames)

    print(f"\nAll outputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
