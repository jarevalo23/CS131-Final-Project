# CS131 Basketball Court Tracking

Converts NBA broadcast video into a top-down court map with player positions, team labels, ball tracking, and spacing analytics. Built for CS131 at Stanford.

## Pipeline Overview

```
Video clip
   ↓
Calibration  →  homography JSON (frame pixels ↔ court feet)
   ↓
src.run_video  →  annotated video + tracks CSV
   ↓
src.metrics    →  spacing metrics + speed/distance + pass events
   ↓
tests/report_analysis.py  →  report figures + analytics summary
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place video clips in `data/`. Calibration JSONs go in `calibrations/`.

---

## Step 1 — Calibrate a clip

### Option A: Auto-calibration (recommended if it works)

Uses a trained court landmark model to detect paint corners automatically.

```bash
python3 -m src.run_video \
  --video data/clip.mp4 \
  --court-model runs/detect/runs/court/nba_court_yolov8n/weights/best.pt \
  --player-model runs/detect/runs/basketball_players_yolov8n/weights/best.pt \
  --hoop-side right \
  --output outputs/clip_overlay.mp4 \
  --csv-output outputs/clip_tracks.csv
```

This auto-calibrates and runs the full pipeline in one shot. Use `--hoop-side left` if the basket is on the left side of the frame. If it fails with "fewer than 4 landmarks", try `--no-paint-only` or fall back to manual calibration.

### Option B: Manual calibration

Opens a GUI on the first frame. Click at least 4 court landmarks (paint corners, free throw line, arc corners) in order.

```bash
python3 -m src.calibrate \
  --video data/clip.mp4 \
  --output calibrations/clip.json \
  --hoop-side right
```

Keys: **click** to place, **u** to undo, **k** to skip, **s** to save, **q** to quit.

Check `calibrations/clip_review.jpg` to verify the homography looks correct.

---

## Step 2 — Run the pipeline

If you already have a calibration JSON:

```bash
python3 -m src.run_video \
  --video data/clip.mp4 \
  --calibration calibrations/clip.json \
  --player-model runs/detect/runs/basketball_players_yolov8n/weights/best.pt \
  --output outputs/clip_overlay.mp4 \
  --csv-output outputs/clip_tracks.csv \
  --ball-trail-length 28 \
  --max-players 10
```

Produces:
- `outputs/clip_overlay.mp4` — side-by-side broadcast + top-down court view
- `outputs/clip_tracks.csv` — per-frame player/ball positions

### Key flags

| Flag | Default | Effect |
|---|---|---|
| `--max-players` | none | Cap detections to N players (use 10 for half-court) |
| `--ball-trail-length` | 18 | Frames of ball trail to draw |
| `--ball-max-jump` | 320 | Max px/frame ball can move (spike filter) |
| `--ball-max-missing` | 24 | Frames to extrapolate ball during occlusion |
| `--dynamic-homography` | off | Refine homography frame-to-frame with optical flow |
| `--court-margin` | 8 | Increase if sideline people bleed into court view |
| `--max-frames N` | all | Process only first N frames (quick test) |

---

## Step 3 — Compute spacing metrics

Watch `outputs/clip_overlay.mp4` to determine which team has the ball (orange boxes = team 0, blue = team 1), then:

```bash
python3 -m src.metrics \
  --tracks outputs/clip_tracks.csv \
  --offense-team 0 \
  --fps 30
```

Produces four CSVs in `outputs/`:

| File | Contents |
|---|---|
| `clip_metrics.csv` | Per-frame: convex hull area, defensive compactness, double-teams, corner occupancy |
| `clip_nearest_defender.csv` | Per offensive player: distance to nearest defender each frame |
| `clip_speed.csv` | Per player: total distance (ft), avg and max speed (mph) |
| `clip_passes.csv` | Possession changes: passes, turnovers, unknowns |

---

## Step 4 — Render convex hull overlay (optional)

Requires Step 3 first. Adds offensive spacing hull, nearest-defender lines, and double-team rings to the video.

```bash
python3 -m src.run_video \
  --video data/clip.mp4 \
  --calibration calibrations/clip.json \
  --player-model runs/detect/runs/basketball_players_yolov8n/weights/best.pt \
  --output outputs/clip_overlay.mp4 \
  --csv-output outputs/clip_tracks.csv \
  --ball-trail-length 28 \
  --max-players 10 \
  --overlay-metrics
```

Produces `outputs/clip_overlay_overlay.mp4`. Overlay colors: green > 6 ft open, yellow 3–6 ft contested, red < 3 ft tight coverage.

---

## Step 5 — Generate report figures

```bash
# Model detection metrics table (Precision / Recall / mAP)
python3 -m tests.model_metrics

# All figures + analytics summary (fast, no videos needed)
python3 -m tests.report_analysis --skip-frames

# Full version including annotated frame screenshots
python3 -m tests.report_analysis
```

Outputs in `tests/report_figures/`:
- `nearest_defender_dist.png` — histogram of nearest-defender distances across clips
- `hull_area_over_time.png` — offensive spacing over time per clip
- `calib_<clip>.jpg` — calibration homography review images
- `<clip>/frame_NNNNN.jpg` — annotated frame screenshots from overlay videos

---

## Full end-to-end (all clips at once)

Edit `HAWKS_OFFENSE` / `PHI_OFFENSE` at the top of the script to match whichever team has the ball, then:

```bash
bash tests/run_all.sh
```

---

## CSV Schema

### `_tracks.csv`
| Column | Description |
|---|---|
| `frame` | Frame number |
| `object_type` | `player` or `ball` |
| `track_id` | ByteTrack persistent ID (-1 for ball) |
| `team_id` | 0 or 1 (jersey cluster), -1 if unlocked |
| `frame_x/y` | Pixel position in broadcast frame |
| `court_x/y` | Position in top-down court pixels (20 px/ft) |
| `ball_airborne` | True if ball not inside any player box |
| `possessing_ball` | True for the player holding the ball; possessor track_id on ball rows |

### `_metrics.csv`
Per-frame spacing stats: `offense_hull_area_ft2`, `defense_hull_area_ft2`, `defense_compactness_ft`, `mean/min/max_nearest_def_ft`, `corner_occ_left/right/total`, `n_double_teamed`

### `_speed.csv`
Per player: `total_distance_ft`, `avg_speed_mph`, `max_speed_mph`, `frames_tracked`

### `_passes.csv`
Per possession change: `from_track_id`, `to_track_id`, `from/to_team_id`, `event_type` (`pass` / `turnover` / `unknown`)

---

## Architecture

| Module | Role |
|---|---|
| `src/detector.py` | YOLO wrapper; ByteTrack + IoU fallback tracker |
| `src/homography.py` | Homography estimation and point projection |
| `src/court.py` | NBA court geometry constants and top-down image generation |
| `src/team_classifier.py` | Jersey-color k-means with overlap contamination guard and dominant-color histogram features |
| `src/smoothing.py` | `TrackSmoother` (EMA per player), `BallSmoother` (velocity-aware with spike filter and gap interpolation) |
| `src/spacing_pipeline.py` | Main per-frame loop: detection → projection → classification → CSV |
| `src/metrics.py` | Post-processing: convex hull, nearest-defender, speed/distance, pass detection |
| `src/calibrate.py` | Interactive GUI calibration tool |
| `src/auto_calibrate_boxes.py` | Automatic calibration from court landmark model |
| `src/dynamic_homography.py` | Frame-to-frame homography refinement via optical flow |
| `src/visualize.py` | All drawing: bounding boxes, top-down court, metric overlays |
| `src/run_video.py` | CLI entry point orchestrating calibration + pipeline |
| `tests/model_metrics.py` | Print YOLO validation metrics from training results |
| `tests/report_analysis.py` | Generate all report figures and analytics summary table |
| `tests/run_all.sh` | End-to-end script for all clips |

---

## Known Issues

- **ID accumulation** — ByteTrack issues new IDs on camera cuts so track IDs grow over long clips. Team classification is unaffected once tracks lock.
- **Team classification near clusters** — players from opposite teams standing together can temporarily share a team label. Overlap contamination guard reduces but does not eliminate this.
- **Auto-calibration failure** — the court model reliably finds paint landmarks but may fail on extreme broadcast angles. Manual calibration is always more robust. Try `--no-paint-only` or `--hoop-side left` before falling back.
- **Player recall** — the player model (recall 0.708) misses ~3 in 10 players in some frames, which inflates nearest-defender distances. More training data would improve this.
- **Pass/turnover accuracy** — depends on team classification being correct. Events labeled `unknown` in `_passes.csv` indicate frames where a player's team was not yet locked.
