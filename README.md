# CS131 Basketball Court Tracking

Converts NBA broadcast video into a top-down court map with player positions, team labels, and ball tracking.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

**1. Calibrate a clip**

Click court landmarks on a single frame to establish the homography. You need at least 4 — press `k` to skip any that are off-screen.

```bash
python -m src.calibrate --video data/clip.mp4 --output calibrations/clip.json
```

Check `calibrations/clip_review.jpg` to verify alignment. If the court overlay is way off, recalibrate. Add `--hoop-side left` if the basket is on the left side of the frame.

**2. Run the pipeline**

```bash
python -m src.main \
  --article-steps-1-4 \
  --video data/clip.mp4 \
  --calibration calibrations/clip.json \
  --output outputs/clip_annotated.mp4 \
  --csv-output outputs/clip_tracks.csv
```

Add `--max-frames 120` for a quick test run.

## Key flags

| Flag | Default | Effect |
|---|---|---|
| `--model` | yolov8n.pt | YOLO weights to use |
| `--player-class-ids` | person class | Class ID for players |
| `--detect-ball --ball-class-ids 0` | off | Enable ball detection |
| `--flip-display-x` | off | Mirror top-down map if players appear on wrong side |
| `--auto-orient-hoop` | off | Infer court orientation from detected hoop |
| `--dynamic-homography` | off | Update homography frame-to-frame with optical flow |
| `--court-margin 25` | 8 | Increase if sideline people bleed into court view |
| `--ball-smoothing-alpha` | 1.0 | Lower = smoother but more lag |

## Output

| File | Contents |
|---|---|
| `outputs/clip_annotated.mp4` | Annotated broadcast + top-down court side by side |
| `outputs/clip_tracks.csv` | Per-frame positions in pixel and court coordinates |
| `calibrations/clip_review.jpg` | Homography quality check |

CSV schema: `frame, object_type, track_id, team_id, frame_x, frame_y, court_x, court_y`

## Auto pipeline

If you have a trained court landmark model, calibration can run automatically:

```bash
python -m src.run_video --video data/clip.mp4
```

Pass `--calibration` to override with a manual calibration if auto fails.

## Known issues

- **ID accumulation** — ByteTrack issues new IDs on camera cuts so IDs grow over long clips. Team classification is unaffected.
- **Ball lag** — the smoothing filter trades jitter for a 1–2 frame position lag during fast movement. Raise `--ball-smoothing-alpha` to reduce lag.
- **Auto-calibration** — the court model reliably finds paint landmarks but misses sidelines and arc points on many broadcast angles. Manual calibration is more robust.