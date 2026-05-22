# CS131 Basketball Tracking Starter

This starter is organized around the first useful MVP from the proposal:

1. Load a short basketball clip.
2. Select or detect court landmarks.
3. Estimate a homography from broadcast view to a top-down court.
4. Detect players with YOLO.
5. Project player locations onto the court.
6. Save an annotated video.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put clips in `data/`, then run:

```bash
python -m src.main --video data/clip.mp4 --output outputs/annotated.mp4
```

## One-Command Auto Pipeline

After training/downloading the player and court models, run one command to auto-calibrate a clip and render the player/team/ball visualization:

```bash
python -m src.run_video --video data/sample3.mp4
```

This writes:

- `calibrations/sample3_auto_pipeline.json`
- `calibrations/sample3_auto_pipeline_review.jpg`
- `outputs/sample3_auto_pipeline.mp4`
- `outputs/sample3_auto_pipeline.csv`

For a quick test, limit frames:

```bash
python -m src.run_video --video data/sample3.mp4 --max-frames 120
```

If auto-calibration is wrong for a clip, pass an existing manual calibration:

```bash
python -m src.run_video --video data/sample2.mp4 --calibration calibrations/sample2.json
```

## Calibrate Court Landmarks

Before running the full pipeline, click landmarks on one frame:

```bash
python -m src.calibrate --video data/clip.mp4 --output calibrations/clip.json
```

For sample 2, click the four half-court corners in the order shown on screen:

1. midcourt near sideline
2. baseline near corner
3. baseline far corner
4. midcourt far sideline

The near sideline is the lower sideline in the broadcast frame. It maps to the bottom of the top-down court.

The calibration helper draws temporary numbered markers as you click. Press `s` to save, `r` to reset, or `q` to quit. It saves both `calibrations/clip.json` and a marked screenshot at `calibrations/clip.jpg`.
It also saves `calibrations/clip_review.jpg`, which overlays the warped court back onto the video frame. If that overlay misses the real court lines, recalibrate.
Press `k` to skip a hidden landmark and `u` to undo. Use as many visible landmarks as possible; the homography uses RANSAC when there are 5 or more points.
The tool also saves `calibrations/clip_errors.txt`. Recalibrate if the max error is high or if one landmark has a much larger error than the others.

Then run with that calibration:

```bash
python -m src.main --video data/clip.mp4 --calibration calibrations/clip.json --output outputs/annotated.mp4
```

This processes the full clip by default. For quick tests, limit the number of frames:

```bash
python -m src.main --video data/clip.mp4 --calibration calibrations/clip.json --output outputs/annotated.mp4 --max-frames 300
```

By default, calibration assumes the hoop/baseline is on the right. This matches sample 2. If your clip needs the hoop on the left, add this when calibrating:

```bash
python -m src.calibrate --video data/clip.mp4 --output calibrations/clip.json --hoop-side left
```

Detections are filtered after projection, so people whose foot points land outside the calibrated court are removed. If sideline people still appear, increase the margin:

```bash
python -m src.main --video data/clip.mp4 --calibration calibrations/clip.json --output outputs/annotated.mp4 --court-margin 25
```

## Article Steps 1-4 Mode

To follow the Medium article's first four steps, use tracking IDs and optionally filter to offensive players:

```bash
python -m src.main \
  --article-steps-1-4 \
  --video data/sample2.mp4 \
  --calibration calibrations/sample2.json \
  --output outputs/sample2_steps_1_4.mp4 \
  --csv-output outputs/sample2_tracks.csv
```

After the first run, look at the IDs drawn above players. Then rerun with only the offensive IDs:

```bash
python -m src.main \
  --article-steps-1-4 \
  --video data/sample2.mp4 \
  --calibration calibrations/sample2.json \
  --output outputs/sample2_offense.mp4 \
  --csv-output outputs/sample2_offense.csv \
  --offense-ids 1,4,7,9,12
```

If you have custom basketball YOLO weights, pass them with `--model path/to/best.pt`. If the projected corner players are stretched too far from the center of the court, try a compression factor below 1:

```bash
python -m src.main --article-steps-1-4 --video data/sample2.mp4 --calibration calibrations/sample2.json --output outputs/sample2_compressed.mp4 --compression-factor 0.85
```

## Train Roboflow Dataset Locally

The downloaded Roboflow dataset is in `Basketball Players.v1i.yolov11`. A local YAML is provided at:

```bash
Basketball Players.v1i.yolov11/data.local.yaml
```

Train a small YOLO model:

```bash
yolo detect train \
  model=yolo11n.pt \
  data="Basketball Players.v1i.yolov11/data.local.yaml" \
  epochs=50 \
  imgsz=640 \
  project=runs \
  name=basketball_players_yolo11n
```

After training, use the best weights:

```bash
python -m src.main \
  --article-steps-1-4 \
  --video data/sample2.mp4 \
  --calibration calibrations/sample2.json \
  --model runs/basketball_players_yolo11n/weights/best.pt \
  --player-class-ids 3 \
  --hoop-class-ids 1 \
  --auto-orient-hoop \
  --player-smoothing-alpha 0.35 \
  --ball-smoothing-alpha 0.2 \
  --ball-max-jump 180 \
  --ball-max-missing 8 \
  --keep-outside-court \
  --dynamic-homography \
  --output outputs/sample2_custom_model.mp4
```

`--dynamic-homography` tracks the clicked floor landmarks frame-to-frame with optical flow and updates the homography as the broadcast camera shifts. It still needs an initial calibration for a new camera angle.

## Court Landmark Model

To move toward automatic calibration, export your clicked calibration frames into a YOLO pose/keypoint dataset:

```bash
python -m src.export_court_landmark_dataset \
  --calibrations calibrations \
  --output court_landmark_dataset \
  --hoop-side right \
  --clean
```

The dataset is written to `court_landmark_dataset/data.yaml`. It uses one `court` object with 12 keypoints:

```text
midcourt near sideline
baseline near corner
baseline far corner
midcourt far sideline
near lane baseline corner
far lane baseline corner
near elbow / FT line lane
far elbow / FT line lane
FT line center
top of 3PT arc
near 3PT corner/arc break
far 3PT corner/arc break
```

Train a pose model once enough calibration frames exist:

```bash
yolo pose train \
  model=yolov8n-pose.pt \
  data=court_landmark_dataset/data.yaml \
  epochs=100 \
  imgsz=640 \
  project=runs \
  name=court_landmarks_pose
```

Then auto-generate a calibration file for a new clip:

```bash
python -m src.auto_calibrate \
  --video data/sample3.mp4 \
  --output calibrations/sample3_auto.json \
  --model runs/pose/court_landmarks_pose/weights/best.pt \
  --hoop-side right
```

If using an object-detection court-landmark model like Roboflow `nba-court`, use box centers instead:

```bash
python -m src.auto_calibrate_boxes \
  --video data/sample3.mp4 \
  --output calibrations/sample3_court_boxes.json \
  --model path/to/nba_court_best.pt \
  --hoop-side right \
  --print-classes
```

If the class names differ from the aliases in `src/auto_calibrate_boxes.py`, update `RIGHT_HOOP_CLASS_TO_LANDMARK` with the printed class names.

You still need more labeled calibration frames before this model will be useful. Calibrate several frames per clip with `--frame`, for example:

```bash
python -m src.calibrate --video data/sample3.mp4 --frame 0 --output calibrations/sample3_f000.json --hoop-side right
python -m src.calibrate --video data/sample3.mp4 --frame 200 --output calibrations/sample3_f200.json --hoop-side right
python -m src.calibrate --video data/sample3.mp4 --frame 400 --output calibrations/sample3_f400.json --hoop-side right
```

Class ID `3` is `Player` and class ID `1` is `Hoop` in this dataset. `--auto-orient-hoop` uses the detected hoop to decide which side of the court is being shown and whether the displayed player dots need to be mirrored.

To also detect and project the ball, add:

```bash
--detect-ball --ball-class-ids 0
```

If the player dots are mirrored left-to-right on the court visualization, add:

```bash
--flip-display-x
```

If the near and far sidelines are swapped, add:

```bash
--flip-display-y
```

## Suggested Milestones

### Milestone 1: Manual Homography

Start here. Pause on one frame, manually click four or more court points, and verify that player feet project to plausible top-down locations.

### Milestone 2: Player Detection

Run YOLO on each frame, filter detections to people, and use the bottom-center of each bounding box as the player's court contact point.

### Milestone 3: Tracking and Teams

Add ByteTrack/BoT-SORT through Ultralytics tracking, then classify teams from jersey color crops.

### Milestone 4: Analytics

Once projected locations are stable, add spacing, nearest defender distance, convex hull area, and double-team heuristics.
