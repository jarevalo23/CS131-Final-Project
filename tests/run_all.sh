#!/usr/bin/env bash
# run_all.sh — Full pipeline for both clips, then all report outputs.
#
# Run from the project root:
#   bash tests/run_all.sh
#
# Prerequisites:
#   - .venv activated (source .venv/bin/activate)
#   - calibration JSONs exist for both clips (already in calibrations/)
#   - pip install matplotlib (if not already installed)
#
# What this produces:
#   outputs/hawks_overlay.mp4          side-by-side tracking video
#   outputs/hawks_overlay_overlay.mp4  + convex hull overlay
#   outputs/hawks_tracks.csv           per-frame player/ball positions
#   outputs/hawks_metrics.csv          spacing metrics
#   outputs/hawks_nearest_defender.csv nearest-defender distances
#   outputs/hawks_speed.csv            per-player speed & distance
#   outputs/hawks_passes.csv           pass / turnover events
#   (same set for phi_*)
#   tests/report_figures/              all report figures and screenshots

set -e  # stop on first error

HAWKS_CAL="calibrations/ny_hawks_game6_away_auto_pipeline.json"
PHI_CAL="calibrations/ny_phi_2ndq_auto_pipeline.json"

# Change --offense-team if you watch the overlay and find team 1 has the ball.
HAWKS_OFFENSE=0
PHI_OFFENSE=0

echo "============================================================"
echo " STEP 1: Run tracking pipeline — Hawks G6"
echo "============================================================"
python3 -m src.run_video \
  --video data/ny_hawks_game6_away.mp4 \
  --calibration "$HAWKS_CAL" \
  --output outputs/hawks_overlay.mp4 \
  --csv-output outputs/hawks_tracks.csv \
  --ball-trail-length 28 \
  --max-players 10

echo ""
echo "============================================================"
echo " STEP 2: Run tracking pipeline — NYK vs PHI"
echo "============================================================"
python3 -m src.run_video \
  --video data/ny_phi_2ndq.mp4 \
  --calibration "$PHI_CAL" \
  --output outputs/phi_overlay.mp4 \
  --csv-output outputs/phi_tracks.csv \
  --ball-trail-length 28 \
  --max-players 10

echo ""
echo "============================================================"
echo " STEP 3: Compute spacing metrics — Hawks G6"
echo "============================================================"
python3 -m src.metrics \
  --tracks outputs/hawks_tracks.csv \
  --offense-team "$HAWKS_OFFENSE" \
  --fps 30

echo ""
echo "============================================================"
echo " STEP 4: Compute spacing metrics — NYK vs PHI"
echo "============================================================"
python3 -m src.metrics \
  --tracks outputs/phi_tracks.csv \
  --offense-team "$PHI_OFFENSE" \
  --fps 30

echo ""
echo "============================================================"
echo " STEP 5: Re-render with convex hull overlay — Hawks G6"
echo "============================================================"
python3 -m src.run_video \
  --video data/ny_hawks_game6_away.mp4 \
  --calibration "$HAWKS_CAL" \
  --output outputs/hawks_overlay.mp4 \
  --csv-output outputs/hawks_tracks.csv \
  --ball-trail-length 28 \
  --max-players 10 \
  --overlay-metrics

echo ""
echo "============================================================"
echo " STEP 6: Re-render with convex hull overlay — NYK vs PHI"
echo "============================================================"
python3 -m src.run_video \
  --video data/ny_phi_2ndq.mp4 \
  --calibration "$PHI_CAL" \
  --output outputs/phi_overlay.mp4 \
  --csv-output outputs/phi_tracks.csv \
  --ball-trail-length 28 \
  --max-players 10 \
  --overlay-metrics

echo ""
echo "============================================================"
echo " STEP 7: Model detection metrics table"
echo "============================================================"
python3 -m tests.model_metrics

echo ""
echo "============================================================"
echo " STEP 8: Generate report figures and analytics summary"
echo "============================================================"
python3 -m tests.report_analysis

echo ""
echo "============================================================"
echo " Done. Outputs:"
echo "   outputs/         — videos and CSVs"
echo "   tests/report_figures/  — figures for the report"
echo "============================================================"
