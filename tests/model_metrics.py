"""Print model detection performance from YOLO training results.

Reads results.csv produced by each training run and prints the best-epoch
Precision, Recall, mAP50, and mAP50-95 for all three models in a table.

Usage:
    python3 -m tests.model_metrics
"""

from __future__ import annotations

import csv
from pathlib import Path


MODELS = [
    (
        "Player detection",
        Path("runs/detect/runs/basketball_players_yolov8n/results.csv"),
    ),
    (
        "Court landmark detection",
        Path("runs/detect/runs/court/nba_court_yolov8n/results.csv"),
    ),
]


def best_epoch(path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(path.open()))
    best = max(rows, key=lambda r: float(r["metrics/mAP50(B)"]))
    return {
        "precision": float(best["metrics/precision(B)"]),
        "recall":    float(best["metrics/recall(B)"]),
        "mAP50":     float(best["metrics/mAP50(B)"]),
        "mAP50-95":  float(best["metrics/mAP50-95(B)"]),
        "epoch":     int(float(best["epoch"])),
    }


def main() -> None:
    print(f"\n{'Model':<30}  {'Epoch':>5}  {'Precision':>9}  {'Recall':>6}  {'mAP50':>5}  {'mAP50-95':>8}")
    print("-" * 70)
    for label, path in MODELS:
        if not path.exists():
            print(f"{label:<30}  results.csv not found at {path}")
            continue
        m = best_epoch(path)
        print(
            f"{label:<30}  {m['epoch']:>5}  {m['precision']:>9.3f}  "
            f"{m['recall']:>6.3f}  {m['mAP50']:>5.3f}  {m['mAP50-95']:>8.3f}"
        )
    print()


if __name__ == "__main__":
    main()
