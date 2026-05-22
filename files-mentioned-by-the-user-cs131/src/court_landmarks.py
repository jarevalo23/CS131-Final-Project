from __future__ import annotations

from src.calibrate import landmarks_for_hoop_side


def landmark_labels(hoop_side: str) -> list[str]:
    return [label for label, _point in landmarks_for_hoop_side(hoop_side)]


def landmark_court_points(hoop_side: str) -> list[tuple[float, float]]:
    return [point for _label, point in landmarks_for_hoop_side(hoop_side)]

