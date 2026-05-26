from __future__ import annotations

import numpy as np


class TrackSmoother:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.positions: dict[int, np.ndarray] = {}

    def smooth(self, track_ids: np.ndarray, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points
        smoothed = points.copy()
        for index, track_id in enumerate(track_ids):
            key = int(track_id)
            current = points[index]
            previous = self.positions.get(key)
            if previous is None:
                output = current
            else:
                output = self.alpha * current + (1.0 - self.alpha) * previous
            self.positions[key] = output
            smoothed[index] = output
        return smoothed


class PointSmoother:
    def __init__(self, alpha: float, max_jump: float | None = None, max_missing: int = 0) -> None:
        self.alpha = alpha
        self.max_jump = max_jump
        self.max_missing = max_missing
        self.position: np.ndarray | None = None
        self.missing = 0

    def smooth(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            if self.position is not None and self.missing < self.max_missing:
                self.missing += 1
                return np.array([self.position], dtype=np.float32)
            return points
        current = points[0]
        self.missing = 0
        if self.position is None:
            self.position = current
            return points
        if self.max_jump is not None and np.linalg.norm(current - self.position) > self.max_jump:
            return np.array([self.position], dtype=np.float32)
        self.position = self.alpha * current + (1.0 - self.alpha) * self.position
        return np.array([self.position], dtype=np.float32)
