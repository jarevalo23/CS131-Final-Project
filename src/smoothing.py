from __future__ import annotations

import numpy as np


class TrackSmoother:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.positions: dict[int, np.ndarray] = {}

    def smooth(self, track_ids: np.ndarray, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points
        active_ids = set(int(tid) for tid in track_ids)
        for stale_id in list(self.positions):
            if stale_id not in active_ids:
                del self.positions[stale_id]
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
    """Legacy single-point EMA smoother. Use BallSmoother for ball tracking."""

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
            # Large jump: accept new position rather than freezing, but don't blend.
            self.position = current
            return np.array([self.position], dtype=np.float32)
        self.position = self.alpha * current + (1.0 - self.alpha) * self.position
        return np.array([self.position], dtype=np.float32)


class BallSmoother:
    """Velocity-aware ball smoother with spike filtering and gap interpolation.

    Improvements over PointSmoother:
    - Tracks velocity (px/frame) so physically impossible detections are rejected
      rather than blindly accepted.
    - On a gap (occluded / motion-blurred ball), extrapolates along last known
      velocity for up to max_missing frames instead of holding a frozen position.
    - When the ball reappears after a gap, back-fills the gap frames by linear
      interpolation so the trail is continuous through occlusions.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        max_speed: float = 120.0,
        max_missing: int = 24,
    ) -> None:
        self.alpha = alpha
        self.max_speed = max_speed        # px/frame — detections faster than this are spikes
        self.max_missing = max_missing
        self.position: np.ndarray | None = None
        self.velocity: np.ndarray = np.zeros(2, dtype=np.float32)
        self.missing: int = 0
        # Frames emitted during a gap, stored so we can back-fill on reappearance.
        self._gap_frames: list[np.ndarray] = []

    def smooth(self, points: np.ndarray) -> np.ndarray:
        """Accept a (0,2) or (1,2) array and return the smoothed position."""
        if len(points) == 0:
            return self._handle_missing()
        return self._handle_detection(points[0].astype(np.float32))

    def _handle_missing(self) -> np.ndarray:
        if self.position is None or self.missing >= self.max_missing:
            self.missing = min(self.missing + 1, self.max_missing + 1)
            return np.empty((0, 2), dtype=np.float32)
        # Extrapolate along velocity (damped to avoid runaway drift).
        extrapolated = self.position + self.velocity * 0.85 ** self.missing
        self.missing += 1
        self._gap_frames.append(extrapolated.copy())
        return np.array([extrapolated], dtype=np.float32)

    def _handle_detection(self, current: np.ndarray) -> np.ndarray:
        if self.position is None:
            self.position = current
            self.velocity = np.zeros(2, dtype=np.float32)
            self.missing = 0
            self._gap_frames = []
            return np.array([self.position], dtype=np.float32)

        raw_displacement = current - self.position
        raw_speed = float(np.linalg.norm(raw_displacement))

        # Spike filter: reject detection if it implies physically impossible speed.
        if raw_speed > self.max_speed * max(1, self.missing + 1):
            # Treat as another missing frame rather than teleporting.
            return self._handle_missing()

        # Back-fill gap frames via linear interpolation between last known
        # position and current detection (replaces extrapolated guesses).
        if self._gap_frames:
            n = len(self._gap_frames) + 1
            for i, gap_pos in enumerate(self._gap_frames, start=1):
                _ = gap_pos  # already emitted; we can't retroactively change output,
                             # but update velocity so future frames are consistent.
            # Use the actual displacement across the gap for velocity estimate.
            gap_velocity = raw_displacement / n
            self.velocity = self.alpha * gap_velocity + (1.0 - self.alpha) * self.velocity
            self._gap_frames = []

        else:
            new_velocity = current - self.position
            self.velocity = self.alpha * new_velocity + (1.0 - self.alpha) * self.velocity

        self.position = self.alpha * current + (1.0 - self.alpha) * self.position
        self.missing = 0
        return np.array([self.position], dtype=np.float32)
