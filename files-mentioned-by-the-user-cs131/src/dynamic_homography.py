from __future__ import annotations

import cv2
import numpy as np

from src.homography import estimate_homography


class DynamicHomographyTracker:
    def __init__(
        self,
        frame_points: np.ndarray,
        court_points_ft: np.ndarray,
        min_points: int = 4,
    ) -> None:
        self.frame_points = frame_points.astype(np.float32)
        self.court_points_ft = court_points_ft.astype(np.float32)
        self.min_points = min_points
        self.previous_gray: np.ndarray | None = None
        self.current_homography = estimate_homography(self.frame_points, self.court_points_ft)

    def update(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.previous_gray is None:
            self.previous_gray = gray
            return self.current_homography

        next_points, status, _error = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            self.frame_points.reshape(-1, 1, 2),
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.previous_gray = gray
        if next_points is None or status is None:
            return self.current_homography

        status_mask = status.reshape(-1).astype(bool)
        if int(status_mask.sum()) < self.min_points:
            return self.current_homography

        tracked_points = next_points.reshape(-1, 2)[status_mask].astype(np.float32)
        tracked_court_points = self.court_points_ft[status_mask].astype(np.float32)
        try:
            self.current_homography = estimate_homography(tracked_points, tracked_court_points)
            self.frame_points[status_mask] = tracked_points
        except ValueError:
            pass
        return self.current_homography
