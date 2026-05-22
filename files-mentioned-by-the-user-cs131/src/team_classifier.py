from __future__ import annotations

from collections import Counter, defaultdict, deque

import cv2
import numpy as np


class JerseyTeamClassifier:
    def __init__(
        self,
        prototype_alpha: float = 0.08,
        min_margin: float = 4.0,
        vote_window: int = 15,
        lock_min_votes: int = 8,
        lock_majority_ratio: float = 0.75,
        show_unlocked_votes: bool = False,
    ) -> None:
        self.prototype_alpha = prototype_alpha
        self.min_margin = min_margin
        self.lock_min_votes = lock_min_votes
        self.lock_majority_ratio = lock_majority_ratio
        self.show_unlocked_votes = show_unlocked_votes
        self.prototypes: np.ndarray | None = None
        self.track_votes: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=vote_window))
        self.locked_track_teams: dict[int, int] = {}

    def classify(self, frame: np.ndarray, boxes: np.ndarray, track_ids: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.empty((0,), dtype=np.int32)

        features: list[np.ndarray | None] = [jersey_feature(frame, box) for box in boxes]
        valid_features = np.array([feature for feature in features if feature is not None], dtype=np.float32)
        if self.prototypes is None and len(valid_features) >= 2:
            self._initialize_prototypes(valid_features)

        team_ids: list[int] = []
        for feature, track_id in zip(features, track_ids):
            if track_id >= 0 and int(track_id) in self.locked_track_teams:
                team_ids.append(self.locked_track_teams[int(track_id)])
                continue

            team_id = -1
            if feature is not None and self.prototypes is not None:
                distances = np.linalg.norm(self.prototypes - feature, axis=1)
                order = np.argsort(distances)
                best = int(order[0])
                second = int(order[1])
                if distances[second] - distances[best] >= self.min_margin:
                    team_id = best
                    self.prototypes[best] = (
                        (1.0 - self.prototype_alpha) * self.prototypes[best]
                        + self.prototype_alpha * feature
                    )

            if team_id >= 0 and track_id >= 0:
                track_key = int(track_id)
                self.track_votes[track_key].append(team_id)
                team_id = self._stable_team_for_track(track_key)
            elif track_id >= 0 and self.track_votes[int(track_id)]:
                team_id = self._stable_team_for_track(int(track_id))
            team_ids.append(team_id)

        return np.array(team_ids, dtype=np.int32)

    def _initialize_prototypes(self, features: np.ndarray) -> None:
        compactness, labels, centers = cv2.kmeans(
            features.astype(np.float32),
            2,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5),
            5,
            cv2.KMEANS_PP_CENTERS,
        )
        _ = compactness, labels
        # Keep team colors stable across runs: team 0 is the darker jersey cluster.
        brightness_order = np.argsort(centers[:, 0])
        self.prototypes = centers[brightness_order].astype(np.float32)

    def _stable_team_for_track(self, track_id: int) -> int:
        if track_id in self.locked_track_teams:
            return self.locked_track_teams[track_id]

        votes = self.track_votes[track_id]
        team_id, count = most_common_with_count(votes)
        if len(votes) >= self.lock_min_votes and count / len(votes) >= self.lock_majority_ratio:
            self.locked_track_teams[track_id] = team_id
            return team_id
        if self.show_unlocked_votes:
            return team_id
        return -1


def most_common(values: deque[int]) -> int:
    return Counter(values).most_common(1)[0][0]


def most_common_with_count(values: deque[int]) -> tuple[int, int]:
    return Counter(values).most_common(1)[0]


def jersey_feature(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    crop_x1 = max(0, x1 + int(0.18 * box_w))
    crop_x2 = min(width, x2 - int(0.18 * box_w))
    crop_y1 = max(0, y1 + int(0.18 * box_h))
    crop_y2 = min(height, y1 + int(0.62 * box_h))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    h, s, v = cv2.split(hsv)
    _ = h

    # Jerseys can be saturated colors or bright whites. Very dark pixels are
    # usually shadows/text/occlusions and are poor team-color evidence.
    mask = (v > 45) & ((s > 28) | (v > 145))
    pixels = lab[mask]
    hsv_pixels = hsv[mask]
    if len(pixels) < 20:
        return None

    lab_median = np.median(pixels, axis=0).astype(np.float32)
    hsv_median = np.median(hsv_pixels[:, 1:3], axis=0).astype(np.float32)
    return np.concatenate([lab_median, hsv_median]).astype(np.float32)
