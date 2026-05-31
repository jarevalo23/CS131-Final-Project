from __future__ import annotations

from collections import Counter, defaultdict, deque

import cv2
import numpy as np


def _boxes_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _overlap_mask(boxes: np.ndarray, iou_threshold: float = 0.12) -> np.ndarray:
    """Return bool array: True where a box overlaps any other box above threshold.

    Players whose bounding boxes overlap have contaminated jersey crops —
    the crop for player A includes pixels from player B's jersey. We skip
    votes and prototype updates for these detections.
    """
    n = len(boxes)
    overlapped = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if _boxes_iou(boxes[i], boxes[j]) >= iou_threshold:
                overlapped[i] = True
                overlapped[j] = True
    return overlapped


class JerseyTeamClassifier:
    def __init__(
        self,
        prototype_alpha: float = 0.08,
        min_margin: float = 4.0,
        vote_window: int = 15,
        lock_min_votes: int = 8,
        lock_majority_ratio: float = 0.75,
        show_unlocked_votes: bool = False,
        overlap_iou_threshold: float = 0.12,
    ) -> None:
        self.prototype_alpha = prototype_alpha
        self.min_margin = min_margin
        self.lock_min_votes = lock_min_votes
        self.lock_majority_ratio = lock_majority_ratio
        self.show_unlocked_votes = show_unlocked_votes
        self.overlap_iou_threshold = overlap_iou_threshold
        self.prototypes: np.ndarray | None = None
        self.track_votes: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=vote_window))
        self.locked_track_teams: dict[int, int] = {}

    def classify(self, frame: np.ndarray, boxes: np.ndarray, track_ids: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.empty((0,), dtype=np.int32)

        overlapped = _overlap_mask(boxes, self.overlap_iou_threshold)
        features: list[np.ndarray | None] = [jersey_feature(frame, box) for box in boxes]

        # Only initialize / update prototypes from non-overlapping detections.
        clean_features = [
            f for f, ov in zip(features, overlapped) if f is not None and not ov
        ]
        if self.prototypes is None and len(clean_features) >= 2:
            self._initialize_prototypes(np.array(clean_features, dtype=np.float32))

        team_ids: list[int] = []
        for feature, track_id, is_overlapped in zip(features, track_ids, overlapped):
            if self.prototypes is None:
                team_ids.append(-1)
                continue
            if track_id >= 0 and int(track_id) in self.locked_track_teams:
                team_ids.append(self.locked_track_teams[int(track_id)])
                continue

            team_id = -1
            if feature is not None:
                distances = np.linalg.norm(self.prototypes - feature, axis=1)
                order = np.argsort(distances)
                best = int(order[0])
                second = int(order[1])
                margin = distances[second] - distances[best]
                if margin >= self.min_margin:
                    team_id = best
                    # Only update prototypes from clean (non-overlapping) detections
                    # with a strong margin, to prevent contaminated crops from
                    # pulling prototypes toward each other over time.
                    if not is_overlapped and margin >= self.min_margin * 2:
                        self.prototypes[best] = (
                            (1.0 - self.prototype_alpha) * self.prototypes[best]
                            + self.prototype_alpha * feature
                        )

            # Only cast votes from non-overlapping detections so that two players
            # standing together don't incorrectly reinforce a wrong team assignment.
            if team_id >= 0 and track_id >= 0 and not is_overlapped:
                track_key = int(track_id)
                self.track_votes[track_key].append(team_id)
                team_id = self._stable_team_for_track(track_key)
            elif track_id >= 0 and self.track_votes[int(track_id)]:
                # Player is overlapping or ambiguous — use existing vote history.
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

    # Crop to jersey region: trim sides and focus on torso (18–60% of height).
    # A tighter horizontal crop (22%) reduces arm pixels which are often skin
    # tone and do not carry team-color signal.
    crop_x1 = max(0, x1 + int(0.22 * box_w))
    crop_x2 = min(width, x2 - int(0.22 * box_w))
    crop_y1 = max(0, y1 + int(0.18 * box_h))
    crop_y2 = min(height, y1 + int(0.60 * box_h))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    h, s, v = cv2.split(hsv)

    # Exclude very dark pixels (shadows/occlusions) and unsaturated near-black.
    mask = (v > 45) & ((s > 28) | (v > 145))
    pixels = lab[mask]
    hsv_pixels = hsv[mask]
    if len(pixels) < 20:
        return None

    # Dominant color via histogram peak is more robust than median for mixed
    # crops (two players overlapping): the dominant jersey color wins the bin
    # vote rather than being pulled toward a blend of both colors.
    lab_dominant = _dominant_color_lab(pixels)
    hsv_median = np.median(hsv_pixels[:, 1:3], axis=0).astype(np.float32)
    return np.concatenate([lab_dominant, hsv_median]).astype(np.float32)


def _dominant_color_lab(pixels: np.ndarray, bins: int = 16) -> np.ndarray:
    """Return the LAB color at the peak bin of a 3D histogram.

    More robust than median when the crop contains a mix of two jersey colors:
    the larger-area jersey dominates the histogram peak instead of the result
    drifting toward a blend midpoint.
    """
    if len(pixels) == 0:
        return np.zeros(3, dtype=np.float32)
    # L: 0-255, a: 0-255, b: 0-255 in OpenCV's uint8 LAB encoding.
    ranges = [(0, 256), (0, 256), (0, 256)]
    hist, edges = np.histogramdd(pixels.astype(np.float32), bins=bins, range=ranges)
    peak = np.unravel_index(int(np.argmax(hist)), hist.shape)
    dominant = np.array(
        [(edges[i][peak[i]] + edges[i][peak[i] + 1]) / 2.0 for i in range(3)],
        dtype=np.float32,
    )
    return dominant
