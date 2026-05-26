from __future__ import annotations

import cv2
import numpy as np


NBA_COURT_WIDTH_FT = 50.0
NBA_HALF_COURT_LENGTH_FT = 47.0
NBA_HOOP_FROM_BASELINE_FT = 5.25
NBA_THREE_POINT_RADIUS_FT = 23.75
NBA_CORNER_THREE_Y_OFFSET_FT = 22.0
DEFAULT_COURT_SIZE = (940, 1000)


def court_x(x_ft: float, width: int, hoop_side: str) -> int:
    if hoop_side == "left":
        return int(width * x_ft / NBA_HALF_COURT_LENGTH_FT)
    if hoop_side == "right":
        return int(width * (NBA_HALF_COURT_LENGTH_FT - x_ft) / NBA_HALF_COURT_LENGTH_FT)
    raise ValueError(f"Unsupported hoop side: {hoop_side}")


def hoop_pixel_x(size: tuple[int, int] = DEFAULT_COURT_SIZE, hoop_side: str = "left") -> int:
    return court_x(NBA_HOOP_FROM_BASELINE_FT, size[0], hoop_side)


def hoop_pixel_y(size: tuple[int, int] = DEFAULT_COURT_SIZE) -> int:
    return size[1] // 2


def top_down_court(size: tuple[int, int] = DEFAULT_COURT_SIZE, hoop_side: str = "left") -> np.ndarray:
    """Return a simple blank half-court image for visualization."""
    width, height = size
    court = np.full((height, width, 3), (238, 203, 150), dtype=np.uint8)
    line = (255, 255, 255)

    cv2.rectangle(court, (0, 0), (width - 1, height - 1), line, 2)
    cv2.line(court, (0, height // 2), (width, height // 2), line, 1)

    key_start_x = court_x(0, width, hoop_side)
    key_end_x = court_x(19, width, hoop_side)
    key_h = int(height * 16 / NBA_COURT_WIDTH_FT)
    key_y0 = height // 2 - key_h // 2
    key_y1 = height // 2 + key_h // 2
    cv2.rectangle(court, (min(key_start_x, key_end_x), key_y0), (max(key_start_x, key_end_x), key_y1), line, 2)

    hoop_x = court_x(NBA_HOOP_FROM_BASELINE_FT, width, hoop_side)
    cv2.circle(court, (hoop_x, height // 2), 8, line, 2)
    cv2.circle(court, (court_x(19, width, hoop_side), height // 2), 60, line, 2)

    corner_y_top = int(height * (NBA_COURT_WIDTH_FT / 2 - NBA_CORNER_THREE_Y_OFFSET_FT) / NBA_COURT_WIDTH_FT)
    corner_y_bottom = int(height * (NBA_COURT_WIDTH_FT / 2 + NBA_CORNER_THREE_Y_OFFSET_FT) / NBA_COURT_WIDTH_FT)
    arc_start_ft = NBA_HOOP_FROM_BASELINE_FT + (
        NBA_THREE_POINT_RADIUS_FT**2 - NBA_CORNER_THREE_Y_OFFSET_FT**2
    ) ** 0.5
    arc_start_x = court_x(arc_start_ft, width, hoop_side)
    baseline_x = court_x(0, width, hoop_side)
    cv2.line(court, (baseline_x, corner_y_top), (arc_start_x, corner_y_top), line, 2)
    cv2.line(court, (baseline_x, corner_y_bottom), (arc_start_x, corner_y_bottom), line, 2)

    radius_x = int(width * NBA_THREE_POINT_RADIUS_FT / NBA_HALF_COURT_LENGTH_FT)
    radius_y = int(height * NBA_THREE_POINT_RADIUS_FT / NBA_COURT_WIDTH_FT)
    if hoop_side == "left":
        cv2.ellipse(court, (hoop_x, height // 2), (radius_x, radius_y), 0, -68, 68, line, 2)
    else:
        cv2.ellipse(court, (hoop_x, height // 2), (radius_x, radius_y), 0, 112, 248, line, 2)
    return court


def court_feet_to_pixels(points_ft: np.ndarray, size: tuple[int, int] = DEFAULT_COURT_SIZE) -> np.ndarray:
    width, height = size
    x = points_ft[:, 0] / NBA_HALF_COURT_LENGTH_FT * width
    y = points_ft[:, 1] / NBA_COURT_WIDTH_FT * height
    return np.column_stack([x, y]).astype(np.float32)
