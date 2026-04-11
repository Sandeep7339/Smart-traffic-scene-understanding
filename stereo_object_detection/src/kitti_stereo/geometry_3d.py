from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


def box3d_corners_camera(
    x_m: float,
    y_m: float,
    z_m: float,
    height_m: float,
    width_m: float,
    length_m: float,
    yaw_ry: float,
) -> np.ndarray:
    x_corners = np.array(
        [length_m / 2, length_m / 2, -length_m / 2, -length_m / 2, length_m / 2, length_m / 2, -length_m / 2, -length_m / 2],
        dtype=np.float32,
    )
    y_corners = np.array([0.0, 0.0, 0.0, 0.0, -height_m, -height_m, -height_m, -height_m], dtype=np.float32)
    z_corners = np.array(
        [width_m / 2, -width_m / 2, -width_m / 2, width_m / 2, width_m / 2, -width_m / 2, -width_m / 2, width_m / 2],
        dtype=np.float32,
    )

    cos_y = math.cos(yaw_ry)
    sin_y = math.sin(yaw_ry)
    rot = np.array(
        [
            [cos_y, 0.0, sin_y],
            [0.0, 1.0, 0.0],
            [-sin_y, 0.0, cos_y],
        ],
        dtype=np.float32,
    )

    corners_obj = np.stack([x_corners, y_corners, z_corners], axis=0)
    corners_cam = (rot @ corners_obj).T
    corners_cam[:, 0] += x_m
    corners_cam[:, 1] += y_m
    corners_cam[:, 2] += z_m
    return corners_cam


def project_points_p2(points_xyz: np.ndarray, p2_matrix: np.ndarray) -> np.ndarray:
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float32)
    points_h = np.concatenate([points_xyz.astype(np.float32), ones], axis=1)
    proj = (p2_matrix.astype(np.float32) @ points_h.T).T
    uv = np.full((points_xyz.shape[0], 2), np.nan, dtype=np.float32)
    valid = np.abs(proj[:, 2]) > 1e-6
    uv[valid, 0] = proj[valid, 0] / proj[valid, 2]
    uv[valid, 1] = proj[valid, 1] / proj[valid, 2]
    return uv


def draw_projected_box3d(image_bgr: np.ndarray, corners_2d: np.ndarray, cv2_module, color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
    output = image_bgr
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        p1 = corners_2d[a]
        p2 = corners_2d[b]
        if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
            continue
        x1, y1 = int(round(float(p1[0]))), int(round(float(p1[1])))
        x2, y2 = int(round(float(p2[0]))), int(round(float(p2[1])))
        cv2_module.line(output, (x1, y1), (x2, y2), color, thickness, cv2_module.LINE_AA)
    return output


def bbox_iou_xyxy(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return inter / union


def valid_finite_points(points: Iterable[np.ndarray]) -> np.ndarray:
    arr = np.asarray(list(points), dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return arr[np.isfinite(arr).all(axis=1)]
