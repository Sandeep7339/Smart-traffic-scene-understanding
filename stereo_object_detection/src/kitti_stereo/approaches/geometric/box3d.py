from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ...detection import Detection2D


@dataclass(frozen=True)
class ObjectBox3D:
    class_id: int
    label: str
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]
    center_uv: Tuple[float, float]
    center_xyz: Tuple[float, float, float]
    dimensions_xyz: Tuple[float, float, float]
    bounds_xyz: Tuple[float, float, float, float, float, float]
    corners_3d: np.ndarray
    corners_2d: np.ndarray
    num_points: int
    depth_seed_m: float


def _clip_bbox(bbox_xyxy: Tuple[int, int, int, int], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _inner_bbox(bbox_xyxy: Tuple[int, int, int, int], inner_ratio: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1
    new_w = max(2, int(round(w * inner_ratio)))
    new_h = max(2, int(round(h * inner_ratio)))
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    ix1 = int(round(cx - new_w / 2.0))
    iy1 = int(round(cy - new_h / 2.0))
    ix2 = ix1 + new_w
    iy2 = iy1 + new_h
    return (ix1, iy1, ix2, iy2)


def _build_corners(bounds_xyz: Tuple[float, float, float, float, float, float]) -> np.ndarray:
    x_min, x_max, y_min, y_max, z_min, z_max = bounds_xyz
    return np.array(
        [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ],
        dtype=np.float32,
    )


def project_points_p2(points_xyz: np.ndarray, p2: np.ndarray) -> np.ndarray:
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float32)
    points_h = np.concatenate([points_xyz.astype(np.float32), ones], axis=1)
    proj = (p2 @ points_h.T).T

    uv = np.full((points_xyz.shape[0], 2), np.nan, dtype=np.float32)
    valid = np.abs(proj[:, 2]) > 1e-6
    uv[valid, 0] = proj[valid, 0] / proj[valid, 2]
    uv[valid, 1] = proj[valid, 1] / proj[valid, 2]
    return uv


def _seed_depth(depth_values: np.ndarray) -> Tuple[float, float]:
    median = float(np.median(depth_values))
    near_seed = float(np.percentile(depth_values, 35))

    mad = float(np.median(np.abs(depth_values - median)))
    if mad > 1e-6:
        sigma = 1.4826 * mad
        band = max(0.5, 2.5 * sigma)
    else:
        band = max(0.5, 0.15 * median)
    return near_seed, float(band)


def _fit_bounds_from_points(points_xyz: np.ndarray) -> Tuple[Tuple[float, float, float, float, float, float], np.ndarray]:
    lo = np.percentile(points_xyz, 2.0, axis=0)
    hi = np.percentile(points_xyz, 98.0, axis=0)
    inlier_mask = np.all((points_xyz >= lo) & (points_xyz <= hi), axis=1)
    filtered = points_xyz[inlier_mask] if np.any(inlier_mask) else points_xyz

    mins = filtered.min(axis=0)
    maxs = filtered.max(axis=0)
    eps = 1e-3
    maxs = np.maximum(maxs, mins + eps)

    bounds = (
        float(mins[0]),
        float(maxs[0]),
        float(mins[1]),
        float(maxs[1]),
        float(mins[2]),
        float(maxs[2]),
    )
    return bounds, filtered


def estimate_3d_box_from_detection(
    detection: Detection2D,
    depth_map_m: np.ndarray,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
    p2_matrix: np.ndarray,
    min_depth_m: float = 0.1,
    max_depth_m: float = 120.0,
    inner_ratio: float = 0.65,
    min_points: int = 50,
) -> Optional[ObjectBox3D]:
    h, w = depth_map_m.shape[:2]
    clipped = _clip_bbox(detection.bbox_xyxy, width=w, height=h)
    if clipped is None:
        return None

    x1, y1, x2, y2 = clipped
    depth_patch = depth_map_m[y1 : y2 + 1, x1 : x2 + 1]
    valid_full = np.isfinite(depth_patch) & (depth_patch > min_depth_m) & (depth_patch < max_depth_m)
    if not np.any(valid_full):
        return None

    inner = _clip_bbox(_inner_bbox(clipped, inner_ratio=inner_ratio), width=w, height=h)
    if inner is None:
        inner = clipped
    ix1, iy1, ix2, iy2 = inner
    inner_patch = depth_map_m[iy1 : iy2 + 1, ix1 : ix2 + 1]
    valid_inner = np.isfinite(inner_patch) & (inner_patch > min_depth_m) & (inner_patch < max_depth_m)

    depth_values = inner_patch[valid_inner] if np.any(valid_inner) else depth_patch[valid_full]
    seed_depth, seed_band = _seed_depth(depth_values.astype(np.float32))

    candidate = valid_full & (np.abs(depth_patch - seed_depth) <= seed_band)
    if int(np.sum(candidate)) < min_points:
        close_cutoff = float(np.percentile(depth_patch[valid_full], 60))
        candidate = valid_full & (depth_patch <= close_cutoff)
    if int(np.sum(candidate)) < min_points:
        candidate = valid_full
    if int(np.sum(candidate)) < max(10, min_points // 3):
        return None

    ys, xs = np.where(candidate)
    us = (xs + x1).astype(np.float32)
    vs = (ys + y1).astype(np.float32)
    zs = depth_patch[candidate].astype(np.float32)

    xs_m = (us - cx_px) * zs / fx_px
    ys_m = (vs - cy_px) * zs / fy_px
    points_xyz = np.stack([xs_m, ys_m, zs], axis=1)

    bounds, filtered_points = _fit_bounds_from_points(points_xyz)
    corners_3d = _build_corners(bounds)
    corners_2d = project_points_p2(corners_3d, p2_matrix)

    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    center_xyz = (
        0.5 * (x_min + x_max),
        0.5 * (y_min + y_max),
        0.5 * (z_min + z_max),
    )
    dims = (x_max - x_min, y_max - y_min, z_max - z_min)
    center_uv = (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    return ObjectBox3D(
        class_id=detection.class_id,
        label=detection.label,
        confidence=detection.confidence,
        bbox_xyxy=clipped,
        center_uv=(float(center_uv[0]), float(center_uv[1])),
        center_xyz=(float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])),
        dimensions_xyz=(float(dims[0]), float(dims[1]), float(dims[2])),
        bounds_xyz=bounds,
        corners_3d=corners_3d,
        corners_2d=corners_2d,
        num_points=int(filtered_points.shape[0]),
        depth_seed_m=float(seed_depth),
    )


def estimate_3d_boxes_from_detections(
    detections: Sequence[Detection2D],
    depth_map_m: np.ndarray,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
    p2_matrix: np.ndarray,
    min_depth_m: float = 0.1,
    max_depth_m: float = 120.0,
    inner_ratio: float = 0.65,
    min_points: int = 50,
) -> List[ObjectBox3D]:
    boxes: List[ObjectBox3D] = []
    for det in detections:
        box3d = estimate_3d_box_from_detection(
            detection=det,
            depth_map_m=depth_map_m,
            fx_px=fx_px,
            fy_px=fy_px,
            cx_px=cx_px,
            cy_px=cy_px,
            p2_matrix=p2_matrix,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            inner_ratio=inner_ratio,
            min_points=min_points,
        )
        if box3d is not None:
            boxes.append(box3d)
    return boxes


def draw_3d_boxes(image_bgr: np.ndarray, boxes_3d: Sequence[ObjectBox3D], cv2_module) -> np.ndarray:
    output = image_bgr.copy()
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for box in boxes_3d:
        pts = box.corners_2d
        for start, end in edges:
            p1 = pts[start]
            p2 = pts[end]
            if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
                continue
            x1, y1 = int(round(float(p1[0]))), int(round(float(p1[1])))
            x2, y2 = int(round(float(p2[0]))), int(round(float(p2[1])))
            cv2_module.line(output, (x1, y1), (x2, y2), (0, 128, 255), 2, cv2_module.LINE_AA)

        finite_y = pts[np.isfinite(pts).all(axis=1)]
        if finite_y.size > 0:
            anchor_idx = int(np.argmin(finite_y[:, 1]))
            text_x = int(round(float(finite_y[anchor_idx, 0])))
            text_y = int(round(float(finite_y[anchor_idx, 1]))) - 8
        else:
            x1, y1, _, _ = box.bbox_xyxy
            text_x, text_y = x1, y1 - 8

        cx, cy, cz = box.center_xyz
        w, h, l = box.dimensions_xyz
        text = f"{box.label} {box.confidence:.2f} | Z={cz:.1f}m | ({w:.1f},{h:.1f},{l:.1f})m"
        cv2_module.putText(
            output,
            text,
            (text_x, max(15, text_y)),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2_module.LINE_AA,
        )
    return output
