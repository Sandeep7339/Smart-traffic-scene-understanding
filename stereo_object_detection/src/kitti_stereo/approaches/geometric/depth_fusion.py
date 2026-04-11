from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ...detection import Detection2D


@dataclass(frozen=True)
class FusedObject3D:
    label: str
    class_id: int
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]
    center_uv: Tuple[float, float]
    depth_m: float
    xyz_m: Tuple[float, float, float]
    valid_depth_pixels: int


def _clip_bbox_to_depth_map(
    bbox_xyxy: Tuple[int, int, int, int],
    width: int,
    height: int,
) -> Tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def robust_depth_from_bbox(
    depth_map_m: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    min_depth_m: float = 0.1,
    max_depth_m: float = 120.0,
) -> Tuple[float, int]:
    h, w = depth_map_m.shape[:2]
    clipped = _clip_bbox_to_depth_map(bbox_xyxy, width=w, height=h)
    if clipped is None:
        return (float("nan"), 0)

    x1, y1, x2, y2 = clipped
    patch = depth_map_m[y1 : y2 + 1, x1 : x2 + 1]
    valid_mask = np.isfinite(patch) & (patch > min_depth_m) & (patch < max_depth_m)
    if not np.any(valid_mask):
        return (float("nan"), 0)

    valid_values = patch[valid_mask].astype(np.float32)
    initial_median = float(np.median(valid_values))

    # Robustify against depth outliers inside a detection region.
    absolute_deviation = np.abs(valid_values - initial_median)
    mad = float(np.median(absolute_deviation))
    if mad > 1e-6:
        robust_sigma = 1.4826 * mad
        inlier_mask = absolute_deviation <= (3.0 * robust_sigma)
        filtered = valid_values[inlier_mask]
        if filtered.size > 0:
            valid_values = filtered

    return (float(np.median(valid_values)), int(valid_values.size))


def project_pixel_to_camera(
    u: float,
    v: float,
    z_m: float,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
) -> Tuple[float, float, float]:
    if not np.isfinite(z_m) or z_m <= 0.0:
        return (float("nan"), float("nan"), float("nan"))

    x_m = (u - cx_px) * z_m / fx_px
    y_m = (v - cy_px) * z_m / fy_px
    return (float(x_m), float(y_m), float(z_m))


def fuse_detections_with_depth(
    detections: List[Detection2D],
    depth_map_m: np.ndarray,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
    min_depth_m: float = 0.1,
    max_depth_m: float = 120.0,
) -> List[FusedObject3D]:
    fused: List[FusedObject3D] = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        depth_m, valid_px = robust_depth_from_bbox(
            depth_map_m=depth_map_m,
            bbox_xyxy=det.bbox_xyxy,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
        )

        u = 0.5 * (x1 + x2)
        v = 0.5 * (y1 + y2)
        xyz = project_pixel_to_camera(
            u=u,
            v=v,
            z_m=depth_m,
            fx_px=fx_px,
            fy_px=fy_px,
            cx_px=cx_px,
            cy_px=cy_px,
        )

        fused.append(
            FusedObject3D(
                label=det.label,
                class_id=det.class_id,
                confidence=det.confidence,
                bbox_xyxy=det.bbox_xyxy,
                center_uv=(float(u), float(v)),
                depth_m=depth_m,
                xyz_m=xyz,
                valid_depth_pixels=valid_px,
            )
        )
    return fused


def draw_fused_objects(
    image_bgr: np.ndarray,
    fused_objects: List[FusedObject3D],
    cv2_module,
    show_xyz: bool = True,
) -> np.ndarray:
    output = image_bgr.copy()
    for obj in fused_objects:
        x1, y1, x2, y2 = obj.bbox_xyxy
        cv2_module.rectangle(output, (x1, y1), (x2, y2), (0, 220, 0), 2)

        if np.isfinite(obj.depth_m):
            text = f"{obj.label} {obj.confidence:.2f} | Z={obj.depth_m:.1f}m"
        else:
            text = f"{obj.label} {obj.confidence:.2f} | Z=n/a"

        text_x = x1
        text_y = y1 - 10 if y1 > 20 else y1 + 18
        cv2_module.putText(
            output,
            text,
            (text_x, text_y),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
            cv2_module.LINE_AA,
        )

        if show_xyz:
            x_m, y_m, z_m = obj.xyz_m
            if np.isfinite(x_m) and np.isfinite(y_m) and np.isfinite(z_m):
                xyz_text = f"X={x_m:.1f} Y={y_m:.1f} Z={z_m:.1f}"
            else:
                xyz_text = "X=n/a Y=n/a Z=n/a"
            cv2_module.putText(
                output,
                xyz_text,
                (x1, min(y2 + 18, output.shape[0] - 5)),
                cv2_module.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 240, 0),
                1,
                cv2_module.LINE_AA,
            )
    return output
