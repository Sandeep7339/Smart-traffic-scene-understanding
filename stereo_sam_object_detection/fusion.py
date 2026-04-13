
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from utils import apply_mask_overlay, ensure_legacy_src_on_path, mask_bbox_xyxy

ensure_legacy_src_on_path()

from kitti_stereo import bbox_iou_xyxy, project_points_p2  # noqa: E402
from kitti_stereo.class_mapping import is_detection_label_compatible  # noqa: E402


@dataclass(frozen=True)
class FusedPrediction:
    det_idx: int
    class_id: int
    label: str
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]
    region_bbox_xyxy: Optional[Tuple[int, int, int, int]]
    depth_m: float
    depth_var_m2: float
    num_depth_points: int
    center_xyz: Tuple[float, float, float]
    dimensions_xyz: Tuple[float, float, float]
    corners_3d: np.ndarray
    corners_2d: np.ndarray
    projected_bbox_xyxy: Optional[Tuple[int, int, int, int]]
    mask: Optional[np.ndarray]


@dataclass(frozen=True)
class MatchRecord:
    abs_depth_err_m: float
    sq_depth_err_m2: float
    depth_var_m2: float
    proj_iou: float
    region_iou: float


def _clip_bbox(
    bbox_xyxy: Tuple[int, int, int, int],
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _bbox_mask(shape_hw: Tuple[int, int], bbox_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=bool)
    clipped = _clip_bbox(bbox_xyxy, width=w, height=h)
    if clipped is None:
        return mask
    x1, y1, x2, y2 = clipped
    mask[y1 : y2 + 1, x1 : x2 + 1] = True
    return mask


def _robust_depth_mask(
    depth_map_m: np.ndarray,
    region_mask: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    lower_percentile: float,
    upper_percentile: float,
) -> np.ndarray:
    valid = region_mask & np.isfinite(depth_map_m) & (depth_map_m > min_depth_m) & (depth_map_m < max_depth_m)
    if not np.any(valid):
        return valid

    values = depth_map_m[valid].astype(np.float32)
    lo, hi = np.percentile(values, [lower_percentile, upper_percentile])
    inlier = valid & (depth_map_m >= lo) & (depth_map_m <= hi)

    values = depth_map_m[inlier].astype(np.float32)
    if values.size == 0:
        return inlier

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 1e-6:
        return inlier

    robust_sigma = 1.4826 * mad
    depth_band = max(0.15, 3.0 * robust_sigma)
    return inlier & (np.abs(depth_map_m - median) <= depth_band)


def _depth_mask_to_points(
    depth_map_m: np.ndarray,
    depth_mask: np.ndarray,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
) -> np.ndarray:
    ys, xs = np.where(depth_mask)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    zs = depth_map_m[ys, xs].astype(np.float32)
    xs_m = (xs.astype(np.float32) - cx_px) * zs / fx_px
    ys_m = (ys.astype(np.float32) - cy_px) * zs / fy_px
    return np.stack([xs_m, ys_m, zs], axis=1)


def _fit_bounds(points_xyz: np.ndarray) -> Optional[Tuple[float, float, float, float, float, float]]:
    if points_xyz.shape[0] < 4:
        return None

    lo = np.percentile(points_xyz, 2.0, axis=0)
    hi = np.percentile(points_xyz, 98.0, axis=0)
    inlier = np.all((points_xyz >= lo) & (points_xyz <= hi), axis=1)
    filtered = points_xyz[inlier] if np.any(inlier) else points_xyz
    if filtered.shape[0] < 4:
        return None

    mins = filtered.min(axis=0)
    maxs = filtered.max(axis=0)
    eps = 1e-3
    maxs = np.maximum(maxs, mins + eps)
    return (float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]), float(mins[2]), float(maxs[2]))


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


def _projected_bbox(
    corners_2d: np.ndarray,
    shape_hw: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    finite = corners_2d[np.isfinite(corners_2d).all(axis=1)]
    if finite.shape[0] == 0:
        return None

    h, w = shape_hw
    x1 = int(np.clip(np.floor(finite[:, 0].min()), 0, w - 1))
    y1 = int(np.clip(np.floor(finite[:, 1].min()), 0, h - 1))
    x2 = int(np.clip(np.ceil(finite[:, 0].max()), 0, w - 1))
    y2 = int(np.clip(np.ceil(finite[:, 1].max()), 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _estimate_prediction(
    det_idx: int,
    det,
    depth_map_m: np.ndarray,
    region_mask: np.ndarray,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
    p2_matrix: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    lower_percentile: float,
    upper_percentile: float,
    min_points: int,
    source_mask: Optional[np.ndarray],
) -> FusedPrediction:
    h, w = depth_map_m.shape[:2]
    bbox_mask = _bbox_mask((h, w), det.bbox_xyxy)
    if np.any(region_mask):
        region_mask = region_mask & bbox_mask
    else:
        region_mask = bbox_mask

    region_bbox = mask_bbox_xyxy(region_mask)

    robust_mask = _robust_depth_mask(
        depth_map_m=depth_map_m,
        region_mask=region_mask,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    values = depth_map_m[robust_mask].astype(np.float32)

    if values.size == 0:
        depth_m = float("nan")
        depth_var = float("nan")
        num_depth_points = 0
    else:
        depth_m = float(np.median(values))
        depth_var = float(np.var(values))
        num_depth_points = int(values.size)

    points_xyz = _depth_mask_to_points(
        depth_map_m=depth_map_m,
        depth_mask=robust_mask,
        fx_px=fx_px,
        fy_px=fy_px,
        cx_px=cx_px,
        cy_px=cy_px,
    )

    corners_3d = np.full((8, 3), np.nan, dtype=np.float32)
    corners_2d = np.full((8, 2), np.nan, dtype=np.float32)
    projected_bbox = None
    dims = (float("nan"), float("nan"), float("nan"))

    if points_xyz.shape[0] >= min_points:
        bounds = _fit_bounds(points_xyz)
        if bounds is not None:
            corners_3d = _build_corners(bounds)
            corners_2d = project_points_p2(corners_3d, p2_matrix.astype(np.float32))
            projected_bbox = _projected_bbox(corners_2d, (h, w))
            x_min, x_max, y_min, y_max, z_min, z_max = bounds
            dims = (x_max - x_min, y_max - y_min, z_max - z_min)
            center_xyz = (
                0.5 * (x_min + x_max),
                0.5 * (y_min + y_max),
                0.5 * (z_min + z_max),
            )
        else:
            center_xyz = _fallback_center(det.bbox_xyxy, depth_m, fx_px, fy_px, cx_px, cy_px)
    else:
        center_xyz = _fallback_center(det.bbox_xyxy, depth_m, fx_px, fy_px, cx_px, cy_px)

    return FusedPrediction(
        det_idx=det_idx,
        class_id=det.class_id,
        label=det.label,
        confidence=det.confidence,
        bbox_xyxy=det.bbox_xyxy,
        region_bbox_xyxy=region_bbox,
        depth_m=depth_m,
        depth_var_m2=depth_var,
        num_depth_points=num_depth_points,
        center_xyz=(float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])),
        dimensions_xyz=(float(dims[0]), float(dims[1]), float(dims[2])),
        corners_3d=corners_3d,
        corners_2d=corners_2d,
        projected_bbox_xyxy=projected_bbox,
        mask=source_mask,
    )


def _fallback_center(
    bbox_xyxy: Tuple[int, int, int, int],
    depth_m: float,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
) -> Tuple[float, float, float]:
    if not np.isfinite(depth_m) or depth_m <= 0.0:
        return (float("nan"), float("nan"), float("nan"))
    x1, y1, x2, y2 = bbox_xyxy
    u = 0.5 * (x1 + x2)
    v = 0.5 * (y1 + y2)
    x_m = (u - cx_px) * depth_m / fx_px
    y_m = (v - cy_px) * depth_m / fy_px
    return (float(x_m), float(y_m), float(depth_m))


def fuse_detections(
    detections: Sequence,
    depth_map_m: np.ndarray,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
    p2_matrix: np.ndarray,
    masks: Optional[Sequence[np.ndarray]] = None,
    min_depth_m: float = 0.1,
    max_depth_m: float = 80.0,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
    min_points: int = 40,
) -> list[FusedPrediction]:
    h, w = depth_map_m.shape[:2]
    predictions: list[FusedPrediction] = []

    for idx, det in enumerate(detections):
        if masks is not None and idx < len(masks) and masks[idx] is not None:
            mask = masks[idx].astype(bool)
            if mask.shape != (h, w):
                safe_mask = np.zeros((h, w), dtype=bool)
                mh = min(h, mask.shape[0])
                mw = min(w, mask.shape[1])
                safe_mask[:mh, :mw] = mask[:mh, :mw]
                mask = safe_mask
        else:
            mask = _bbox_mask((h, w), det.bbox_xyxy)

        pred = _estimate_prediction(
            det_idx=idx,
            det=det,
            depth_map_m=depth_map_m,
            region_mask=mask,
            fx_px=fx_px,
            fy_px=fy_px,
            cx_px=cx_px,
            cy_px=cy_px,
            p2_matrix=p2_matrix,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            min_points=min_points,
            source_mask=mask if masks is not None else None,
        )
        predictions.append(pred)

    return predictions


def draw_fused_predictions(
    image_bgr: np.ndarray,
    predictions: Sequence[FusedPrediction],
    cv2_module,
    show_masks: bool = False,
    draw_detection_bbox: bool = True,
) -> np.ndarray:
    out = image_bgr.copy()
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    for pred in predictions:
        if show_masks and pred.mask is not None and np.any(pred.mask):
            out = apply_mask_overlay(out, pred.mask, color_bgr=(255, 120, 0), alpha=0.32)

        x1, y1, x2, y2 = pred.bbox_xyxy
        if draw_detection_bbox:
            cv2_module.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)

        if pred.region_bbox_xyxy is not None and show_masks:
            rx1, ry1, rx2, ry2 = pred.region_bbox_xyxy
            cv2_module.rectangle(out, (rx1, ry1), (rx2, ry2), (255, 120, 0), 1)

        for a, b in edges:
            p1 = pred.corners_2d[a]
            p2 = pred.corners_2d[b]
            if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
                continue
            c1 = (int(round(float(p1[0]))), int(round(float(p1[1]))))
            c2 = (int(round(float(p2[0]))), int(round(float(p2[1]))))
            cv2_module.line(out, c1, c2, (0, 128, 255), 2, cv2_module.LINE_AA)

        depth_txt = f"{pred.label} {pred.confidence:.2f} | Z={pred.depth_m:.2f}m"
        var_txt = f"var={pred.depth_var_m2:.3f} | n={pred.num_depth_points}"
        text_y = max(15, y1 - 10)
        cv2_module.putText(
            out,
            depth_txt,
            (x1, text_y),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2_module.LINE_AA,
        )
        cv2_module.putText(
            out,
            var_txt,
            (x1, min(out.shape[0] - 6, y2 + 16)),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 240, 0),
            1,
            cv2_module.LINE_AA,
        )

    return out


def match_predictions_to_gt(
    predictions: Sequence[FusedPrediction],
    gt_objects: Sequence,
    iou_threshold: float = 0.3,
) -> list[MatchRecord]:
    matched: list[MatchRecord] = []
    used_gt: set[int] = set()

    for pred in predictions:
        best_iou = 0.0
        best_idx = -1
        for gt_idx, gt in enumerate(gt_objects):
            if gt_idx in used_gt:
                continue
            if not is_detection_label_compatible(pred.label, gt.object_type):
                continue

            iou = bbox_iou_xyxy(pred.bbox_xyxy, gt.bbox_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_idx = gt_idx

        if best_idx < 0 or best_iou < iou_threshold or not np.isfinite(pred.depth_m):
            continue

        used_gt.add(best_idx)
        gt = gt_objects[best_idx]
        err = float(pred.depth_m - gt.z_m)

        proj_iou = float("nan")
        if pred.projected_bbox_xyxy is not None:
            proj_iou = bbox_iou_xyxy(pred.projected_bbox_xyxy, gt.bbox_xyxy)

        region_iou = float("nan")
        if pred.region_bbox_xyxy is not None:
            region_iou = bbox_iou_xyxy(pred.region_bbox_xyxy, gt.bbox_xyxy)

        matched.append(
            MatchRecord(
                abs_depth_err_m=abs(err),
                sq_depth_err_m2=err * err,
                depth_var_m2=pred.depth_var_m2,
                proj_iou=proj_iou,
                region_iou=region_iou,
            )
        )

    return matched


def aggregate_matches(matches: Sequence[MatchRecord]) -> dict[str, float]:
    if len(matches) == 0:
        return {
            "count": 0.0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "mean_var": float("nan"),
            "proj_iou": float("nan"),
            "region_iou": float("nan"),
        }

    abs_err = np.asarray([m.abs_depth_err_m for m in matches], dtype=np.float32)
    sq_err = np.asarray([m.sq_depth_err_m2 for m in matches], dtype=np.float32)
    depth_var = np.asarray([m.depth_var_m2 for m in matches], dtype=np.float32)
    proj_iou = np.asarray([m.proj_iou for m in matches], dtype=np.float32)
    region_iou = np.asarray([m.region_iou for m in matches], dtype=np.float32)

    finite_proj = proj_iou[np.isfinite(proj_iou)]
    finite_region = region_iou[np.isfinite(region_iou)]

    return {
        "count": float(len(matches)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(sq_err))),
        "mean_var": float(np.mean(depth_var[np.isfinite(depth_var)])) if np.any(np.isfinite(depth_var)) else float("nan"),
        "proj_iou": float(np.mean(finite_proj)) if finite_proj.size > 0 else float("nan"),
        "region_iou": float(np.mean(finite_region)) if finite_region.size > 0 else float("nan"),
    }
