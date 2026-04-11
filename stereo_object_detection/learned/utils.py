import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kitti_stereo.geometry_3d import bbox_iou_xyxy
from kitti_stereo.fusion import project_pixel_to_camera, robust_depth_from_bbox


def clip_bbox(bbox_xyxy, width, height):
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def crop_rgb_depth_patch(image_bgr, depth_map_m, bbox_xyxy, cv2_module, patch_size=96, max_depth_m=80.0):
    h, w = image_bgr.shape[:2]
    clipped = clip_bbox(bbox_xyxy, width=w, height=h)
    if clipped is None:
        return np.zeros((4, patch_size, patch_size), dtype=np.float32)

    x1, y1, x2, y2 = clipped
    rgb = image_bgr[y1 : y2 + 1, x1 : x2 + 1]
    depth = depth_map_m[y1 : y2 + 1, x1 : x2 + 1].astype(np.float32)

    rgb = cv2_module.cvtColor(rgb, cv2_module.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)

    valid = np.isfinite(depth) & (depth > 0.0)
    depth[~valid] = 0.0
    depth = np.clip(depth, 0.0, float(max_depth_m)) / float(max_depth_m)

    rgb = cv2_module.resize(rgb, (patch_size, patch_size), interpolation=cv2_module.INTER_LINEAR)
    depth = cv2_module.resize(depth, (patch_size, patch_size), interpolation=cv2_module.INTER_NEAREST)

    stacked = np.concatenate([rgb, depth[..., None]], axis=2)
    return np.transpose(stacked, (2, 0, 1)).astype(np.float32)


def build_bbox_features(bbox_xyxy, image_width, image_height):
    x1, y1, x2, y2 = bbox_xyxy
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    return np.asarray(
        [
            cx / float(image_width),
            cy / float(image_height),
            bw / float(image_width),
            bh / float(image_height),
            bw / bh,
            (bw * bh) / float(image_width * image_height),
            y1 / float(image_height),
            y2 / float(image_height),
        ],
        dtype=np.float32,
    )


def compute_detection_anchor(bbox_xyxy, depth_map_m, fx_px, fy_px, cx_px, cy_px):
    x1, y1, x2, y2 = bbox_xyxy
    u = 0.5 * (x1 + x2)
    v = 0.5 * (y1 + y2)
    depth_m, _ = robust_depth_from_bbox(depth_map_m, bbox_xyxy=bbox_xyxy, min_depth_m=0.1, max_depth_m=120.0)
    ax, ay, az = project_pixel_to_camera(
        u=u,
        v=v,
        z_m=depth_m,
        fx_px=fx_px,
        fy_px=fy_px,
        cx_px=cx_px,
        cy_px=cy_px,
    )
    if not np.isfinite(ax) or not np.isfinite(ay) or not np.isfinite(az):
        ax, ay, az = 0.0, 0.0, 10.0
    return np.asarray([ax, ay, max(0.1, az)], dtype=np.float32)


def encode_target_from_label(label_obj, anchor_xyz):
    # Target order: (dx, dy, dz, log(w), log(h), log(l), sin(theta), cos(theta))
    ax, ay, az = float(anchor_xyz[0]), float(anchor_xyz[1]), float(anchor_xyz[2])
    tx = float(label_obj.x_m) - ax
    ty = float(label_obj.y_m) - ay
    tz = float(label_obj.z_m) - az
    theta = float(label_obj.rotation_y)
    return np.asarray(
        [
            tx,
            ty,
            tz,
            math.log(max(1e-3, float(label_obj.width_m))),
            math.log(max(1e-3, float(label_obj.height_m))),
            math.log(max(1e-3, float(label_obj.length_m))),
            math.sin(theta),
            math.cos(theta),
        ],
        dtype=np.float32,
    )


def angular_l1_loss(pred_theta, target_theta):
    diff = pred_theta - target_theta
    wrapped = np.arctan2(np.sin(diff), np.cos(diff))
    return np.abs(wrapped)


def best_iou_match(det_bbox, gt_objects, used_gt, label_compat_fn=None, det_label=None):
    best_iou = 0.0
    best_idx = -1
    for i, gt in enumerate(gt_objects):
        if i in used_gt:
            continue
        if label_compat_fn is not None and det_label is not None:
            if not label_compat_fn(det_label, gt.object_type):
                continue
        iou = bbox_iou_xyxy(det_bbox, gt.bbox_xyxy)
        if iou > best_iou:
            best_iou = iou
            best_idx = i
    return best_idx, best_iou


def decode_prediction_with_anchor(pred_vec, anchor_xyz):
    dx, dy, dz, log_w, log_h, log_l, sin_t, cos_t = [float(v) for v in pred_vec]
    ax, ay, az = [float(v) for v in anchor_xyz]
    theta = math.atan2(sin_t, cos_t)
    return {
        "x": ax + dx,
        "y": ay + dy,
        "z": max(0.1, az + dz),
        "w": max(0.1, math.exp(log_w)),
        "h": max(0.1, math.exp(log_h)),
        "l": max(0.1, math.exp(log_l)),
        "theta": math.atan2(math.sin(theta), math.cos(theta)),
    }
