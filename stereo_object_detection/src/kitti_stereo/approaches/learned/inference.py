from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import numpy as np
import torch

from ...detection import Detection2D
from ...geometry_3d import box3d_corners_camera, draw_projected_box3d, project_points_p2, valid_finite_points
from .dataset import build_geom_features, build_roi_tensor
from .models import decode_rgbd_transformer_output


@dataclass(frozen=True)
class DLBox3DResult:
    class_id: int
    label: str
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]
    center_xyz: Tuple[float, float, float]
    dimensions: Tuple[float, float, float]  # h,w,l
    yaw_ry: float
    corners_3d: np.ndarray
    corners_2d: np.ndarray


def predict_dl_3d_boxes(
    detections: Sequence[Detection2D],
    left_bgr: np.ndarray,
    depth_map_m: np.ndarray,
    p2_matrix: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    cv2_module: Any,
    roi_size: int = 96,
    max_depth_m: float = 80.0,
) -> List[DLBox3DResult]:
    if not detections:
        return []

    rois = []
    geoms = []
    for det in detections:
        rois.append(
            build_roi_tensor(
                image_bgr=left_bgr,
                depth_map_m=depth_map_m,
                bbox_xyxy=det.bbox_xyxy,
                cv2_module=cv2_module,
                roi_size=roi_size,
                max_depth_m=max_depth_m,
            )
        )
        geoms.append(
            build_geom_features(
                bbox_xyxy=det.bbox_xyxy,
                image_width=left_bgr.shape[1],
                image_height=left_bgr.shape[0],
            )
        )

    roi_tensor = torch.from_numpy(np.stack(rois, axis=0)).to(device=device, dtype=torch.float32)
    geom_tensor = torch.from_numpy(np.stack(geoms, axis=0)).to(device=device, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        pred, _ = model(roi_tensor, geom_tensor)

    results: List[DLBox3DResult] = []
    pred_np = pred.detach().cpu()
    for det, row in zip(detections, pred_np):
        decoded = decode_rgbd_transformer_output(row)
        corners_3d = box3d_corners_camera(
            x_m=decoded["x_m"],
            y_m=decoded["y_m"],
            z_m=decoded["z_m"],
            height_m=decoded["h_m"],
            width_m=decoded["w_m"],
            length_m=decoded["l_m"],
            yaw_ry=decoded["ry"],
        )
        corners_2d = project_points_p2(corners_3d, p2_matrix)
        results.append(
            DLBox3DResult(
                class_id=det.class_id,
                label=det.label,
                confidence=det.confidence,
                bbox_xyxy=det.bbox_xyxy,
                center_xyz=(decoded["x_m"], decoded["y_m"], decoded["z_m"]),
                dimensions=(decoded["h_m"], decoded["w_m"], decoded["l_m"]),
                yaw_ry=decoded["ry"],
                corners_3d=corners_3d,
                corners_2d=corners_2d,
            )
        )
    return results


def draw_dl_3d_boxes(image_bgr: np.ndarray, boxes: Sequence[DLBox3DResult], cv2_module: Any) -> np.ndarray:
    output = image_bgr.copy()
    for box in boxes:
        output = draw_projected_box3d(output, box.corners_2d, cv2_module, color=(0, 255, 0), thickness=2)
        finite = valid_finite_points(box.corners_2d)
        if finite.shape[0] > 0:
            idx = int(np.argmin(finite[:, 1]))
            tx, ty = int(round(float(finite[idx, 0]))), int(round(float(finite[idx, 1]))) - 10
        else:
            tx, ty = box.bbox_xyxy[0], box.bbox_xyxy[1] - 10

        x_m, y_m, z_m = box.center_xyz
        h_m, w_m, l_m = box.dimensions
        text = f"{box.label} {box.confidence:.2f} Z={z_m:.1f}m dims=({h_m:.1f},{w_m:.1f},{l_m:.1f})"
        cv2_module.putText(
            output,
            text,
            (tx, max(15, ty)),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2_module.LINE_AA,
        )
    return output
