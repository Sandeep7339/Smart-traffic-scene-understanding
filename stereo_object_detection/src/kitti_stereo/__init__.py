"""Utilities for loading KITTI stereo data and estimating depth."""

from .kitti_data import (
    KittiSamplePaths,
    StereoGeometry,
    get_kitti_sample_paths,
    load_calibration,
    load_stereo_pair,
    stereo_geometry_from_calibration,
)
from .kitti_labels import (
    KittiLabelObject,
    default_training_types,
    load_kitti_labels,
    parse_kitti_label_line,
)
from .detection import Detection2D, run_yolo_detection
from .depth_pipeline import DepthFrame, compute_depth_frame
from .dl_dataset import KittiRoi3DDataset
from .dl_fusion import DLBox3DResult, draw_dl_3d_boxes, predict_dl_3d_boxes
from .approaches import geometric, learned
from .geometry_3d import box3d_corners_camera, bbox_iou_xyxy, draw_projected_box3d
from .models import DepthAware3DHead, RGBDTransformer3DHead
from .box3d import (
    ObjectBox3D,
    draw_3d_boxes,
    estimate_3d_box_from_detection,
    estimate_3d_boxes_from_detections,
    project_points_p2,
)
from .fusion import (
    FusedObject3D,
    draw_fused_objects,
    fuse_detections_with_depth,
    project_pixel_to_camera,
    robust_depth_from_bbox,
)
from .stereo_depth import (
    compute_disparity_bm,
    compute_disparity_sgbm,
    disparity_to_depth,
    normalize_for_display,
)

__all__ = [
    "KittiSamplePaths",
    "StereoGeometry",
    "get_kitti_sample_paths",
    "load_calibration",
    "load_stereo_pair",
    "stereo_geometry_from_calibration",
    "KittiLabelObject",
    "default_training_types",
    "load_kitti_labels",
    "parse_kitti_label_line",
    "Detection2D",
    "run_yolo_detection",
    "DepthFrame",
    "compute_depth_frame",
    "KittiRoi3DDataset",
    "DLBox3DResult",
    "predict_dl_3d_boxes",
    "draw_dl_3d_boxes",
    "geometric",
    "learned",
    "DepthAware3DHead",
    "RGBDTransformer3DHead",
    "box3d_corners_camera",
    "bbox_iou_xyxy",
    "draw_projected_box3d",
    "ObjectBox3D",
    "draw_3d_boxes",
    "estimate_3d_box_from_detection",
    "estimate_3d_boxes_from_detections",
    "project_points_p2",
    "FusedObject3D",
    "draw_fused_objects",
    "fuse_detections_with_depth",
    "project_pixel_to_camera",
    "robust_depth_from_bbox",
    "compute_disparity_bm",
    "compute_disparity_sgbm",
    "disparity_to_depth",
    "normalize_for_display",
]
