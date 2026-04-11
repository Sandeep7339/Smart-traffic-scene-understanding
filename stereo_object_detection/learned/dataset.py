from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geometric.depth import compute_depth_frame_kitti
from geometric.yolo_detect import detect_2d_boxes
from kitti_stereo.class_mapping import is_detection_label_compatible
from kitti_stereo.kitti_data import get_kitti_sample_paths
from kitti_stereo.kitti_labels import default_training_types, load_kitti_labels

from learned.utils import (
    best_iou_match,
    build_bbox_features,
    compute_detection_anchor,
    crop_rgb_depth_patch,
    encode_target_from_label,
)


@dataclass(frozen=True)
class TrainingEntry:
    sample_id: str
    bbox_xyxy: tuple
    anchor_xyz: np.ndarray
    target: np.ndarray


class StereoYoloCropDataset(Dataset):
    """Creates training crops from YOLO boxes matched to KITTI GT objects."""

    def __init__(
        self,
        dataset_root,
        sample_ids,
        cv2_module,
        patch_size=96,
        max_depth_m=80.0,
        stereo_method="sgbm",
        num_disparities=128,
        block_size=7,
        yolo_model=str(ROOT / "yolov8n.pt"),
        conf_threshold=0.25,
        iou_threshold=0.45,
        match_iou=0.3,
        include_types=None,
        max_frames=None,
    ):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.cv2 = cv2_module
        self.patch_size = int(patch_size)
        self.max_depth_m = float(max_depth_m)
        self.stereo_method = stereo_method
        self.num_disparities = int(num_disparities)
        self.block_size = int(block_size)
        self.yolo_model = yolo_model
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.match_iou = float(match_iou)
        self.include_types = tuple(include_types) if include_types is not None else default_training_types()

        if max_frames is not None and max_frames > 0:
            sample_ids = list(sample_ids)[: int(max_frames)]
        self.sample_ids = list(sample_ids)

        self.entries = []
        self._frame_cache = {}
        self._build_entries()

    def _load_frame(self, sample_id):
        if sample_id in self._frame_cache:
            return self._frame_cache[sample_id]

        frame = compute_depth_frame_kitti(
            dataset_root=self.dataset_root,
            split="training",
            sample_id=sample_id,
            cv2_module=self.cv2,
            stereo_method=self.stereo_method,
            num_disparities=self.num_disparities,
            block_size=self.block_size,
            max_depth_m=self.max_depth_m,
        )

        if len(self._frame_cache) >= 3:
            first_key = next(iter(self._frame_cache.keys()))
            self._frame_cache.pop(first_key)
        self._frame_cache[sample_id] = frame
        return frame

    def _build_entries(self):
        for sid in self.sample_ids:
            try:
                sample = get_kitti_sample_paths(self.dataset_root, "training", sid)
            except FileNotFoundError:
                continue

            gt_objects = load_kitti_labels(sample.label_file, include_types=self.include_types, ignore_dontcare=True)
            if not gt_objects:
                continue

            frame = self._load_frame(sample.sample_id)
            detections = detect_2d_boxes(
                image_bgr=frame.left_bgr,
                model_name=self.yolo_model,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                max_detections=100,
                imgsz=640,
                device=None,
            )

            used_gt = set()
            for det in detections:
                gt_idx, best_iou = best_iou_match(
                    det_bbox=det.bbox_xyxy,
                    gt_objects=gt_objects,
                    used_gt=used_gt,
                    label_compat_fn=is_detection_label_compatible,
                    det_label=det.label,
                )
                if gt_idx < 0 or best_iou < self.match_iou:
                    continue

                gt = gt_objects[gt_idx]
                used_gt.add(gt_idx)
                anchor_xyz = compute_detection_anchor(
                    bbox_xyxy=det.bbox_xyxy,
                    depth_map_m=frame.depth_m,
                    fx_px=frame.geometry.fx_px,
                    fy_px=frame.geometry.fy_px,
                    cx_px=frame.geometry.cx_px,
                    cy_px=frame.geometry.cy_px,
                )
                self.entries.append(
                    TrainingEntry(
                        sample_id=sample.sample_id,
                        bbox_xyxy=det.bbox_xyxy,
                        anchor_xyz=anchor_xyz,
                        target=encode_target_from_label(gt, anchor_xyz=anchor_xyz),
                    )
                )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        frame = self._load_frame(entry.sample_id)

        patch = crop_rgb_depth_patch(
            image_bgr=frame.left_bgr,
            depth_map_m=frame.depth_m,
            bbox_xyxy=entry.bbox_xyxy,
            cv2_module=self.cv2,
            patch_size=self.patch_size,
            max_depth_m=self.max_depth_m,
        )
        geom = build_bbox_features(
            bbox_xyxy=entry.bbox_xyxy,
            image_width=frame.left_bgr.shape[1],
            image_height=frame.left_bgr.shape[0],
        )

        return {
            "patch": torch.from_numpy(patch),
            "geom": torch.from_numpy(geom.astype(np.float32)),
            "anchor_xyz": torch.from_numpy(entry.anchor_xyz.astype(np.float32)),
            "target": torch.from_numpy(entry.target.astype(np.float32)),
            "sample_id": entry.sample_id,
            "bbox": torch.tensor(entry.bbox_xyxy, dtype=torch.int64),
        }
