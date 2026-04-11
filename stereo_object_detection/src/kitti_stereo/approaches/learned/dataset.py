from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ...depth_pipeline import DepthFrame, compute_depth_frame
from ...kitti_data import get_kitti_sample_paths
from ...kitti_labels import KittiLabelObject, default_training_types, load_kitti_labels


@dataclass(frozen=True)
class RoiTrainingSample:
    sample_id: str
    bbox_xyxy: Tuple[int, int, int, int]
    label_obj: KittiLabelObject


def _clip_bbox(bbox_xyxy: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def build_roi_tensor(
    image_bgr: np.ndarray,
    depth_map_m: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    cv2_module: Any,
    roi_size: int = 96,
    max_depth_m: float = 80.0,
) -> np.ndarray:
    h, w = depth_map_m.shape[:2]
    clipped = _clip_bbox(bbox_xyxy, width=w, height=h)
    if clipped is None:
        return np.zeros((4, roi_size, roi_size), dtype=np.float32)

    x1, y1, x2, y2 = clipped
    rgb = image_bgr[y1 : y2 + 1, x1 : x2 + 1]
    depth = depth_map_m[y1 : y2 + 1, x1 : x2 + 1]

    rgb = cv2_module.cvtColor(rgb, cv2_module.COLOR_BGR2RGB).astype(np.float32) / 255.0
    depth = depth.astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    depth[~valid] = 0.0
    depth = np.clip(depth, 0.0, max_depth_m) / max_depth_m

    rgb_resized = cv2_module.resize(rgb, (roi_size, roi_size), interpolation=cv2_module.INTER_LINEAR)
    depth_resized = cv2_module.resize(depth, (roi_size, roi_size), interpolation=cv2_module.INTER_NEAREST)

    stacked = np.concatenate([rgb_resized, depth_resized[..., None]], axis=2)
    stacked = np.transpose(stacked, (2, 0, 1)).astype(np.float32)
    return stacked


def build_geom_features(
    bbox_xyxy: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    nx = cx / float(image_width)
    ny = cy / float(image_height)
    nw = bw / float(image_width)
    nh = bh / float(image_height)
    aspect = bw / bh
    area_ratio = (bw * bh) / float(image_width * image_height)
    top = y1 / float(image_height)
    bottom = y2 / float(image_height)

    return np.asarray([nx, ny, nw, nh, aspect, area_ratio, top, bottom], dtype=np.float32)


def build_target_vector(obj: KittiLabelObject) -> np.ndarray:
    return np.asarray(
        [
            obj.x_m,
            obj.y_m,
            obj.z_m,
            np.log(max(1e-3, obj.height_m)),
            np.log(max(1e-3, obj.width_m)),
            np.log(max(1e-3, obj.length_m)),
            np.sin(obj.rotation_y),
            np.cos(obj.rotation_y),
        ],
        dtype=np.float32,
    )


class KittiRoi3DDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        sample_ids: Sequence[str],
        cv2_module: Any,
        roi_size: int = 96,
        max_depth_m: float = 80.0,
        stereo_method: str = "sgbm",
        num_disparities: int = 128,
        block_size: int = 7,
        include_types: Sequence[str] | None = None,
        depth_cache_dir: str | Path | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.sample_ids = list(sample_ids)
        self.cv2 = cv2_module
        self.roi_size = int(roi_size)
        self.max_depth_m = float(max_depth_m)
        self.stereo_method = stereo_method
        self.num_disparities = int(num_disparities)
        self.block_size = int(block_size)
        self.include_types = tuple(include_types) if include_types is not None else default_training_types()

        self.depth_cache_dir = None if depth_cache_dir is None else Path(depth_cache_dir).expanduser().resolve()
        if self.depth_cache_dir is not None:
            self.depth_cache_dir.mkdir(parents=True, exist_ok=True)

        self.entries: List[RoiTrainingSample] = []
        for sid in self.sample_ids:
            sample = get_kitti_sample_paths(self.dataset_root, "training", sid)
            labels = load_kitti_labels(sample.label_file, include_types=self.include_types, ignore_dontcare=True)
            for obj in labels:
                x1, y1, x2, y2 = obj.bbox_xyxy
                if (x2 - x1) < 8 or (y2 - y1) < 8:
                    continue
                if obj.z_m <= 0.1:
                    continue
                self.entries.append(
                    RoiTrainingSample(
                        sample_id=sample.sample_id,
                        bbox_xyxy=obj.bbox_xyxy,
                        label_obj=obj,
                    )
                )

        self._frame_cache: Dict[str, DepthFrame] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def _depth_cache_path(self, sample_id: str) -> Path | None:
        if self.depth_cache_dir is None:
            return None
        return self.depth_cache_dir / f"{sample_id}_{self.stereo_method}_depth.npy"

    def _load_frame(self, sample_id: str) -> DepthFrame:
        if sample_id in self._frame_cache:
            return self._frame_cache[sample_id]

        cached_depth = None
        cache_path = self._depth_cache_path(sample_id)
        if cache_path is not None and cache_path.exists():
            cached_depth = np.load(cache_path)

        frame = compute_depth_frame(
            dataset_root=self.dataset_root,
            split="training",
            sample_id=sample_id,
            cv2_module=self.cv2,
            stereo_method=self.stereo_method,
            num_disparities=self.num_disparities,
            block_size=self.block_size,
            max_depth_m=self.max_depth_m,
        )
        if cached_depth is not None and cached_depth.shape == frame.depth_m.shape:
            frame = DepthFrame(
                sample=frame.sample,
                calibration=frame.calibration,
                geometry=frame.geometry,
                left_bgr=frame.left_bgr,
                right_bgr=frame.right_bgr,
                disparity=frame.disparity,
                depth_m=cached_depth.astype(np.float32),
            )
        elif cache_path is not None:
            np.save(cache_path, frame.depth_m.astype(np.float32))

        # tiny LRU behavior: keep at most 2 frames in memory
        if len(self._frame_cache) >= 2:
            first_key = next(iter(self._frame_cache.keys()))
            self._frame_cache.pop(first_key)
        self._frame_cache[sample_id] = frame
        return frame

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        entry = self.entries[idx]
        frame = self._load_frame(entry.sample_id)

        roi = build_roi_tensor(
            image_bgr=frame.left_bgr,
            depth_map_m=frame.depth_m,
            bbox_xyxy=entry.bbox_xyxy,
            cv2_module=self.cv2,
            roi_size=self.roi_size,
            max_depth_m=self.max_depth_m,
        )
        geom = build_geom_features(
            bbox_xyxy=entry.bbox_xyxy,
            image_width=frame.left_bgr.shape[1],
            image_height=frame.left_bgr.shape[0],
        )
        target = build_target_vector(entry.label_obj)

        return {
            "roi": torch.from_numpy(roi),
            "geom": torch.from_numpy(geom),
            "target": torch.from_numpy(target),
            "sample_id": entry.sample_id,
        }
