
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import KITTI_CLASS_TO_ID

# Reuse existing stereo/depth/label code without modifying that project.
_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_STEREO_SRC = _WORKSPACE_ROOT / "stereo_object_detection" / "src"
import sys

if str(_STEREO_SRC) not in sys.path:
    sys.path.insert(0, str(_STEREO_SRC))

from kitti_stereo.depth_pipeline import compute_depth_frame  # noqa: E402
from kitti_stereo.kitti_data import get_kitti_sample_paths  # noqa: E402
from kitti_stereo.kitti_labels import load_kitti_labels  # noqa: E402


@dataclass(frozen=True)
class SampleMeta:
    sample_id: str
    left_image_path: Path
    right_image_path: Path
    label_path: Path


def _to_xywh_norm(box_xyxy: tuple[int, int, int, int], img_w: int, img_h: int) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    cx = float(x1) + 0.5 * bw
    cy = float(y1) + 0.5 * bh
    return np.asarray([cx / img_w, cy / img_h, bw / img_w, bh / img_h], dtype=np.float32)


def _resize_boxes_xywh_norm(boxes_xywh_norm: np.ndarray) -> np.ndarray:
    # Boxes are already normalized, so resize does not change coordinates.
    return boxes_xywh_norm


class KittiRgbDepthEdgeDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "training",
        img_size: int = 384,
        max_depth_m: float = 80.0,
        stereo_method: str = "sgbm",
        num_disparities: int = 128,
        block_size: int = 7,
        sample_ids: list[str] | None = None,
        depth_cache_dir: str | Path | None = None,
        cv2_module: Any | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split = split
        self.img_size = int(img_size)
        self.max_depth_m = float(max_depth_m)
        self.stereo_method = stereo_method
        self.num_disparities = int(num_disparities)
        self.block_size = int(block_size)

        try:
            import cv2 as _cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for dataset loading.") from exc
        self.cv2 = cv2_module if cv2_module is not None else _cv2

        if depth_cache_dir is None:
            self.depth_cache_dir = None
        else:
            self.depth_cache_dir = Path(depth_cache_dir).expanduser().resolve()
            self.depth_cache_dir.mkdir(parents=True, exist_ok=True)

        if sample_ids is None:
            label_dir = self.dataset_root / "data_object_label_2" / "training" / "label_2"
            sample_ids = sorted(p.stem for p in label_dir.glob("*.txt"))

        self.samples: list[SampleMeta] = []
        for sid in sample_ids:
            paths = get_kitti_sample_paths(self.dataset_root, self.split, sid)
            if paths.label_file is None:
                continue
            self.samples.append(
                SampleMeta(
                    sample_id=paths.sample_id,
                    left_image_path=paths.left_image,
                    right_image_path=paths.right_image,
                    label_path=paths.label_file,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _depth_cache_path(self, sample_id: str) -> Path | None:
        if self.depth_cache_dir is None:
            return None
        return self.depth_cache_dir / f"{sample_id}_{self.stereo_method}_{self.img_size}.npy"

    def _load_depth(self, sample_id: str) -> tuple[np.ndarray, np.ndarray]:
        cache_path = self._depth_cache_path(sample_id)
        if cache_path is not None and cache_path.exists():
            depth = np.load(cache_path)
            frame = compute_depth_frame(
                dataset_root=self.dataset_root,
                split=self.split,
                sample_id=sample_id,
                cv2_module=self.cv2,
                stereo_method=self.stereo_method,
                num_disparities=self.num_disparities,
                block_size=self.block_size,
                max_depth_m=self.max_depth_m,
            )
            return frame.left_bgr, depth.astype(np.float32)

        frame = compute_depth_frame(
            dataset_root=self.dataset_root,
            split=self.split,
            sample_id=sample_id,
            cv2_module=self.cv2,
            stereo_method=self.stereo_method,
            num_disparities=self.num_disparities,
            block_size=self.block_size,
            max_depth_m=self.max_depth_m,
        )
        depth = frame.depth_m.astype(np.float32)
        if cache_path is not None:
            np.save(cache_path, depth)
        return frame.left_bgr, depth

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | dict[str, torch.Tensor]]:
        sample = self.samples[idx]
        image_bgr, depth_m = self._load_depth(sample.sample_id)
        objects = load_kitti_labels(sample.label_path, include_types=list(KITTI_CLASS_TO_ID.keys()), ignore_dontcare=True)

        img_h, img_w = image_bgr.shape[:2]

        # RGB and edge processing
        rgb = self.cv2.cvtColor(image_bgr, self.cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gray = self.cv2.cvtColor(image_bgr, self.cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gx = self.cv2.Sobel(gray, self.cv2.CV_32F, 1, 0, ksize=3)
        gy = self.cv2.Sobel(gray, self.cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx * gx + gy * gy)
        edge = edge / (edge.max() + 1e-6)

        # Depth normalization
        depth = np.clip(depth_m, 0.0, self.max_depth_m) / self.max_depth_m
        invalid = ~np.isfinite(depth)
        depth[invalid] = 0.0

        rgb_resized = self.cv2.resize(rgb, (self.img_size, self.img_size), interpolation=self.cv2.INTER_LINEAR)
        depth_resized = self.cv2.resize(depth, (self.img_size, self.img_size), interpolation=self.cv2.INTER_NEAREST)
        edge_resized = self.cv2.resize(edge, (self.img_size, self.img_size), interpolation=self.cv2.INTER_LINEAR)

        five_ch = np.concatenate(
            [
                rgb_resized,
                depth_resized[..., None],
                edge_resized[..., None],
            ],
            axis=2,
        )
        x = torch.from_numpy(np.transpose(five_ch.astype(np.float32), (2, 0, 1)))

        labels: list[int] = []
        boxes_xywh_norm: list[np.ndarray] = []
        for obj in objects:
            if obj.object_type not in KITTI_CLASS_TO_ID:
                continue
            x1, y1, x2, y2 = obj.bbox_xyxy
            if (x2 - x1) < 4 or (y2 - y1) < 4:
                continue
            labels.append(KITTI_CLASS_TO_ID[obj.object_type])
            boxes_xywh_norm.append(_to_xywh_norm(obj.bbox_xyxy, img_w=img_w, img_h=img_h))

        if boxes_xywh_norm:
            boxes_np = np.stack(boxes_xywh_norm, axis=0)
            boxes_np = _resize_boxes_xywh_norm(boxes_np)
            boxes = torch.from_numpy(boxes_np)
            cls = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            cls = torch.zeros((0,), dtype=torch.long)

        target = {
            "boxes": boxes,
            "labels": cls,
        }

        return {
            "image": x,
            "target": target,
            "sample_id": sample.sample_id,
        }


def detection_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    targets = [item["target"] for item in batch]
    sample_ids = [item["sample_id"] for item in batch]
    return {
        "images": images,
        "targets": targets,
        "sample_ids": sample_ids,
    }
