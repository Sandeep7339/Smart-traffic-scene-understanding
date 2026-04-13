from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .constants import DEFAULT_SGBM_CONFIG, KITTI_CLASS_TO_ID
from .utils import xyxy_to_xywhn


@dataclass
class SplitInfo:
    train_ids: list[str]
    val_ids: list[str]
    seed: int
    val_fraction: float


def parse_kitti_label_file(label_path: Path) -> list[dict[str, Any]]:
    anns: list[dict[str, Any]] = []
    if not label_path.exists():
        return anns

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 15:
            continue

        cls_name = parts[0]
        if cls_name not in KITTI_CLASS_TO_ID:
            continue

        truncation = float(parts[1])
        occlusion = int(float(parts[2]))

        x1 = float(parts[4])
        y1 = float(parts[5])
        x2 = float(parts[6])
        y2 = float(parts[7])

        if x2 <= x1 or y2 <= y1:
            continue

        anns.append(
            {
                "class_name": cls_name,
                "class_id": KITTI_CLASS_TO_ID[cls_name],
                "bbox_xyxy": [x1, y1, x2, y2],
                "truncation": truncation,
                "occlusion": occlusion,
            }
        )

    return anns


def build_train_val_split(
    dataset_root: str | Path,
    val_fraction: float,
    seed: int,
    max_samples: int = 0,
) -> SplitInfo:
    dataset_root = Path(dataset_root).expanduser().resolve()
    label_dir = dataset_root / "data_object_label_2" / "training" / "label_2"
    sample_ids = sorted(p.stem for p in label_dir.glob("*.txt"))
    if max_samples > 0:
        sample_ids = sample_ids[:max_samples]

    if len(sample_ids) < 2:
        raise RuntimeError("Need at least 2 labeled KITTI samples for train/val split.")

    rng = random.Random(seed)
    rng.shuffle(sample_ids)

    val_count = max(1, int(len(sample_ids) * val_fraction))
    val_count = min(val_count, len(sample_ids) - 1)

    val_ids = sorted(sample_ids[:val_count])
    train_ids = sorted(sample_ids[val_count:])
    return SplitInfo(train_ids=train_ids, val_ids=val_ids, seed=seed, val_fraction=val_fraction)


def summarize_label_quality(dataset_root: str | Path, sample_ids: list[str]) -> dict[str, int]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    label_dir = dataset_root / "data_object_label_2" / "training" / "label_2"

    total_objects = 0
    empty_files = 0

    for sid in sample_ids:
        anns = parse_kitti_label_file(label_dir / f"{sid}.txt")
        total_objects += len(anns)
        if len(anns) == 0:
            empty_files += 1

    return {
        "num_samples": len(sample_ids),
        "total_objects": total_objects,
        "empty_label_files": empty_files,
    }


def _normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    min_v = float(np.min(x))
    max_v = float(np.max(x))
    if max_v - min_v < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return (x - min_v) / (max_v - min_v)


def compute_depth_map(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(**DEFAULT_SGBM_CONFIG)
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disparity[disparity <= 0] = np.nan

    # Relative depth proxy from disparity. Absolute metric depth needs calibrated intrinsics.
    inv_disp = 1.0 / (disparity + 1e-6)
    inv_disp = np.nan_to_num(inv_disp, nan=0.0, posinf=0.0, neginf=0.0)

    valid = inv_disp > 0
    if np.any(valid):
        clip = np.percentile(inv_disp[valid], 99.0)
        inv_disp = np.clip(inv_disp, 0.0, clip)
    return _normalize_map(inv_disp)


def compute_sobel_edge(rgb_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    return _normalize_map(edge_mag)


class KITTIMultiModalDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        sample_ids: list[str],
        img_size: int,
        use_depth: bool,
        use_edge: bool,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.sample_ids = list(sample_ids)
        self.img_size = int(img_size)
        self.use_depth = bool(use_depth)
        self.use_edge = bool(use_edge)

        self.left_dir = self.dataset_root / "data_object_image_2" / "training" / "image_2"
        self.right_dir = self.dataset_root / "data_object_image_3" / "training" / "image_3"
        self.label_dir = self.dataset_root / "data_object_label_2" / "training" / "label_2"

        self.annotations: dict[str, list[dict[str, Any]]] = {
            sid: parse_kitti_label_file(self.label_dir / f"{sid}.txt") for sid in self.sample_ids
        }

        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        if self.cache_dir is not None and (self.use_depth or self.use_edge):
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_modalities(self, sid: str, left_bgr: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not (self.use_depth or self.use_edge):
            return None, None

        cache_file = self.cache_dir / f"{sid}.npz" if self.cache_dir is not None else None
        if cache_file is not None and cache_file.exists():
            cached = np.load(str(cache_file))
            depth = cached["depth"].astype(np.float32) if self.use_depth else None
            edge = cached["edge"].astype(np.float32) if self.use_edge else None
            return depth, edge

        right_path = self.right_dir / f"{sid}.png"
        right_bgr = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if right_bgr is None:
            raise FileNotFoundError(f"Missing right stereo image: {right_path}")

        rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)

        depth = compute_depth_map(left_bgr, right_bgr) if self.use_depth else None
        edge = compute_sobel_edge(rgb) if self.use_edge else None

        if cache_file is not None:
            payload = {}
            if depth is not None:
                payload["depth"] = depth.astype(np.float16)
            if edge is not None:
                payload["edge"] = edge.astype(np.float16)
            np.savez_compressed(str(cache_file), **payload)

        return depth, edge

    def __getitem__(self, index: int) -> dict[str, Any]:
        sid = self.sample_ids[index]
        left_path = self.left_dir / f"{sid}.png"
        left_bgr = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        if left_bgr is None:
            raise FileNotFoundError(f"Missing left RGB image: {left_path}")

        h0, w0 = left_bgr.shape[:2]
        rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)

        anns = self.annotations[sid]

        boxes_xyxy: list[list[float]] = []
        classes: list[int] = []
        has_occlusion = False
        has_small_object = False
        min_area = 0.01 * float(h0 * w0)

        for ann in anns:
            x1, y1, x2, y2 = ann["bbox_xyxy"]
            x1 = float(np.clip(x1, 0.0, w0 - 1.0))
            x2 = float(np.clip(x2, 0.0, w0 - 1.0))
            y1 = float(np.clip(y1, 0.0, h0 - 1.0))
            y2 = float(np.clip(y2, 0.0, h0 - 1.0))
            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1) * (y2 - y1)
            if area < min_area:
                has_small_object = True
            if int(ann["occlusion"]) >= 2:
                has_occlusion = True

            boxes_xyxy.append([x1, y1, x2, y2])
            classes.append(int(ann["class_id"]))

        depth_map, edge_map = self._load_modalities(sid, left_bgr)

        resized_rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        rgb_float = resized_rgb.astype(np.float32) / 255.0

        channels = [rgb_float]
        if self.use_depth:
            assert depth_map is not None
            depth_rs = cv2.resize(depth_map, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            channels.append(depth_rs[..., None])
        if self.use_edge:
            assert edge_map is not None
            edge_rs = cv2.resize(edge_map, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            channels.append(edge_rs[..., None])

        image = np.concatenate(channels, axis=2).astype(np.float32)
        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).contiguous()

        if boxes_xyxy:
            boxes_np = np.array(boxes_xyxy, dtype=np.float32)
            boxes_np[:, [0, 2]] *= float(self.img_size) / float(w0)
            boxes_np[:, [1, 3]] *= float(self.img_size) / float(h0)
            boxes_np[:, [0, 2]] = np.clip(boxes_np[:, [0, 2]], 0, self.img_size - 1)
            boxes_np[:, [1, 3]] = np.clip(boxes_np[:, [1, 3]], 0, self.img_size - 1)
            boxes_xywhn = xyxy_to_xywhn(boxes_np, width=self.img_size, height=self.img_size)
            cls_tensor = torch.tensor(classes, dtype=torch.float32).view(-1, 1)
            box_tensor = torch.tensor(boxes_xywhn, dtype=torch.float32)
        else:
            cls_tensor = torch.zeros((0, 1), dtype=torch.float32)
            box_tensor = torch.zeros((0, 4), dtype=torch.float32)

        return {
            "img": image_tensor,
            "cls": cls_tensor,
            "bboxes": box_tensor,
            "image_id": sid,
            "rgb_vis": resized_rgb,
            "orig_shape": (h0, w0),
            "difficulty": {
                "has_occlusion": has_occlusion,
                "has_small_object": has_small_object,
            },
        }


def kitti_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    imgs = torch.stack([sample["img"] for sample in batch], dim=0)

    cls_list: list[torch.Tensor] = []
    box_list: list[torch.Tensor] = []
    batch_idx_list: list[torch.Tensor] = []

    for i, sample in enumerate(batch):
        cls = sample["cls"]
        boxes = sample["bboxes"]
        if cls.shape[0] == 0:
            continue
        cls_list.append(cls)
        box_list.append(boxes)
        batch_idx_list.append(torch.full((cls.shape[0],), i, dtype=torch.int64))

    if cls_list:
        cls_out = torch.cat(cls_list, dim=0)
        boxes_out = torch.cat(box_list, dim=0)
        batch_idx_out = torch.cat(batch_idx_list, dim=0)
    else:
        cls_out = torch.zeros((0, 1), dtype=torch.float32)
        boxes_out = torch.zeros((0, 4), dtype=torch.float32)
        batch_idx_out = torch.zeros((0,), dtype=torch.int64)

    return {
        "img": imgs,
        "cls": cls_out,
        "bboxes": boxes_out,
        "batch_idx": batch_idx_out,
        "image_ids": [sample["image_id"] for sample in batch],
        "rgb_vis": [sample["rgb_vis"] for sample in batch],
        "orig_shapes": [sample["orig_shape"] for sample in batch],
        "difficulties": [sample["difficulty"] for sample in batch],
    }


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=kitti_collate_fn,
        drop_last=False,
    )
