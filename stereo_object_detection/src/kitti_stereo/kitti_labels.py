from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class KittiLabelObject:
    object_type: str
    truncation: float
    occlusion: int
    alpha: float
    bbox_left: float
    bbox_top: float
    bbox_right: float
    bbox_bottom: float
    height_m: float
    width_m: float
    length_m: float
    x_m: float
    y_m: float
    z_m: float
    rotation_y: float

    @property
    def bbox_xyxy(self) -> tuple[int, int, int, int]:
        return (
            int(round(self.bbox_left)),
            int(round(self.bbox_top)),
            int(round(self.bbox_right)),
            int(round(self.bbox_bottom)),
        )


_DEFAULT_TRAIN_TYPES = (
    "Car",
    "Van",
    "Truck",
    "Pedestrian",
    "Person_sitting",
    "Cyclist",
    "Tram",
)


def parse_kitti_label_line(line: str) -> KittiLabelObject:
    parts = line.strip().split()
    if len(parts) < 15:
        raise ValueError(f"Invalid KITTI label line with {len(parts)} columns: {line}")

    return KittiLabelObject(
        object_type=parts[0],
        truncation=float(parts[1]),
        occlusion=int(float(parts[2])),
        alpha=float(parts[3]),
        bbox_left=float(parts[4]),
        bbox_top=float(parts[5]),
        bbox_right=float(parts[6]),
        bbox_bottom=float(parts[7]),
        height_m=float(parts[8]),
        width_m=float(parts[9]),
        length_m=float(parts[10]),
        x_m=float(parts[11]),
        y_m=float(parts[12]),
        z_m=float(parts[13]),
        rotation_y=float(parts[14]),
    )


def load_kitti_labels(
    label_path: str | Path,
    include_types: Sequence[str] | None = None,
    ignore_dontcare: bool = True,
) -> List[KittiLabelObject]:
    label_path = Path(label_path)
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    type_set = set(include_types) if include_types is not None else None
    objects: List[KittiLabelObject] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = parse_kitti_label_line(line)
            if ignore_dontcare and obj.object_type == "DontCare":
                continue
            if type_set is not None and obj.object_type not in type_set:
                continue
            objects.append(obj)
    return objects


def default_training_types() -> tuple[str, ...]:
    return _DEFAULT_TRAIN_TYPES


def labels_to_numpy(objects: Iterable[KittiLabelObject]) -> np.ndarray:
    rows = []
    for obj in objects:
        rows.append(
            [
                obj.x_m,
                obj.y_m,
                obj.z_m,
                obj.height_m,
                obj.width_m,
                obj.length_m,
                obj.rotation_y,
            ]
        )
    if not rows:
        return np.zeros((0, 7), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)
