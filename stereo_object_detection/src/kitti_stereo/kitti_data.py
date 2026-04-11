from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class KittiSamplePaths:
    split: str
    sample_id: str
    left_image: Path
    right_image: Path
    calib_file: Path
    label_file: Path | None


@dataclass(frozen=True)
class StereoGeometry:
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    baseline_m: float
    left_camera_key: str = "P2"
    right_camera_key: str = "P3"


def _normalize_sample_id(sample_id: str | int) -> str:
    return f"{int(sample_id):06d}" if isinstance(sample_id, int) else sample_id.zfill(6)


def get_kitti_sample_paths(dataset_root: str | Path, split: str, sample_id: str | int) -> KittiSamplePaths:
    dataset_root = Path(dataset_root).expanduser().resolve()
    split = split.lower().strip()
    if split not in {"training", "testing"}:
        raise ValueError(f"Unsupported split '{split}'. Use 'training' or 'testing'.")

    sid = _normalize_sample_id(sample_id)

    left_image = dataset_root / "data_object_image_2" / split / "image_2" / f"{sid}.png"
    right_image = dataset_root / "data_object_image_3" / split / "image_3" / f"{sid}.png"
    calib_file = dataset_root / "data_object_calib" / split / "calib" / f"{sid}.txt"
    label_file = (
        dataset_root / "data_object_label_2" / "training" / "label_2" / f"{sid}.txt"
        if split == "training"
        else None
    )

    required_paths = [left_image, right_image, calib_file]
    missing = [str(p) for p in required_paths if not p.exists()]
    if split == "training" and label_file is not None and not label_file.exists():
        missing.append(str(label_file))
    if missing:
        raise FileNotFoundError("Missing KITTI files:\n" + "\n".join(missing))

    return KittiSamplePaths(
        split=split,
        sample_id=sid,
        left_image=left_image,
        right_image=right_image,
        calib_file=calib_file,
        label_file=label_file,
    )


def load_calibration(calib_path: str | Path) -> dict[str, np.ndarray]:
    calib_path = Path(calib_path)
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    calibration: dict[str, np.ndarray] = {}
    with calib_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, values = line.split(":", 1)
            data = np.fromstring(values, sep=" ", dtype=np.float64)

            if key.startswith("P"):
                calibration[key] = data.reshape(3, 4)
            elif key == "R0_rect":
                calibration[key] = data.reshape(3, 3)
            elif key.startswith("Tr_"):
                calibration[key] = data.reshape(3, 4)
            else:
                calibration[key] = data
    return calibration


def stereo_geometry_from_calibration(
    calibration: dict[str, np.ndarray],
    left_key: str = "P2",
    right_key: str = "P3",
) -> StereoGeometry:
    if left_key not in calibration or right_key not in calibration:
        raise KeyError(f"Calibration must contain {left_key} and {right_key}.")

    p_left = calibration[left_key]
    p_right = calibration[right_key]

    fx_left = float(p_left[0, 0])
    fy_left = float(p_left[1, 1])
    fx_right = float(p_right[0, 0])

    left_center_x = -float(p_left[0, 3]) / fx_left
    right_center_x = -float(p_right[0, 3]) / fx_right
    baseline = abs(right_center_x - left_center_x)

    return StereoGeometry(
        fx_px=fx_left,
        fy_px=fy_left,
        cx_px=float(p_left[0, 2]),
        cy_px=float(p_left[1, 2]),
        baseline_m=baseline,
        left_camera_key=left_key,
        right_camera_key=right_key,
    )


def load_stereo_pair(left_path: str | Path, right_path: str | Path, cv2_module: Any) -> tuple[np.ndarray, np.ndarray]:
    left = cv2_module.imread(str(left_path), cv2_module.IMREAD_COLOR)
    right = cv2_module.imread(str(right_path), cv2_module.IMREAD_COLOR)
    if left is None or right is None:
        raise RuntimeError(f"Failed to load stereo images: {left_path} | {right_path}")
    return left, right
