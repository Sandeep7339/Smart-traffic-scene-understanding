
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LEGACY_SRC = ROOT / "stereo_object_detection" / "src"


def ensure_legacy_src_on_path() -> None:
    legacy_str = str(LEGACY_SRC)
    if legacy_str not in sys.path:
        sys.path.insert(0, legacy_str)


ensure_legacy_src_on_path()

from kitti_stereo import (  # noqa: E402
    KittiLabelObject,
    KittiSamplePaths,
    StereoGeometry,
    default_training_types,
    get_kitti_sample_paths,
    load_calibration,
    load_kitti_labels,
    load_stereo_pair,
    stereo_geometry_from_calibration,
)


@dataclass(frozen=True)
class LoadedSample:
    sample: KittiSamplePaths
    calibration: dict[str, np.ndarray]
    geometry: StereoGeometry
    left_bgr: np.ndarray
    right_bgr: np.ndarray
    labels: Optional[Sequence[KittiLabelObject]]


def resolve_device(requested: Optional[str] = None) -> str:
    if requested:
        return requested

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def iter_sample_ids(sample_id: Optional[str], start_id: int, num_samples: int) -> list[str]:
    if sample_id is not None:
        return [sample_id.zfill(6)]
    return [f"{idx:06d}" for idx in range(start_id, start_id + num_samples)]


def load_sample(
    dataset_root: str | Path,
    split: str,
    sample_id: str,
    cv2_module,
    load_labels: bool = False,
) -> LoadedSample:
    sample = get_kitti_sample_paths(dataset_root, split, sample_id)
    calibration = load_calibration(sample.calib_file)
    geometry = stereo_geometry_from_calibration(calibration, left_key="P2", right_key="P3")
    left_bgr, right_bgr = load_stereo_pair(sample.left_image, sample.right_image, cv2_module)

    labels: Optional[Sequence[KittiLabelObject]] = None
    if load_labels and sample.label_file is not None:
        labels = load_kitti_labels(
            sample.label_file,
            include_types=default_training_types(),
            ignore_dontcare=True,
        )

    return LoadedSample(
        sample=sample,
        calibration=calibration,
        geometry=geometry,
        left_bgr=left_bgr,
        right_bgr=right_bgr,
        labels=labels,
    )


def depth_to_colormap(depth_map_m: np.ndarray, cv2_module, lower_pct: float = 2.0, upper_pct: float = 98.0) -> np.ndarray:
    vis = np.zeros_like(depth_map_m, dtype=np.float32)
    valid = np.isfinite(depth_map_m)
    if np.any(valid):
        lo, hi = np.percentile(depth_map_m[valid], [lower_pct, upper_pct])
        if not np.isclose(lo, hi):
            vis[valid] = np.clip((depth_map_m[valid] - lo) / (hi - lo), 0.0, 1.0)
        else:
            vis[valid] = 1.0
    u8 = (255.0 * vis).astype(np.uint8)
    return cv2_module.applyColorMap(u8, cv2_module.COLORMAP_INFERNO)


def mask_bbox_xyxy(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def apply_mask_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int] = (255, 120, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    out = image_bgr.copy()
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    if not np.any(mask):
        return out

    color = np.zeros_like(out, dtype=np.uint8)
    color[:, :] = np.array(color_bgr, dtype=np.uint8)
    out[mask] = np.clip((1.0 - alpha) * out[mask] + alpha * color[mask], 0, 255).astype(np.uint8)
    return out


def compose_side_by_side(left_bgr: np.ndarray, right_bgr: np.ndarray, cv2_module) -> np.ndarray:
    if left_bgr.shape[0] != right_bgr.shape[0]:
        target_h = min(left_bgr.shape[0], right_bgr.shape[0])
        left_bgr = cv2_module.resize(left_bgr, (int(left_bgr.shape[1] * target_h / left_bgr.shape[0]), target_h))
        right_bgr = cv2_module.resize(right_bgr, (int(right_bgr.shape[1] * target_h / right_bgr.shape[0]), target_h))
    return np.hstack([left_bgr, right_bgr])


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def finite_rmse(errors: Iterable[float]) -> float:
    arr = np.asarray(list(errors), dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(arr**2)))
