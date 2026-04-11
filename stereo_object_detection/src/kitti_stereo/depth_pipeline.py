from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .kitti_data import (
    KittiSamplePaths,
    StereoGeometry,
    get_kitti_sample_paths,
    load_calibration,
    load_stereo_pair,
    stereo_geometry_from_calibration,
)
from .stereo_depth import compute_disparity_bm, compute_disparity_sgbm, disparity_to_depth


@dataclass(frozen=True)
class DepthFrame:
    sample: KittiSamplePaths
    calibration: dict[str, np.ndarray]
    geometry: StereoGeometry
    left_bgr: np.ndarray
    right_bgr: np.ndarray
    disparity: np.ndarray
    depth_m: np.ndarray


def compute_depth_frame(
    dataset_root: str | Path,
    split: str,
    sample_id: str | int,
    cv2_module: Any,
    stereo_method: str = "sgbm",
    num_disparities: int = 128,
    block_size: int = 7,
    max_depth_m: float = 80.0,
) -> DepthFrame:
    sample = get_kitti_sample_paths(dataset_root, split, sample_id)
    calibration = load_calibration(sample.calib_file)
    geometry = stereo_geometry_from_calibration(calibration, left_key="P2", right_key="P3")
    left_bgr, right_bgr = load_stereo_pair(sample.left_image, sample.right_image, cv2_module)
    left_gray = cv2_module.cvtColor(left_bgr, cv2_module.COLOR_BGR2GRAY)
    right_gray = cv2_module.cvtColor(right_bgr, cv2_module.COLOR_BGR2GRAY)

    if stereo_method == "sgbm":
        disparity = compute_disparity_sgbm(
            left_gray,
            right_gray,
            cv2_module=cv2_module,
            min_disparity=0,
            num_disparities=num_disparities,
            block_size=block_size,
        )
    elif stereo_method == "bm":
        bm_block = block_size if block_size >= 9 else 9
        if bm_block % 2 == 0:
            bm_block += 1
        disparity = compute_disparity_bm(
            left_gray,
            right_gray,
            cv2_module=cv2_module,
            num_disparities=num_disparities,
            block_size=bm_block,
        )
    else:
        raise ValueError(f"Unsupported stereo_method '{stereo_method}'. Use 'sgbm' or 'bm'.")

    depth_m = disparity_to_depth(
        disparity=disparity,
        focal_length_px=geometry.fx_px,
        baseline_m=geometry.baseline_m,
        min_disparity_px=0.1,
        max_depth_m=max_depth_m,
    )
    return DepthFrame(
        sample=sample,
        calibration=calibration,
        geometry=geometry,
        left_bgr=left_bgr,
        right_bgr=right_bgr,
        disparity=disparity,
        depth_m=depth_m,
    )
