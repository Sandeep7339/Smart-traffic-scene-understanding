from __future__ import annotations

from typing import Any

import numpy as np


def _validate_num_disparities(num_disparities: int) -> int:
    if num_disparities <= 0:
        raise ValueError("num_disparities must be > 0.")
    if num_disparities % 16 != 0:
        raise ValueError("num_disparities must be divisible by 16.")
    return num_disparities


def _validate_block_size(block_size: int) -> int:
    if block_size < 5 or block_size % 2 == 0:
        raise ValueError("block_size must be odd and >= 5.")
    return block_size


def compute_disparity_sgbm(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    cv2_module: Any,
    min_disparity: int = 0,
    num_disparities: int = 128,
    block_size: int = 7,
) -> np.ndarray:
    num_disparities = _validate_num_disparities(num_disparities)
    block_size = _validate_block_size(block_size)

    channels = 1
    p1 = 8 * channels * block_size**2
    p2 = 32 * channels * block_size**2

    matcher = cv2_module.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2_module.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disparity[disparity <= float(min_disparity)] = np.nan
    return disparity


def compute_disparity_bm(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    cv2_module: Any,
    num_disparities: int = 128,
    block_size: int = 15,
) -> np.ndarray:
    num_disparities = _validate_num_disparities(num_disparities)
    block_size = _validate_block_size(block_size)

    matcher = cv2_module.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    disparity = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disparity[disparity <= 0.0] = np.nan
    return disparity


def disparity_to_depth(
    disparity: np.ndarray,
    focal_length_px: float,
    baseline_m: float,
    min_disparity_px: float = 0.1,
    max_depth_m: float | None = 80.0,
) -> np.ndarray:
    depth = np.full_like(disparity, np.nan, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > min_disparity_px)
    depth[valid] = (focal_length_px * baseline_m) / disparity[valid]
    if max_depth_m is not None:
        depth[depth > max_depth_m] = np.nan
    return depth


def normalize_for_display(image: np.ndarray, lower_percentile: float = 2.0, upper_percentile: float = 98.0) -> np.ndarray:
    result = np.zeros_like(image, dtype=np.float32)
    valid = np.isfinite(image)
    if not np.any(valid):
        return result

    lo, hi = np.percentile(image[valid], [lower_percentile, upper_percentile])
    if np.isclose(hi, lo):
        result[valid] = 1.0
        return result

    result[valid] = np.clip((image[valid] - lo) / (hi - lo), 0.0, 1.0)
    return result
