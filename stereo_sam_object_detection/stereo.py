
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils import LoadedSample, load_sample
from utils import ensure_legacy_src_on_path

ensure_legacy_src_on_path()

from kitti_stereo import compute_disparity_bm, compute_disparity_sgbm, disparity_to_depth  # noqa: E402


@dataclass(frozen=True)
class StereoFrame:
    loaded: LoadedSample
    disparity: np.ndarray
    depth_m: np.ndarray


def compute_depth(
    loaded: LoadedSample,
    cv2_module,
    stereo_method: str = "sgbm",
    num_disparities: int = 128,
    block_size: int = 7,
    max_depth_m: float = 80.0,
) -> StereoFrame:
    left_gray = cv2_module.cvtColor(loaded.left_bgr, cv2_module.COLOR_BGR2GRAY)
    right_gray = cv2_module.cvtColor(loaded.right_bgr, cv2_module.COLOR_BGR2GRAY)

    stereo_method = stereo_method.lower().strip()
    if stereo_method == "sgbm":
        disparity = compute_disparity_sgbm(
            left_gray=left_gray,
            right_gray=right_gray,
            cv2_module=cv2_module,
            min_disparity=0,
            num_disparities=num_disparities,
            block_size=block_size,
        )
    elif stereo_method == "bm":
        bm_block_size = max(9, block_size)
        if bm_block_size % 2 == 0:
            bm_block_size += 1
        disparity = compute_disparity_bm(
            left_gray=left_gray,
            right_gray=right_gray,
            cv2_module=cv2_module,
            num_disparities=num_disparities,
            block_size=bm_block_size,
        )
    else:
        raise ValueError(f"Unsupported stereo method '{stereo_method}'. Use 'sgbm' or 'bm'.")

    depth_m = disparity_to_depth(
        disparity=disparity,
        focal_length_px=loaded.geometry.fx_px,
        baseline_m=loaded.geometry.baseline_m,
        min_disparity_px=0.1,
        max_depth_m=max_depth_m,
    )
    return StereoFrame(loaded=loaded, disparity=disparity, depth_m=depth_m)


def load_and_compute_depth(
    dataset_root: str | Path,
    split: str,
    sample_id: str,
    cv2_module,
    stereo_method: str = "sgbm",
    num_disparities: int = 128,
    block_size: int = 7,
    max_depth_m: float = 80.0,
    load_labels: bool = False,
) -> StereoFrame:
    loaded = load_sample(
        dataset_root=dataset_root,
        split=split,
        sample_id=sample_id,
        cv2_module=cv2_module,
        load_labels=load_labels,
    )
    return compute_depth(
        loaded=loaded,
        cv2_module=cv2_module,
        stereo_method=stereo_method,
        num_disparities=num_disparities,
        block_size=block_size,
        max_depth_m=max_depth_m,
    )
