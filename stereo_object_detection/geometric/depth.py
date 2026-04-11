from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kitti_stereo.depth_pipeline import compute_depth_frame
from kitti_stereo.stereo_depth import compute_disparity_bm, compute_disparity_sgbm, disparity_to_depth


def compute_stereo_depth_from_arrays(
    left_gray,
    right_gray,
    cv2_module,
    method="sgbm",
    num_disparities=128,
    block_size=7,
    fx_px=None,
    baseline_m=None,
    max_depth_m=80.0,
):
    if method == "sgbm":
        disparity = compute_disparity_sgbm(
            left_gray=left_gray,
            right_gray=right_gray,
            cv2_module=cv2_module,
            min_disparity=0,
            num_disparities=num_disparities,
            block_size=block_size,
        )
    elif method == "bm":
        bm_block = block_size if block_size >= 9 else 9
        if bm_block % 2 == 0:
            bm_block += 1
        disparity = compute_disparity_bm(
            left_gray=left_gray,
            right_gray=right_gray,
            cv2_module=cv2_module,
            num_disparities=num_disparities,
            block_size=bm_block,
        )
    else:
        raise ValueError("method must be 'sgbm' or 'bm'.")

    depth_m = None
    if fx_px is not None and baseline_m is not None:
        depth_m = disparity_to_depth(
            disparity=disparity,
            focal_length_px=float(fx_px),
            baseline_m=float(baseline_m),
            min_disparity_px=0.1,
            max_depth_m=float(max_depth_m),
        )
    return disparity, depth_m


def compute_depth_frame_kitti(
    dataset_root,
    split,
    sample_id,
    cv2_module,
    stereo_method="sgbm",
    num_disparities=128,
    block_size=7,
    max_depth_m=80.0,
):
    return compute_depth_frame(
        dataset_root=dataset_root,
        split=split,
        sample_id=sample_id,
        cv2_module=cv2_module,
        stereo_method=stereo_method,
        num_disparities=num_disparities,
        block_size=block_size,
        max_depth_m=max_depth_m,
    )
