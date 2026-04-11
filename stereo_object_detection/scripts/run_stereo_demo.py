#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kitti_stereo import (  # noqa: E402
    compute_disparity_bm,
    compute_disparity_sgbm,
    disparity_to_depth,
    get_kitti_sample_paths,
    load_calibration,
    load_stereo_pair,
    normalize_for_display,
    stereo_geometry_from_calibration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KITTI stereo loading + disparity/depth demo.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "kitti-dataset",
        help="Path to extracted KITTI dataset root.",
    )
    parser.add_argument("--split", type=str, default="training", choices=["training", "testing"])
    parser.add_argument("--sample-id", type=str, default="000000", help="Frame id, e.g., 000000")
    parser.add_argument("--method", type=str, default="sgbm", choices=["sgbm", "bm"])
    parser.add_argument("--num-disparities", type=int, default=128, help="Must be divisible by 16.")
    parser.add_argument("--block-size", type=int, default=7, help="Odd, >= 5.")
    parser.add_argument("--max-depth-m", type=float, default=80.0)
    parser.add_argument("--save-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--no-show", action="store_true", help="Skip interactive plot display.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "OpenCV is required. Install dependencies with:\n"
            "pip install -r requirements.txt"
        ) from exc

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Matplotlib is required. Install dependencies with:\n"
            "pip install -r requirements.txt"
        ) from exc

    sample = get_kitti_sample_paths(args.dataset_root, args.split, args.sample_id)
    calibration = load_calibration(sample.calib_file)
    geometry = stereo_geometry_from_calibration(calibration, left_key="P2", right_key="P3")

    left_bgr, right_bgr = load_stereo_pair(sample.left_image, sample.right_image, cv2)
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    if args.method == "sgbm":
        disparity = compute_disparity_sgbm(
            left_gray,
            right_gray,
            cv2_module=cv2,
            min_disparity=0,
            num_disparities=args.num_disparities,
            block_size=args.block_size,
        )
    else:
        bm_block_size = args.block_size if args.block_size >= 9 else 9
        if bm_block_size % 2 == 0:
            bm_block_size += 1
        disparity = compute_disparity_bm(
            left_gray,
            right_gray,
            cv2_module=cv2,
            num_disparities=args.num_disparities,
            block_size=bm_block_size,
        )

    depth_m = disparity_to_depth(
        disparity,
        focal_length_px=geometry.fx_px,
        baseline_m=geometry.baseline_m,
        max_depth_m=args.max_depth_m,
    )

    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
    disparity_vis = normalize_for_display(disparity)

    depth_vis = depth_m.copy()
    if not (depth_vis.shape == disparity.shape):
        raise RuntimeError("Depth/disparity shape mismatch.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0, 0].imshow(left_rgb)
    axes[0, 0].set_title(f"Left (image_2) - {sample.sample_id}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(right_rgb)
    axes[0, 1].set_title(f"Right (image_3) - {sample.sample_id}")
    axes[0, 1].axis("off")

    disp_plot = axes[1, 0].imshow(disparity_vis, cmap="plasma")
    axes[1, 0].set_title(f"Disparity ({args.method.upper()})")
    axes[1, 0].axis("off")
    fig.colorbar(disp_plot, ax=axes[1, 0], fraction=0.046, pad=0.04)

    depth_plot = axes[1, 1].imshow(depth_vis, cmap="inferno", vmin=0.0, vmax=args.max_depth_m)
    axes[1, 1].set_title("Depth (meters)")
    axes[1, 1].axis("off")
    fig.colorbar(depth_plot, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"fx={geometry.fx_px:.2f}px | baseline={geometry.baseline_m:.4f}m | split={sample.split}",
        fontsize=12,
    )
    plt.tight_layout()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.save_dir / f"{sample.split}_{sample.sample_id}_{args.method}_stereo_depth.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Saved visualization: {out_file}")
    print(f"Left image:  {sample.left_image}")
    print(f"Right image: {sample.right_image}")
    print(f"Calib file:  {sample.calib_file}")
    if sample.label_file is not None:
        print(f"Label file:  {sample.label_file}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
