#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

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
    run_yolo_detection,
    stereo_geometry_from_calibration,
)
from kitti_stereo.approaches.geometric import draw_3d_boxes, estimate_3d_boxes_from_detections  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KITTI fusion demo: YOLO + stereo depth + 3D projection.")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "kitti-dataset")
    parser.add_argument("--split", type=str, default="training", choices=["training", "testing"])
    parser.add_argument("--sample-id", type=str, default="000000")

    parser.add_argument("--stereo-method", type=str, default="sgbm", choices=["sgbm", "bm"])
    parser.add_argument("--num-disparities", type=int, default=128, help="Must be divisible by 16.")
    parser.add_argument("--block-size", type=int, default=7, help="Odd, >= 5.")
    parser.add_argument("--max-depth-m", type=float, default=80.0)
    parser.add_argument("--min-depth-m", type=float, default=0.1)

    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None, help="e.g. cpu, 0")
    parser.add_argument("--inner-ratio", type=float, default=0.65, help="Central-box ratio for depth seeding.")
    parser.add_argument("--min-box-points", type=int, default=50, help="Minimum depth points for 3D box fitting.")

    parser.add_argument("--save-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--no-show", action="store_true")
    return parser


def _compute_disparity(left_gray, right_gray, cv2, method: str, num_disparities: int, block_size: int):
    if method == "sgbm":
        return compute_disparity_sgbm(
            left_gray=left_gray,
            right_gray=right_gray,
            cv2_module=cv2,
            min_disparity=0,
            num_disparities=num_disparities,
            block_size=block_size,
        )

    bm_block_size = block_size if block_size >= 9 else 9
    if bm_block_size % 2 == 0:
        bm_block_size += 1
    return compute_disparity_bm(
        left_gray=left_gray,
        right_gray=right_gray,
        cv2_module=cv2,
        num_disparities=num_disparities,
        block_size=bm_block_size,
    )


def _print_box3d_table(boxes_3d) -> None:
    if not boxes_3d:
        print("No valid 3D boxes estimated.")
        return

    print(
        "idx | class         | conf | bbox(x1,y1,x2,y2)      | "
        "center_xyz (m)         | dims_xyz (m)           | depth_seed | pts"
    )
    print("-" * 132)
    for idx, obj in enumerate(boxes_3d):
        x1, y1, x2, y2 = obj.bbox_xyxy
        cx, cy, cz = obj.center_xyz
        dx, dy, dz = obj.dimensions_xyz

        print(
            f"{idx:>3} | {obj.label:<13} | {obj.confidence:>4.2f} | "
            f"({x1:>4},{y1:>4},{x2:>4},{y2:>4}) | "
            f"({cx:>6.2f},{cy:>6.2f},{cz:>6.2f}) | "
            f"({dx:>6.2f},{dy:>6.2f},{dz:>6.2f}) | {obj.depth_seed_m:>10.2f} | {obj.num_points:>4}"
        )


def main() -> None:
    args = build_parser().parse_args()

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is required. Install with: pip install -r requirements.txt") from exc

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Matplotlib is required. Install with: pip install -r requirements.txt") from exc

    sample = get_kitti_sample_paths(args.dataset_root, args.split, args.sample_id)
    calibration = load_calibration(sample.calib_file)
    geometry = stereo_geometry_from_calibration(calibration, left_key="P2", right_key="P3")

    left_bgr, right_bgr = load_stereo_pair(sample.left_image, sample.right_image, cv2)
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    disparity = _compute_disparity(
        left_gray=left_gray,
        right_gray=right_gray,
        cv2=cv2,
        method=args.stereo_method,
        num_disparities=args.num_disparities,
        block_size=args.block_size,
    )
    depth_m = disparity_to_depth(
        disparity=disparity,
        focal_length_px=geometry.fx_px,
        baseline_m=geometry.baseline_m,
        min_disparity_px=0.1,
        max_depth_m=args.max_depth_m,
    )

    detections = run_yolo_detection(
        image_bgr=left_bgr,
        model_name=args.yolo_model,
        conf_threshold=args.conf_thres,
        iou_threshold=args.iou_thres,
        max_detections=args.max_detections,
        imgsz=args.imgsz,
        device=args.device,
    )
    p2 = calibration["P2"]
    boxes_3d = estimate_3d_boxes_from_detections(
        detections=detections,
        depth_map_m=depth_m,
        fx_px=geometry.fx_px,
        fy_px=geometry.fy_px,
        cx_px=geometry.cx_px,
        cy_px=geometry.cy_px,
        p2_matrix=p2,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        inner_ratio=args.inner_ratio,
        min_points=args.min_box_points,
    )

    annotated_bgr = draw_3d_boxes(
        image_bgr=left_bgr,
        boxes_3d=boxes_3d,
        cv2_module=cv2,
    )
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    depth_vis = normalize_for_display(depth_m)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    img_out = args.save_dir / f"{args.split}_{sample.sample_id}_fusion_3dbox_annotated.png"
    panel_out = args.save_dir / f"{args.split}_{sample.sample_id}_fusion_3dbox_panel.png"
    cv2.imwrite(str(img_out), annotated_bgr)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(annotated_rgb)
    axes[0].set_title("Detections + Depth/3D Overlay")
    axes[0].axis("off")

    depth_plot = axes[1].imshow(depth_vis, cmap="inferno")
    axes[1].set_title("Depth Map (normalized)")
    axes[1].axis("off")
    fig.colorbar(depth_plot, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Sample {sample.sample_id} | fx={geometry.fx_px:.2f}px baseline={geometry.baseline_m:.4f}m",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(panel_out, dpi=150, bbox_inches="tight")

    print("Fusion run completed.")
    print(f"Sample:      {sample.sample_id} ({sample.split})")
    print(f"Left image:  {sample.left_image}")
    print(f"Right image: {sample.right_image}")
    print(f"Calib file:  {sample.calib_file}")
    print(f"Saved image: {img_out}")
    print(f"Saved panel: {panel_out}")
    print(f"Detections:  {len(detections)}")
    print(f"3D boxes:    {len(boxes_3d)}")
    _print_box3d_table(boxes_3d)

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
