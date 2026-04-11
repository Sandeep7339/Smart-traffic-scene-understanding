#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kitti_stereo.depth_pipeline import compute_depth_frame
from kitti_stereo.detection import run_yolo_detection
from kitti_stereo.approaches.learned.inference import draw_dl_3d_boxes, predict_dl_3d_boxes
from kitti_stereo.approaches.learned.models import RGBDTransformer3DHead
from kitti_stereo.stereo_depth import normalize_for_display


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DL-based 3D box inference using YOLO + stereo depth.")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "kitti-dataset")
    parser.add_argument("--split", type=str, default="training", choices=["training", "testing"])
    parser.add_argument("--sample-id", type=str, default="000000")
    parser.add_argument("--stereo-method", type=str, default="sgbm", choices=["sgbm", "bm"])
    parser.add_argument("--num-disparities", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--max-depth-m", type=float, default=80.0)

    parser.add_argument("--yolo-model", type=str, default=str(ROOT / "yolov8n.pt"))
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "rgbd_transformer_3d_head.pt")
    parser.add_argument("--save-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def _print_boxes(boxes) -> None:
    if not boxes:
        print("No DL 3D boxes predicted.")
        return
    print("idx | class         | conf | center_xyz(m)         | dims(h,w,l)m         | yaw(rad)")
    print("-" * 92)
    for idx, box in enumerate(boxes):
        cx, cy, cz = box.center_xyz
        h, w, l = box.dimensions
        print(
            f"{idx:>3} | {box.label:<13} | {box.confidence:>4.2f} | "
            f"({cx:>6.2f},{cy:>6.2f},{cz:>6.2f}) | ({h:>5.2f},{w:>5.2f},{l:>5.2f}) | {box.yaw_ry:>7.3f}"
        )


def main() -> None:
    args = parse_args()
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

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}\nRun scripts/train_3d_head.py first.")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    roi_size = int(ckpt.get("roi_size", 96))
    max_depth_m = float(ckpt.get("max_depth_m", args.max_depth_m))

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RGBDTransformer3DHead(
        roi_size=roi_size,
        geom_dim=int(ckpt.get("geom_dim", 8)),
        d_model=int(ckpt.get("d_model", 192)),
        num_heads=int(ckpt.get("num_heads", 8)),
        num_layers=int(ckpt.get("num_layers", 4)),
        dropout=float(ckpt.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    frame = compute_depth_frame(
        dataset_root=args.dataset_root,
        split=args.split,
        sample_id=args.sample_id,
        cv2_module=cv2,
        stereo_method=args.stereo_method,
        num_disparities=args.num_disparities,
        block_size=args.block_size,
        max_depth_m=args.max_depth_m,
    )

    detections = run_yolo_detection(
        image_bgr=frame.left_bgr,
        model_name=args.yolo_model,
        conf_threshold=args.conf_thres,
        iou_threshold=args.iou_thres,
        max_detections=args.max_detections,
        imgsz=args.imgsz,
        device=args.device,
    )
    boxes_dl = predict_dl_3d_boxes(
        detections=detections,
        left_bgr=frame.left_bgr,
        depth_map_m=frame.depth_m,
        p2_matrix=frame.calibration["P2"],
        model=model,
        device=device,
        cv2_module=cv2,
        roi_size=roi_size,
        max_depth_m=max_depth_m,
    )

    annotated = draw_dl_3d_boxes(frame.left_bgr, boxes_dl, cv2_module=cv2)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    depth_vis = normalize_for_display(frame.depth_m)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    img_out = args.save_dir / f"{args.split}_{frame.sample.sample_id}_dl3d_annotated.png"
    panel_out = args.save_dir / f"{args.split}_{frame.sample.sample_id}_dl3d_panel.png"
    cv2.imwrite(str(img_out), annotated)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(annotated_rgb)
    axes[0].set_title("DL 3D Boxes (YOLO + Stereo Depth)")
    axes[0].axis("off")
    depth_plot = axes[1].imshow(depth_vis, cmap="inferno")
    axes[1].set_title("Depth Map (normalized)")
    axes[1].axis("off")
    fig.colorbar(depth_plot, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(panel_out, dpi=150, bbox_inches="tight")

    print("DL fusion inference completed.")
    print(f"Sample:      {frame.sample.sample_id} ({frame.sample.split})")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Saved image: {img_out}")
    print(f"Saved panel: {panel_out}")
    print(f"Detections:  {len(detections)}")
    print(f"DL 3D boxes: {len(boxes_dl)}")
    _print_boxes(boxes_dl)

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
