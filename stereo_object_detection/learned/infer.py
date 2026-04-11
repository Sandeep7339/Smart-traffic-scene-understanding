#!/usr/bin/env python3
import argparse
from pathlib import Path
import os
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geometric.depth import compute_depth_frame_kitti
from geometric.yolo_detect import detect_2d_boxes
from kitti_stereo.geometry_3d import box3d_corners_camera, draw_projected_box3d, project_points_p2
from kitti_stereo.stereo_depth import normalize_for_display
from learned.model import StereoFusion3DNet
from learned.utils import build_bbox_features, compute_detection_anchor, crop_rgb_depth_patch, decode_prediction_with_anchor


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with learned 3D fusion model.")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--split", type=str, default="training", choices=["training", "testing"])
    parser.add_argument("--sample-id", type=str, default="000000")

    parser.add_argument("--stereo-method", type=str, default="sgbm", choices=["sgbm", "bm"])
    parser.add_argument("--num-disparities", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--max-depth-m", type=float, default=80.0)

    parser.add_argument("--yolo-model", type=str, default=str(ROOT / "yolov8n.pt"))
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)

    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "learned_fusion_3d.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def _draw_yolo_2d(image_bgr, detections, cv2_module):
    out = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        cv2_module.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        txt = f"{det.label} {det.confidence:.2f}"
        cv2_module.putText(
            out,
            txt,
            (x1, max(15, y1 - 8)),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2_module.LINE_AA,
        )
    return out


def _depth_to_colormap(depth_m, cv2_module):
    norm = normalize_for_display(depth_m)  # [0,1], finite-aware
    gray_u8 = (255.0 * norm).astype("uint8")
    return cv2_module.applyColorMap(gray_u8, cv2_module.COLORMAP_INFERNO)


def main():
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

    import cv2
    import matplotlib.pyplot as plt

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StereoFusion3DNet(
        geom_dim=int(ckpt.get("geom_dim", 8)),
        d_model=int(ckpt.get("d_model", 128)),
        nhead=int(ckpt.get("nhead", 4)),
        num_layers=int(ckpt.get("num_layers", 2)),
        patch_size=int(ckpt.get("patch_size", 96)),
    )
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError as exc:
        raise SystemExit(
            "Checkpoint architecture mismatch with current learned model. "
            "Please retrain with `python3 learned/train.py ...` and use the new checkpoint."
        ) from exc
    model.to(device).eval()

    frame = compute_depth_frame_kitti(
        dataset_root=args.dataset_root,
        split=args.split,
        sample_id=args.sample_id,
        cv2_module=cv2,
        stereo_method=args.stereo_method,
        num_disparities=args.num_disparities,
        block_size=args.block_size,
        max_depth_m=args.max_depth_m,
    )

    detections = detect_2d_boxes(
        image_bgr=frame.left_bgr,
        model_name=args.yolo_model,
        conf_threshold=args.conf_thres,
        iou_threshold=args.iou_thres,
        max_detections=100,
        imgsz=640,
        device=args.device,
    )

    yolo_2d_img = _draw_yolo_2d(frame.left_bgr, detections, cv2)
    depth_vis = _depth_to_colormap(frame.depth_m, cv2)

    out = frame.left_bgr.copy()
    for det in detections:
        patch = crop_rgb_depth_patch(
            image_bgr=frame.left_bgr,
            depth_map_m=frame.depth_m,
            bbox_xyxy=det.bbox_xyxy,
            cv2_module=cv2,
            patch_size=int(ckpt.get("patch_size", 96)),
            max_depth_m=float(ckpt.get("max_depth_m", args.max_depth_m)),
        )
        geom = build_bbox_features(
            bbox_xyxy=det.bbox_xyxy,
            image_width=frame.left_bgr.shape[1],
            image_height=frame.left_bgr.shape[0],
        )

        patch_t = torch.from_numpy(patch).unsqueeze(0).to(device=device, dtype=torch.float32)
        geom_t = torch.from_numpy(geom).unsqueeze(0).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            pred = model(patch_t, geom_t)[0].detach().cpu().numpy()

        anchor_xyz = compute_detection_anchor(
            bbox_xyxy=det.bbox_xyxy,
            depth_map_m=frame.depth_m,
            fx_px=frame.geometry.fx_px,
            fy_px=frame.geometry.fy_px,
            cx_px=frame.geometry.cx_px,
            cy_px=frame.geometry.cy_px,
        )
        box = decode_prediction_with_anchor(pred, anchor_xyz=anchor_xyz)
        corners_3d = box3d_corners_camera(
            x_m=box["x"],
            y_m=box["y"],
            z_m=box["z"],
            height_m=box["h"],
            width_m=box["w"],
            length_m=box["l"],
            yaw_ry=box["theta"],
        )
        corners_2d = project_points_p2(corners_3d, frame.calibration["P2"])
        out = draw_projected_box3d(out, corners_2d, cv2, color=(0, 255, 0), thickness=2)

        x1, y1, _, _ = det.bbox_xyxy
        text = f"{det.label} z={box['z']:.1f}m"
        cv2.putText(out, text, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.save_dir / f"{args.split}_{frame.sample.sample_id}_learned_3dbox.png"
    depth_file = args.save_dir / f"{args.split}_{frame.sample.sample_id}_learned_depth_map.png"
    yolo_file = args.save_dir / f"{args.split}_{frame.sample.sample_id}_learned_yolo_2d.png"
    cv2.imwrite(str(out_file), out)
    cv2.imwrite(str(depth_file), depth_vis)
    cv2.imwrite(str(yolo_file), yolo_2d_img)

    print("Learned inference completed.")
    print(f"Saved: {out_file}")
    print(f"Saved: {depth_file}")
    print(f"Saved: {yolo_file}")
    print(f"Detections: {len(detections)}")

    if not args.no_show:
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        plt.title("Learned 3D Boxes")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
