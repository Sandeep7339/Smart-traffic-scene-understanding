#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometric.depth import compute_depth_frame_kitti
from geometric.fusion_math import draw_projected_3d_boxes, estimate_3d_boxes
from geometric.yolo_detect import detect_2d_boxes
from kitti_stereo.stereo_depth import normalize_for_display


def parse_args():
    parser = argparse.ArgumentParser(description="Run geometric stereo + YOLO + 3D fusion pipeline.")
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
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
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
        max_detections=args.max_detections,
        imgsz=args.imgsz,
        device=args.device,
    )
    yolo_2d_img = _draw_yolo_2d(frame.left_bgr, detections, cv2)
    depth_vis = _depth_to_colormap(frame.depth_m, cv2)

    boxes_3d = estimate_3d_boxes(
        detections=detections,
        depth_map_m=frame.depth_m,
        fx_px=frame.geometry.fx_px,
        fy_px=frame.geometry.fy_px,
        cx_px=frame.geometry.cx_px,
        cy_px=frame.geometry.cy_px,
        p2_matrix=frame.calibration["P2"],
        min_depth_m=0.1,
        max_depth_m=args.max_depth_m,
    )

    annotated = draw_projected_3d_boxes(frame.left_bgr, boxes_3d, cv2)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.save_dir / f"{args.split}_{frame.sample.sample_id}_geo_3dbox.png"
    depth_file = args.save_dir / f"{args.split}_{frame.sample.sample_id}_geo_depth_map.png"
    yolo_file = args.save_dir / f"{args.split}_{frame.sample.sample_id}_geo_yolo_2d.png"
    cv2.imwrite(str(out_file), annotated)
    cv2.imwrite(str(depth_file), depth_vis)
    cv2.imwrite(str(yolo_file), yolo_2d_img)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.imshow(annotated_rgb)
    ax.set_title("Geometric 3D Boxes")
    ax.axis("off")
    fig.tight_layout()

    print("Geometric pipeline completed.")
    print(f"Saved: {out_file}")
    print(f"Saved: {depth_file}")
    print(f"Saved: {yolo_file}")
    print(f"Detections: {len(detections)}")
    print(f"3D boxes: {len(boxes_3d)}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
