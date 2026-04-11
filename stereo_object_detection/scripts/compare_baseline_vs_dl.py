#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kitti_stereo.box3d import estimate_3d_boxes_from_detections
from kitti_stereo.class_mapping import is_detection_label_compatible
from kitti_stereo.depth_pipeline import compute_depth_frame
from kitti_stereo.detection import run_yolo_detection
from kitti_stereo.approaches.learned.inference import predict_dl_3d_boxes
from kitti_stereo.approaches.learned.models import RGBDTransformer3DHead
from kitti_stereo.geometry_3d import bbox_iou_xyxy
from kitti_stereo.kitti_data import get_kitti_sample_paths
from kitti_stereo.kitti_labels import default_training_types, load_kitti_labels


@dataclass(frozen=True)
class EvalMatch:
    abs_depth_err_m: float
    center_l2_m: float
    iou_2d: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare geometric baseline 3D fusion vs DL 3D fusion on KITTI.")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "kitti-dataset")
    parser.add_argument("--split", type=str, default="training", choices=["training"])
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=50)
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
    parser.add_argument("--match-iou", type=float, default=0.3)
    return parser.parse_args()


def _gt_objects_for_sample(dataset_root: Path, sample_id: str):
    sample = get_kitti_sample_paths(dataset_root, "training", sample_id)
    return load_kitti_labels(sample.label_file, include_types=default_training_types(), ignore_dontcare=True)


def _match_predictions_to_gt(
    pred_items: Sequence[dict],
    gt_objects,
    iou_threshold: float,
) -> List[EvalMatch]:
    matches: List[EvalMatch] = []
    used_gt = set()
    for pred in pred_items:
        best_idx = -1
        best_iou = 0.0
        for gi, gt in enumerate(gt_objects):
            if gi in used_gt:
                continue
            if not is_detection_label_compatible(pred["label"], gt.object_type):
                continue
            iou = bbox_iou_xyxy(pred["bbox"], gt.bbox_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_idx = gi

        if best_idx < 0 or best_iou < iou_threshold:
            continue
        used_gt.add(best_idx)
        gt = gt_objects[best_idx]
        px, py, pz = pred["center_xyz"]
        abs_depth_err = abs(float(pz) - float(gt.z_m))
        center_l2 = math.sqrt((float(px) - float(gt.x_m)) ** 2 + (float(py) - float(gt.y_m)) ** 2 + (float(pz) - float(gt.z_m)) ** 2)
        matches.append(EvalMatch(abs_depth_err_m=abs_depth_err, center_l2_m=center_l2, iou_2d=best_iou))
    return matches


def _aggregate(matches: Sequence[EvalMatch]) -> dict:
    if not matches:
        return {
            "count": 0,
            "depth_mae": float("nan"),
            "center_l2_mean": float("nan"),
            "iou_mean": float("nan"),
        }
    depth = np.asarray([m.abs_depth_err_m for m in matches], dtype=np.float32)
    l2 = np.asarray([m.center_l2_m for m in matches], dtype=np.float32)
    iou = np.asarray([m.iou_2d for m in matches], dtype=np.float32)
    return {
        "count": int(len(matches)),
        "depth_mae": float(depth.mean()),
        "center_l2_mean": float(l2.mean()),
        "iou_mean": float(iou.mean()),
    }


def main() -> None:
    args = parse_args()

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is required. Install with: pip install -r requirements.txt") from exc

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    roi_size = 96
    max_depth_model = args.max_depth_m
    if args.checkpoint.exists():
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        roi_size = int(ckpt.get("roi_size", 96))
        max_depth_model = float(ckpt.get("max_depth_m", args.max_depth_m))
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
    else:
        raise SystemExit(f"DL checkpoint not found: {args.checkpoint}")

    sample_ids = [f"{i:06d}" for i in range(args.start_id, args.start_id + args.num_samples)]

    baseline_all: List[EvalMatch] = []
    dl_all: List[EvalMatch] = []

    for sid in sample_ids:
        try:
            frame = compute_depth_frame(
                dataset_root=args.dataset_root,
                split="training",
                sample_id=sid,
                cv2_module=cv2,
                stereo_method=args.stereo_method,
                num_disparities=args.num_disparities,
                block_size=args.block_size,
                max_depth_m=args.max_depth_m,
            )
            gt_objs = _gt_objects_for_sample(args.dataset_root, sid)
        except FileNotFoundError:
            continue

        detections = run_yolo_detection(
            image_bgr=frame.left_bgr,
            model_name=args.yolo_model,
            conf_threshold=args.conf_thres,
            iou_threshold=args.iou_thres,
            max_detections=args.max_detections,
            imgsz=args.imgsz,
            device=args.device,
        )

        baseline_boxes = estimate_3d_boxes_from_detections(
            detections=detections,
            depth_map_m=frame.depth_m,
            fx_px=frame.geometry.fx_px,
            fy_px=frame.geometry.fy_px,
            cx_px=frame.geometry.cx_px,
            cy_px=frame.geometry.cy_px,
            p2_matrix=frame.calibration["P2"],
            min_depth_m=0.1,
            max_depth_m=args.max_depth_m,
            inner_ratio=0.65,
            min_points=50,
        )
        baseline_preds = [
            {
                "label": b.label,
                "bbox": b.bbox_xyxy,
                "center_xyz": b.center_xyz,
            }
            for b in baseline_boxes
        ]
        baseline_all.extend(_match_predictions_to_gt(baseline_preds, gt_objs, iou_threshold=args.match_iou))

        dl_boxes = predict_dl_3d_boxes(
            detections=detections,
            left_bgr=frame.left_bgr,
            depth_map_m=frame.depth_m,
            p2_matrix=frame.calibration["P2"],
            model=model,
            device=device,
            cv2_module=cv2,
            roi_size=roi_size,
            max_depth_m=max_depth_model,
        )
        dl_preds = [
            {
                "label": b.label,
                "bbox": b.bbox_xyxy,
                "center_xyz": b.center_xyz,
            }
            for b in dl_boxes
        ]
        dl_all.extend(_match_predictions_to_gt(dl_preds, gt_objs, iou_threshold=args.match_iou))

    baseline_stats = _aggregate(baseline_all)
    dl_stats = _aggregate(dl_all)

    print("Comparison completed.")
    print(f"Samples attempted: {len(sample_ids)}")
    print(f"Matching IoU threshold: {args.match_iou:.2f}")
    print("")
    print("Baseline (geometry fusion):")
    print(
        f"  matches={baseline_stats['count']} depth_MAE={baseline_stats['depth_mae']:.3f}m "
        f"center_L2={baseline_stats['center_l2_mean']:.3f}m mean_IoU2D={baseline_stats['iou_mean']:.3f}"
    )
    print("DL fusion:")
    print(
        f"  matches={dl_stats['count']} depth_MAE={dl_stats['depth_mae']:.3f}m "
        f"center_L2={dl_stats['center_l2_mean']:.3f}m mean_IoU2D={dl_stats['iou_mean']:.3f}"
    )


if __name__ == "__main__":
    main()
