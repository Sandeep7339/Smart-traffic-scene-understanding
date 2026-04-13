#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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
from learned.model import StereoFusion3DNet
from learned.utils import build_bbox_features, compute_detection_anchor, crop_rgb_depth_patch, decode_prediction_with_anchor


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
    parser.add_argument("--save-dir", type=Path, default=ROOT / "outputs")
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


def _fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _render_metrics_txt(
    args: argparse.Namespace,
    requested_samples: int,
    processed_samples: int,
    baseline_stats: dict,
    dl_stats: dict,
    dl_enabled: bool,
    dl_model_type: str,
) -> str:
    lines = [
        "Baseline vs DL 3D Fusion Metrics",
        "===============================",
        f"Split: {args.split}",
        f"Requested samples: {requested_samples}",
        f"Processed samples: {processed_samples}",
        f"Matching IoU threshold: {_fmt(args.match_iou, 2)}",
        f"DL checkpoint mode: {dl_model_type}",
        "",
        "Baseline (geometry fusion)",
        f"  matches: {baseline_stats['count']}",
        f"  Depth MAE: {_fmt(float(baseline_stats['depth_mae']))} m",
        f"  Center L2 mean: {_fmt(float(baseline_stats['center_l2_mean']))} m",
        f"  Mean IoU2D: {_fmt(float(baseline_stats['iou_mean']))}",
        "",
        "DL fusion",
    ]
    if dl_enabled:
        lines.extend(
            [
                f"  matches: {dl_stats['count']}",
                f"  Depth MAE: {_fmt(float(dl_stats['depth_mae']))} m",
                f"  Center L2 mean: {_fmt(float(dl_stats['center_l2_mean']))} m",
                f"  Mean IoU2D: {_fmt(float(dl_stats['iou_mean']))}",
            ]
        )
    else:
        lines.append("  skipped: checkpoint unavailable or incompatible")
    return "\n".join(lines) + "\n"


def _detect_checkpoint_mode(state_dict: dict) -> str:
    keys = set(state_dict.keys())
    if any(k.startswith("rgb_encoder.net.") for k in keys) or "rgb_proj.weight" in keys:
        return "legacy_stereo_fusion"
    if "rgb_modality_token" in keys or any(k.startswith("fusion_head.") for k in keys):
        return "rgbd_transformer"
    return "unknown"


def _predict_legacy_dl_center_preds(
    detections,
    frame,
    model: StereoFusion3DNet,
    device: torch.device,
    cv2_module,
    patch_size: int,
    max_depth_m: float,
) -> list[dict]:
    if not detections:
        return []

    preds: list[dict] = []
    model.eval()
    for det in detections:
        patch = crop_rgb_depth_patch(
            image_bgr=frame.left_bgr,
            depth_map_m=frame.depth_m,
            bbox_xyxy=det.bbox_xyxy,
            cv2_module=cv2_module,
            patch_size=patch_size,
            max_depth_m=max_depth_m,
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
        preds.append(
            {
                "label": det.label,
                "bbox": det.bbox_xyxy,
                "center_xyz": (float(box["x"]), float(box["y"]), float(box["z"])),
            }
        )
    return preds


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

    model: Optional[torch.nn.Module] = None
    roi_size = 96
    max_depth_model = args.max_depth_m
    dl_enabled = False
    dl_model_type = "none"
    if args.checkpoint.exists():
        try:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            state_dict = ckpt["model_state_dict"]
            dl_model_type = _detect_checkpoint_mode(state_dict)
            max_depth_model = float(ckpt.get("max_depth_m", args.max_depth_m))

            if dl_model_type == "rgbd_transformer":
                roi_size = int(ckpt.get("roi_size", 96))
                model = RGBDTransformer3DHead(
                    roi_size=roi_size,
                    geom_dim=int(ckpt.get("geom_dim", 8)),
                    d_model=int(ckpt.get("d_model", 192)),
                    num_heads=int(ckpt.get("num_heads", 8)),
                    num_layers=int(ckpt.get("num_layers", 4)),
                    dropout=float(ckpt.get("dropout", 0.1)),
                )
                model.load_state_dict(state_dict)
                model.to(device).eval()
                dl_enabled = True
            elif dl_model_type == "legacy_stereo_fusion":
                roi_size = int(ckpt.get("patch_size", 96))
                model = StereoFusion3DNet(
                    geom_dim=int(ckpt.get("geom_dim", 8)),
                    d_model=int(ckpt.get("d_model", 128)),
                    nhead=int(ckpt.get("nhead", 4)),
                    num_layers=int(ckpt.get("num_layers", 2)),
                    patch_size=roi_size,
                )
                model.load_state_dict(state_dict)
                model.to(device).eval()
                dl_enabled = True
            else:
                print(
                    f"Warning: unknown DL checkpoint architecture in {args.checkpoint}. "
                    "Expected RGBD transformer or legacy StereoFusion3DNet."
                )
        except Exception as exc:
            short_error = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            print(f"Warning: failed to load DL checkpoint ({args.checkpoint}): {short_error}")
            print("Warning: continuing with baseline-only evaluation.")
            model = None
            dl_model_type = "none"
    else:
        print(f"Warning: DL checkpoint not found: {args.checkpoint}")
        print("Warning: continuing with baseline-only evaluation.")
        dl_model_type = "none"

    sample_ids = [f"{i:06d}" for i in range(args.start_id, args.start_id + args.num_samples)]
    processed_samples = 0

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
        processed_samples += 1

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

        if dl_enabled and model is not None:
            if dl_model_type == "rgbd_transformer":
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
            else:
                dl_preds = _predict_legacy_dl_center_preds(
                    detections=detections,
                    frame=frame,
                    model=model,
                    device=device,
                    cv2_module=cv2,
                    patch_size=roi_size,
                    max_depth_m=max_depth_model,
                )
            dl_all.extend(_match_predictions_to_gt(dl_preds, gt_objs, iou_threshold=args.match_iou))

    baseline_stats = _aggregate(baseline_all)
    dl_stats = _aggregate(dl_all)
    summary = {
        "split": args.split,
        "requested_samples": len(sample_ids),
        "processed_samples": processed_samples,
        "match_iou": args.match_iou,
        "baseline": baseline_stats,
        "dl_fusion": dl_stats,
        "dl_enabled": dl_enabled,
        "dl_model_type": dl_model_type,
    }

    args.save_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = args.save_dir / "baseline_vs_dl_metrics.json"
    metrics_txt = args.save_dir / "baseline_vs_dl_metrics.txt"
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    metrics_txt.write_text(
        _render_metrics_txt(
            args=args,
            requested_samples=len(sample_ids),
            processed_samples=processed_samples,
            baseline_stats=baseline_stats,
            dl_stats=dl_stats,
            dl_enabled=dl_enabled,
            dl_model_type=dl_model_type,
        ),
        encoding="utf-8",
    )

    print("Comparison completed.")
    print(f"Samples requested: {len(sample_ids)}")
    print(f"Samples processed: {processed_samples}")
    print(f"Matching IoU threshold: {args.match_iou:.2f}")
    print("")
    print("Baseline (geometry fusion):")
    print(
        f"  matches={baseline_stats['count']} depth_MAE={baseline_stats['depth_mae']:.3f}m "
        f"center_L2={baseline_stats['center_l2_mean']:.3f}m mean_IoU2D={baseline_stats['iou_mean']:.3f}"
    )
    print("DL fusion:")
    if dl_enabled:
        print(
            f"  mode={dl_model_type} matches={dl_stats['count']} depth_MAE={dl_stats['depth_mae']:.3f}m "
            f"center_L2={dl_stats['center_l2_mean']:.3f}m mean_IoU2D={dl_stats['iou_mean']:.3f}"
        )
    else:
        print("  skipped (checkpoint unavailable or incompatible)")
    print(f"Saved metrics JSON: {metrics_json}")
    print(f"Saved metrics TXT: {metrics_txt}")


if __name__ == "__main__":
    main()
