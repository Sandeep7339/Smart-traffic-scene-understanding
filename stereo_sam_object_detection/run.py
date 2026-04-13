#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np

from fusion import aggregate_matches, draw_fused_predictions, fuse_detections, match_predictions_to_gt
from sam_segment import SamSegmenter
from stereo import load_and_compute_depth
from utils import apply_mask_overlay, depth_to_colormap, ensure_dir, iter_sample_ids, resolve_device
from yolo import detect_objects

ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stereo + YOLOv8 + SAM geometric depth fusion.")

    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--split", type=str, default="training", choices=["training", "testing"])
    parser.add_argument("--sample-id", type=str, default=None, help="Single sample id (e.g. 000000).")
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--random-samples", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random-seed", type=int, default=None)

    parser.add_argument("--stereo-method", type=str, default="sgbm", choices=["sgbm", "bm"])
    parser.add_argument("--num-disparities", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--min-depth-m", type=float, default=0.1)
    parser.add_argument("--max-depth-m", type=float, default=80.0)

    parser.add_argument("--yolo-model", type=str, default=str(ROOT / "yolov8n.pt"))
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--sam-checkpoint", type=Path, default=ROOT / "sam_b.pt")
    parser.add_argument("--sam-model-type", type=str, default="vit_b")
    parser.add_argument("--sam-device", type=str, default=None)

    parser.add_argument("--lower-percentile", type=float, default=5.0)
    parser.add_argument("--upper-percentile", type=float, default=95.0)
    parser.add_argument("--min-points", type=int, default=40)
    parser.add_argument("--match-iou", type=float, default=0.3)

    parser.add_argument("--output-root", type=Path, default=MODULE_ROOT / "outputs")
    parser.add_argument("--global-compare-dir", type=Path, default=ROOT / "outputs" / "stereo_sam_vs_baseline")
    parser.add_argument("--save-depth-map", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean-legacy-baseline-output", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _to_float(v: float) -> float:
    return float(v) if np.isfinite(v) else float("nan")


def _pct_improvement(old: float, new: float) -> float:
    if not np.isfinite(old) or old == 0.0 or not np.isfinite(new):
        return float("nan")
    return 100.0 * (old - new) / old


def _fmt(v: float, digits: int = 4) -> str:
    if not np.isfinite(v):
        return "nan"
    return f"{float(v):.{digits}f}"


def _draw_yolo_2d(image_bgr: np.ndarray, detections, cv2_module) -> np.ndarray:
    out = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        cv2_module.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        txt = f"{det.label} {det.confidence:.2f}"
        cv2_module.putText(
            out,
            txt,
            (x1, max(18, y1 - 8)),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2_module.LINE_AA,
        )
    return out


def _draw_sam_segmented(image_bgr: np.ndarray, detections, masks: list[np.ndarray], cv2_module) -> np.ndarray:
    out = image_bgr.copy()
    colors = [(255, 120, 0), (0, 180, 255), (180, 255, 0), (255, 80, 140)]
    for idx, det in enumerate(detections):
        if idx < len(masks) and masks[idx] is not None:
            out = apply_mask_overlay(out, masks[idx], color_bgr=colors[idx % len(colors)], alpha=0.35)
        cv2_module.putText(
            out,
            f"{det.label} {det.confidence:.2f}",
            (12, min(out.shape[0] - 10, 24 + 18 * idx)),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2_module.LINE_AA,
        )
    cv2_module.putText(
        out,
        "SAM segmented output",
        (12, 28),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2_module.LINE_AA,
    )
    return out


def _list_available_sample_ids(dataset_root: Path, split: str) -> list[str]:
    image_dir = dataset_root / "data_object_image_2" / split / "image_2"
    if not image_dir.exists():
        return []
    return sorted(p.stem for p in image_dir.glob("*.png"))


def _select_sample_ids(args: argparse.Namespace) -> list[str]:
    if args.sample_id is not None:
        return [args.sample_id.zfill(6)]

    if args.random_samples:
        available = _list_available_sample_ids(args.dataset_root, args.split)
        if not available:
            return []
        k = min(int(args.num_samples), len(available))
        rng = random.Random(args.random_seed)
        return rng.sample(available, k)

    return iter_sample_ids(sample_id=None, start_id=args.start_id, num_samples=args.num_samples)


def _draw_3d_edges(
    image_bgr: np.ndarray,
    corners_2d: np.ndarray,
    cv2_module,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for a, b in edges:
        p1 = corners_2d[a]
        p2 = corners_2d[b]
        if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
            continue
        c1 = (int(round(float(p1[0]))), int(round(float(p1[1]))))
        c2 = (int(round(float(p2[0]))), int(round(float(p2[1]))))
        cv2_module.line(image_bgr, c1, c2, color, thickness, cv2_module.LINE_AA)


def _draw_compare_overlay(
    image_bgr: np.ndarray,
    baseline_predictions,
    sam_predictions,
    cv2_module,
) -> np.ndarray:
    out = image_bgr.copy()
    baseline_color = (0, 70, 255)  # red-ish
    sam_color = (0, 220, 0)  # green

    for pred in baseline_predictions:
        _draw_3d_edges(out, pred.corners_2d, cv2_module, baseline_color, thickness=2)

    for pred in sam_predictions:
        _draw_3d_edges(out, pred.corners_2d, cv2_module, sam_color, thickness=2)

    cv2_module.rectangle(out, (12, 14), (26, 28), baseline_color, -1)
    cv2_module.putText(
        out,
        "Baseline 3D box",
        (32, 27),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2_module.LINE_AA,
    )
    cv2_module.rectangle(out, (170, 14), (184, 28), sam_color, -1)
    cv2_module.putText(
        out,
        "SAM fusion 3D box",
        (190, 27),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2_module.LINE_AA,
    )
    return out


def _build_metrics_summary(
    attempted: int,
    split: str,
    match_iou: float,
    baseline_stats: dict[str, float],
    sam_stats: dict[str, float],
) -> dict[str, float | dict[str, float] | int | str]:
    return {
        "attempted_samples": attempted,
        "split": split,
        "match_iou": match_iou,
        "baseline": baseline_stats,
        "sam_fusion": sam_stats,
        "rmse_improvement_percent": _pct_improvement(baseline_stats["rmse"], sam_stats["rmse"]),
    }


def _render_metrics_txt(summary: dict[str, float | dict[str, float] | int | str]) -> str:
    baseline = summary["baseline"]
    sam = summary["sam_fusion"]
    assert isinstance(baseline, dict)
    assert isinstance(sam, dict)

    lines = [
        "Stereo + YOLO + SAM Metrics",
        "===========================",
        f"Split: {summary['split']}",
        f"Attempted samples: {summary['attempted_samples']}",
        f"Matching IoU threshold: {_fmt(float(summary['match_iou']), 2)}",
        "",
        "Baseline (YOLO bbox + Stereo)",
        f"  matches: {int(float(baseline['count']))}",
        f"  MAE: {_fmt(float(baseline['mae']))} m",
        f"  RMSE: {_fmt(float(baseline['rmse']))} m",
        f"  Mean depth variance: {_fmt(float(baseline['mean_var']), 6)}",
        f"  Projected IoU: {_fmt(float(baseline['proj_iou']))}",
        f"  Region IoU: {_fmt(float(baseline['region_iou']))}",
        "",
        "SAM Fusion (YOLO prompt -> SAM mask + Stereo)",
        f"  matches: {int(float(sam['count']))}",
        f"  MAE: {_fmt(float(sam['mae']))} m",
        f"  RMSE: {_fmt(float(sam['rmse']))} m",
        f"  Mean depth variance: {_fmt(float(sam['mean_var']), 6)}",
        f"  Projected IoU: {_fmt(float(sam['proj_iou']))}",
        f"  Region IoU: {_fmt(float(sam['region_iou']))}",
        "",
        f"RMSE Improvement (%): {_fmt(float(summary['rmse_improvement_percent']), 2)}",
    ]
    return "\n".join(lines) + "\n"


def _write_metrics_files(
    output_root: Path,
    summary: dict[str, float | dict[str, float] | int | str],
) -> tuple[Path, Path]:
    metrics_dir = ensure_dir(output_root / "metrics")
    json_path = metrics_dir / "comparison_metrics.json"
    txt_path = metrics_dir / "comparison_metrics.txt"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    txt_path.write_text(_render_metrics_txt(summary), encoding="utf-8")
    return json_path, txt_path


def _print_final_report(summary: dict[str, float | dict[str, float] | int | str]) -> None:
    baseline_stats = summary["baseline"]
    sam_stats = summary["sam_fusion"]
    assert isinstance(baseline_stats, dict)
    assert isinstance(sam_stats, dict)

    print("")
    print("Final Comparison")
    print("================")
    print(f"MAE (baseline): {_to_float(float(baseline_stats['mae'])):.4f}")
    print(f"MAE (SAM): {_to_float(float(sam_stats['mae'])):.4f}")
    print(f"RMSE (baseline): {_to_float(float(baseline_stats['rmse'])):.4f}")
    print(f"RMSE (SAM): {_to_float(float(sam_stats['rmse'])):.4f}")
    print(f"RMSE improvement: {_to_float(float(summary['rmse_improvement_percent'])):.2f}%")
    print(f"Depth variance (baseline): {_to_float(float(baseline_stats['mean_var'])):.6f}")
    print(f"Depth variance (SAM): {_to_float(float(sam_stats['mean_var'])):.6f}")

    if np.isfinite(float(baseline_stats["proj_iou"])) or np.isfinite(float(sam_stats["proj_iou"])):
        print(f"Projected IoU (baseline): {_to_float(float(baseline_stats['proj_iou'])):.4f}")
        print(f"Projected IoU (SAM): {_to_float(float(sam_stats['proj_iou'])):.4f}")

    if np.isfinite(float(baseline_stats["region_iou"])) or np.isfinite(float(sam_stats["region_iou"])):
        print(f"Region IoU (baseline bbox): {_to_float(float(baseline_stats['region_iou'])):.4f}")
        print(f"Region IoU (SAM mask bbox): {_to_float(float(sam_stats['region_iou'])):.4f}")


def main() -> None:
    args = parse_args()

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is required. Install with: python3 -m pip install opencv-python") from exc

    device = resolve_device(args.device)
    sam_device = resolve_device(args.sam_device if args.sam_device is not None else args.device)

    sample_ids = _select_sample_ids(args)
    if len(sample_ids) == 0:
        print("No sample ids were selected. Check dataset path/split.")
        return
    print(f"Selected sample ids ({len(sample_ids)}): {', '.join(sample_ids)}")

    output_root = ensure_dir(args.output_root)
    legacy_baseline_dir = output_root / "baseline"
    if args.clean_legacy_baseline_output and legacy_baseline_dir.exists():
        shutil.rmtree(legacy_baseline_dir)

    sam_depth_dir = ensure_dir(output_root / "sam" / "depth_map")
    sam_segmented_dir = ensure_dir(output_root / "sam" / "segmented")
    sam_2d_dir = ensure_dir(output_root / "sam" / "yolo_2d")
    sam_fusion_dir = ensure_dir(output_root / "sam" / "fusion_3d")
    global_compare_dir = ensure_dir(args.global_compare_dir)

    segmenter = SamSegmenter(
        checkpoint_path=args.sam_checkpoint,
        model_type=args.sam_model_type,
        device=sam_device,
    )

    baseline_matches = []
    sam_matches = []
    attempted = 0

    for sid in sample_ids:
        try:
            frame = load_and_compute_depth(
                dataset_root=args.dataset_root,
                split=args.split,
                sample_id=sid,
                cv2_module=cv2,
                stereo_method=args.stereo_method,
                num_disparities=args.num_disparities,
                block_size=args.block_size,
                max_depth_m=args.max_depth_m,
                load_labels=(args.split == "training"),
            )
        except FileNotFoundError:
            continue

        attempted += 1

        detections = detect_objects(
            image_bgr=frame.loaded.left_bgr,
            model_name=args.yolo_model,
            conf_threshold=args.conf_thres,
            iou_threshold=args.iou_thres,
            max_detections=args.max_detections,
            imgsz=args.imgsz,
            device=device,
        )

        sam_masks = segmenter.predict_masks(frame.loaded.left_bgr, detections)

        baseline_predictions = fuse_detections(
            detections=detections,
            depth_map_m=frame.depth_m,
            fx_px=frame.loaded.geometry.fx_px,
            fy_px=frame.loaded.geometry.fy_px,
            cx_px=frame.loaded.geometry.cx_px,
            cy_px=frame.loaded.geometry.cy_px,
            p2_matrix=frame.loaded.calibration["P2"],
            masks=None,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            lower_percentile=args.lower_percentile,
            upper_percentile=args.upper_percentile,
            min_points=args.min_points,
        )

        sam_predictions = fuse_detections(
            detections=detections,
            depth_map_m=frame.depth_m,
            fx_px=frame.loaded.geometry.fx_px,
            fy_px=frame.loaded.geometry.fy_px,
            cx_px=frame.loaded.geometry.cx_px,
            cy_px=frame.loaded.geometry.cy_px,
            p2_matrix=frame.loaded.calibration["P2"],
            masks=sam_masks,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            lower_percentile=args.lower_percentile,
            upper_percentile=args.upper_percentile,
            min_points=args.min_points,
        )

        sam_vis = draw_fused_predictions(
            image_bgr=frame.loaded.left_bgr,
            predictions=sam_predictions,
            cv2_module=cv2,
            show_masks=True,
            draw_detection_bbox=False,
        )
        compare_overlay = _draw_compare_overlay(
            image_bgr=frame.loaded.left_bgr,
            baseline_predictions=baseline_predictions,
            sam_predictions=sam_predictions,
            cv2_module=cv2,
        )
        yolo_2d_vis = _draw_yolo_2d(frame.loaded.left_bgr, detections, cv2)
        sam_segmented_vis = _draw_sam_segmented(frame.loaded.left_bgr, detections, sam_masks, cv2)

        sam_out = sam_fusion_dir / f"{args.split}_{sid}_sam_fusion_3d.png"
        sam_seg_out = sam_segmented_dir / f"{args.split}_{sid}_sam_segmented.png"
        yolo_out = sam_2d_dir / f"{args.split}_{sid}_yolo_2d.png"
        compare_out = global_compare_dir / f"{args.split}_{sid}_baseline_vs_sam.png"

        cv2.imwrite(str(sam_out), sam_vis)
        cv2.imwrite(str(sam_seg_out), sam_segmented_vis)
        cv2.imwrite(str(yolo_out), yolo_2d_vis)
        cv2.imwrite(str(compare_out), compare_overlay)

        if args.save_depth_map:
            depth_vis = depth_to_colormap(frame.depth_m, cv2)
            cv2.imwrite(str(sam_depth_dir / f"{args.split}_{sid}_depth_map.png"), depth_vis)

        if frame.loaded.labels is not None:
            baseline_matches.extend(match_predictions_to_gt(baseline_predictions, frame.loaded.labels, iou_threshold=args.match_iou))
            sam_matches.extend(match_predictions_to_gt(sam_predictions, frame.loaded.labels, iou_threshold=args.match_iou))

        print(
            f"sample={sid} det={len(detections)} "
            f"baseline_valid={sum(np.isfinite([p.depth_m for p in baseline_predictions]))} "
            f"sam_valid={sum(np.isfinite([p.depth_m for p in sam_predictions]))}"
        )

    baseline_stats = aggregate_matches(baseline_matches)
    sam_stats = aggregate_matches(sam_matches)

    summary = _build_metrics_summary(
        attempted=attempted,
        split=args.split,
        match_iou=args.match_iou,
        baseline_stats=baseline_stats,
        sam_stats=sam_stats,
    )
    metrics_json_path, metrics_txt_path = _write_metrics_files(output_root, summary)

    print("")
    print(f"Attempted samples: {attempted}")
    print(f"Local outputs saved to: {output_root}")
    print(f"Overlay comparisons: {global_compare_dir}")
    print(f"Metrics JSON: {metrics_json_path}")
    print(f"Metrics TXT: {metrics_txt_path}")

    if attempted == 0:
        print("No valid samples found for the provided range.")
        return

    if args.split != "training":
        print("Ground-truth labels are unavailable for testing split; report contains NaN metrics.")
        return

    _print_final_report(summary)


if __name__ == "__main__":
    main()
