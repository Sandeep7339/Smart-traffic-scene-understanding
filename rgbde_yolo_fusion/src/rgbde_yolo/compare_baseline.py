import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .data_prep import prepare_kitti_yolo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fine-tuned YOLOv8n vs baseline yolov8n.pt on KITTI val.")
    parser.add_argument("--dataset-root", type=Path, default=Path("../dataset"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("outputs/ultralytics_data"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/comparisons/results.json"))
    parser.add_argument("--output-md", type=Path, default=Path("outputs/comparisons/results.md"))
    parser.add_argument("--vis-count", type=int, default=8)
    parser.add_argument("--vis-dir", type=Path, default=Path("outputs/comparisons/visualizations"))
    parser.add_argument("--sample-id", type=str, default="")
    parser.add_argument("--sample-split", type=str, default="training", choices=["training", "testing"])
    return parser.parse_args()


def _safe_float(x, default=float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _eval_map50(model: YOLO, data_yaml: Path, img_size: int, batch_size: int, device: str) -> float:
    m = model.val(
        data=str(data_yaml),
        imgsz=img_size,
        batch=batch_size,
        device=device if device else None,
        split="val",
        verbose=False,
    )
    return _safe_float(getattr(m.box, "map50", float("nan")))


def _make_separate_visuals(
    baseline_model: YOLO,
    tuned_model: YOLO,
    val_image_paths: list[Path],
    vis_dir: Path,
    img_size: int,
    device: str,
    max_count: int,
) -> dict[str, list[str]]:
    """Generate separate baseline and fine-tuned visualizations."""
    vis_dir.mkdir(parents=True, exist_ok=True)
    baseline_saved: list[str] = []
    tuned_saved: list[str] = []

    count = max(0, min(max_count, len(val_image_paths)))
    for i in range(count):
        image_path = val_image_paths[i]

        base_pred = baseline_model.predict(
            source=str(image_path),
            imgsz=img_size,
            device=device if device else None,
            verbose=False,
        )[0]
        tuned_pred = tuned_model.predict(
            source=str(image_path),
            imgsz=img_size,
            device=device if device else None,
            verbose=False,
        )[0]

        left = base_pred.plot()
        right = tuned_pred.plot()

        # Save baseline
        baseline_out = f"baseline_{image_path.stem}.png"
        baseline_path = vis_dir / baseline_out
        cv2.imwrite(str(baseline_path), left)
        baseline_saved.append(str(baseline_path))

        # Save fine-tuned
        tuned_out = f"finetuned_{image_path.stem}.png"
        tuned_path = vis_dir / tuned_out
        cv2.imwrite(str(tuned_path), right)
        tuned_saved.append(str(tuned_path))

    return {"baseline": baseline_saved, "finetuned": tuned_saved}


def main() -> None:
    args = parse_args()

    data_yaml = prepare_kitti_yolo_dataset(
        dataset_root=args.dataset_root,
        out_dir=args.prepared_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_samples=args.max_samples,
    )

    tuned_model = YOLO(str(args.checkpoint))
    base_model = YOLO(str(args.baseline_model))

    tuned_map50 = _eval_map50(tuned_model, data_yaml, args.img_size, args.batch_size, args.device)
    base_map50 = _eval_map50(base_model, data_yaml, args.img_size, args.batch_size, args.device)

    if args.sample_id:
        sid = args.sample_id.zfill(6)
        sample_image = args.dataset_root / "data_object_image_2" / args.sample_split / "image_2" / f"{sid}.png"
        if not sample_image.exists():
            raise SystemExit(f"Sample image not found: {sample_image}")
        val_images = [sample_image]
    else:
        val_dir = args.prepared_dir / "images" / "val"
        val_images = sorted(val_dir.glob("*.png")) + sorted(val_dir.glob("*.jpg"))
    vis_map = _make_separate_visuals(
        baseline_model=base_model,
        tuned_model=tuned_model,
        val_image_paths=val_images,
        vis_dir=args.vis_dir,
        img_size=args.img_size,
        device=args.device,
        max_count=args.vis_count,
    )

    out = {
        "prepared_data_yaml": str(data_yaml),
        "finetuned_checkpoint": str(args.checkpoint),
        "baseline_model": str(args.baseline_model),
        "finetuned_map50": tuned_map50,
        "baseline_map50": base_map50,
        "delta_map50": float(tuned_map50 - base_map50),
        "baseline_images": vis_map["baseline"],
        "finetuned_images": vis_map["finetuned"],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md_lines = [
        "# Baseline vs Fine-tuned YOLOv8n Comparison",
        "",
        f"Data YAML: {data_yaml}",
        "",
        "| Method | mAP@0.50 |",
        "|---|---:|",
        f"| Baseline YOLOv8n ({args.baseline_model}) | {base_map50:.4f} |",
        f"| Fine-tuned YOLOv8n ({args.checkpoint}) | {tuned_map50:.4f} |",
        f"| Delta (Fine-tuned - Baseline) | {out['delta_map50']:.4f} |",
        "",
        "## Predictions",
        "",
    ]
    if vis_map["baseline"] or vis_map["finetuned"]:
        md_lines.append("### Baseline Model")
        md_lines.append("")
        for p in vis_map["baseline"]:
            md_lines.append(f"- {p}")
        md_lines.append("")
        md_lines.append("### Fine-tuned Model")
        md_lines.append("")
        for p in vis_map["finetuned"]:
            md_lines.append(f"- {p}")
    else:
        md_lines.append("No predictions were generated.")
    md_lines.append("")
    args.output_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
