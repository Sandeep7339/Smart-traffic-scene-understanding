import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from .constants import KITTI_CLASSES
from .data_prep import prepare_kitti_yolo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned YOLOv8n on KITTI val split.")
    parser.add_argument("--dataset-root", type=Path, default=Path("../dataset"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("outputs/ultralytics_data"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--results-path", type=Path, default=Path("outputs/results/eval_metrics.json"))
    return parser.parse_args()


def _safe_float(x, default=float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def main() -> None:
    args = parse_args()

    data_yaml = prepare_kitti_yolo_dataset(
        dataset_root=args.dataset_root,
        out_dir=args.prepared_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_samples=args.max_samples,
    )

    model = YOLO(str(args.checkpoint))
    metrics = model.val(
        data=str(data_yaml),
        imgsz=args.img_size,
        batch=args.batch_size,
        device=args.device if args.device else None,
        split="val",
    )

    out = {
        "checkpoint": str(args.checkpoint),
        "prepared_data_yaml": str(data_yaml),
        "num_classes": len(KITTI_CLASSES),
        "map50": _safe_float(getattr(metrics.box, "map50", float("nan"))),
        "map50_95": _safe_float(getattr(metrics.box, "map", float("nan"))),
        "mp": _safe_float(getattr(metrics.box, "mp", float("nan"))),
        "mr": _safe_float(getattr(metrics.box, "mr", float("nan"))),
    }

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
