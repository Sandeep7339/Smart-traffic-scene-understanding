import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

from .constants import KITTI_CLASSES
from .data_prep import prepare_kitti_yolo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8n (pretrained) on KITTI RGB labels.")
    parser.add_argument("--dataset-root", type=Path, default=Path("../dataset"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("outputs/ultralytics_data"))
    parser.add_argument("--weights", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=Path, default=Path("outputs/results"))
    parser.add_argument("--name", type=str, default="yolov8n_rgb_finetune")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
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

    model = YOLO(str(args.weights))
    train_results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=args.device if args.device else None,
        project=str(args.project),
        name=args.name,
        seed=args.seed,
        exist_ok=True,
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    save_dir = Path(train_results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"

    if best_pt.exists():
        shutil.copy2(best_pt, args.checkpoint_dir / "fusion_yolo_best.pt")
    if last_pt.exists():
        shutil.copy2(last_pt, args.checkpoint_dir / "fusion_yolo_last.pt")

    metrics = {
        "model": "yolov8n.pt fine-tune",
        "num_classes": len(KITTI_CLASSES),
        "classes": KITTI_CLASSES,
        "prepared_data_yaml": str(data_yaml),
        "save_dir": str(save_dir),
        "metrics": {
            "map50": _safe_float(getattr(train_results.box, "map50", float("nan"))),
            "map50_95": _safe_float(getattr(train_results.box, "map", float("nan"))),
        },
    }
    metrics_path = args.results_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Training complete. Data YAML: {data_yaml}")
    print(f"Best checkpoint: {args.checkpoint_dir / 'fusion_yolo_best.pt'}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
