import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from .constants import KITTI_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8n checkpoint on one KITTI sample.")
    parser.add_argument("--dataset-root", type=Path, default=Path("../dataset"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-id", type=str, default="000000")
    parser.add_argument("--split", type=str, default="training", choices=["training", "testing"])
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--save-json", type=Path, default=Path("outputs/results/infer_sample.json"))
    parser.add_argument("--save-image", type=Path, default=Path("outputs/results/infer_sample.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sid = args.sample_id.zfill(6)

    image_path = args.dataset_root / "data_object_image_2" / args.split / "image_2" / f"{sid}.png"
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    model = YOLO(str(args.checkpoint))
    pred = model.predict(
        source=str(image_path),
        imgsz=args.img_size,
        conf=args.conf_thres,
        device=args.device if args.device else None,
        verbose=False,
    )[0]

    names = pred.names
    detections = []
    if pred.boxes is not None:
        for box in pred.boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            conf = float(box.conf.item())
            cls_id = int(box.cls.item())
            label = str(names.get(cls_id, cls_id))
            if cls_id < len(KITTI_CLASSES):
                label = KITTI_CLASSES[cls_id]
            detections.append(
                {
                    "bbox_xyxy": [float(v) for v in xyxy],
                    "score": conf,
                    "class_id": cls_id,
                    "label": label,
                }
            )

    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(
        json.dumps(
            {
                "sample_id": sid,
                "image_path": str(image_path),
                "num_detections": len(detections),
                "detections": detections,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    vis = pred.plot()
    args.save_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.save_image), vis)

    print(f"Saved JSON: {args.save_json}")
    print(f"Saved image: {args.save_image}")


if __name__ == "__main__":
    main()
