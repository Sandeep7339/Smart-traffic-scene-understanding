from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _draw_prediction(
    image_rgb: np.ndarray,
    prediction: dict[str, Any],
    class_names: list[str],
    color: tuple[int, int, int],
    title: str,
) -> np.ndarray:
    img = image_rgb.copy()
    h, w = img.shape[:2]

    for box, score, cls_id in zip(prediction.get("boxes_xyxy", []), prediction.get("scores", []), prediction.get("classes", [])):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        cls_name = class_names[int(cls_id)] if 0 <= int(cls_id) < len(class_names) else str(cls_id)
        label = f"{cls_name} {float(score):.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.putText(img, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return img


def _load_rgb(dataset_root: Path, image_id: str) -> np.ndarray:
    img_path = dataset_root / "data_object_image_2" / "training" / "image_2" / f"{image_id}.png"
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image for visualization: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def _rescale_prediction_boxes_to_original(
    prediction: dict[str, Any],
    model_img_size: int,
    orig_w: int,
    orig_h: int,
) -> dict[str, Any]:
    scaled = dict(prediction)
    boxes = prediction.get("boxes_xyxy", [])
    if not boxes:
        scaled["boxes_xyxy"] = []
        return scaled

    sx = float(orig_w) / float(model_img_size)
    sy = float(orig_h) / float(model_img_size)

    out_boxes: list[list[float]] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        out_boxes.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
    scaled["boxes_xyxy"] = out_boxes
    return scaled


def save_side_by_side_predictions(
    dataset_root: str | Path,
    image_ids: list[str],
    baseline_preds: dict[str, dict[str, Any]],
    fusion_preds: dict[str, dict[str, Any]],
    class_names: list[str],
    img_size: int,
    baseline_dir: str | Path,
    fusion_dir: str | Path,
    side_by_side_dir: str | Path,
) -> list[dict[str, str]]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    baseline_dir = Path(baseline_dir)
    fusion_dir = Path(fusion_dir)
    side_by_side_dir = Path(side_by_side_dir)

    baseline_dir.mkdir(parents=True, exist_ok=True)
    fusion_dir.mkdir(parents=True, exist_ok=True)
    side_by_side_dir.mkdir(parents=True, exist_ok=True)

    saved: list[dict[str, str]] = []

    for sid in image_ids:
        rgb = _load_rgb(dataset_root, sid)
        orig_h, orig_w = rgb.shape[:2]

        base_pred = baseline_preds.get(sid, {"boxes_xyxy": [], "scores": [], "classes": []})
        fusion_pred = fusion_preds.get(sid, {"boxes_xyxy": [], "scores": [], "classes": []})
        base_pred = _rescale_prediction_boxes_to_original(base_pred, img_size, orig_w=orig_w, orig_h=orig_h)
        fusion_pred = _rescale_prediction_boxes_to_original(fusion_pred, img_size, orig_w=orig_w, orig_h=orig_h)

        left = _draw_prediction(rgb, base_pred, class_names, color=(220, 30, 30), title="Baseline YOLOv8 (RGB)")
        right = _draw_prediction(rgb, fusion_pred, class_names, color=(30, 180, 40), title="Fusion YOLOv8 (RGB+Depth+Edge)")

        side = np.concatenate([left, right], axis=1)

        base_path = baseline_dir / f"baseline_{sid}.png"
        fusion_path = fusion_dir / f"fusion_{sid}.png"
        side_path = side_by_side_dir / f"side_by_side_{sid}.png"

        cv2.imwrite(str(base_path), cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(fusion_path), cv2.cvtColor(right, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(side_path), cv2.cvtColor(side, cv2.COLOR_RGB2BGR))

        saved.append(
            {
                "image_id": sid,
                "baseline_image": str(base_path),
                "fusion_image": str(fusion_path),
                "side_by_side": str(side_path),
            }
        )

    return saved
