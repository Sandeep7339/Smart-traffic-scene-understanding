from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Detection2D:
    class_id: int
    label: str
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]


_MODEL_CACHE: Dict[str, Any] = {}


def _clip_bbox(xyxy: np.ndarray, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x1 = int(np.floor(float(xyxy[0])))
    y1 = int(np.floor(float(xyxy[1])))
    x2 = int(np.ceil(float(xyxy[2])))
    y2 = int(np.ceil(float(xyxy[3])))

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _get_yolo_model(model_name: str) -> Any:
    if model_name not in _MODEL_CACHE:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Install with:\n"
                "python3 -m pip install ultralytics"
            ) from exc
        _MODEL_CACHE[model_name] = YOLO(model_name)
    return _MODEL_CACHE[model_name]


def run_yolo_detection(
    image_bgr: np.ndarray,
    model_name: str = "yolov8n.pt",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 100,
    classes: Optional[Sequence[int]] = None,
    imgsz: int = 640,
    device: Optional[str] = None,
) -> List[Detection2D]:
    height, width = image_bgr.shape[:2]
    model = _get_yolo_model(model_name)

    predict_kwargs = {
        "source": image_bgr,
        "conf": conf_threshold,
        "iou": iou_threshold,
        "imgsz": imgsz,
        "max_det": max_detections,
        "verbose": False,
    }
    if device:
        predict_kwargs["device"] = device
    if classes is not None:
        predict_kwargs["classes"] = list(classes)

    result = model.predict(**predict_kwargs)[0]
    names = result.names

    detections: List[Detection2D] = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        xyxy = box.xyxy[0].detach().cpu().numpy()
        clipped = _clip_bbox(xyxy, width=width, height=height)
        if clipped is None:
            continue

        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        label = str(names.get(cls_id, cls_id))
        detections.append(
            Detection2D(
                class_id=cls_id,
                label=label,
                confidence=conf,
                bbox_xyxy=clipped,
            )
        )
    return detections
