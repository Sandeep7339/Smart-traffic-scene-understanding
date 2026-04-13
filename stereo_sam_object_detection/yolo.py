
from typing import Optional, Sequence

import numpy as np

from utils import ensure_legacy_src_on_path

ensure_legacy_src_on_path()

from kitti_stereo import Detection2D, run_yolo_detection  # noqa: E402


def detect_objects(
    image_bgr: np.ndarray,
    model_name: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 50,
    classes: Optional[Sequence[int]] = None,
    imgsz: int = 640,
    device: Optional[str] = None,
) -> list[Detection2D]:
    return run_yolo_detection(
        image_bgr=image_bgr,
        model_name=model_name,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
        classes=classes,
        imgsz=imgsz,
        device=device,
    )
