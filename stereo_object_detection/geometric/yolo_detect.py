from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kitti_stereo.detection import run_yolo_detection


def detect_2d_boxes(
    image_bgr,
    model_name="yolov8n.pt",
    conf_threshold=0.25,
    iou_threshold=0.45,
    max_detections=100,
    imgsz=640,
    device=None,
):
    return run_yolo_detection(
        image_bgr=image_bgr,
        model_name=model_name,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
        imgsz=imgsz,
        device=device,
    )
