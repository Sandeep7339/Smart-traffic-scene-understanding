from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kitti_stereo.box3d import draw_3d_boxes, estimate_3d_boxes_from_detections


def estimate_3d_boxes(
    detections,
    depth_map_m,
    fx_px,
    fy_px,
    cx_px,
    cy_px,
    p2_matrix,
    min_depth_m=0.1,
    max_depth_m=80.0,
    inner_ratio=0.65,
    min_points=50,
):
    return estimate_3d_boxes_from_detections(
        detections=detections,
        depth_map_m=depth_map_m,
        fx_px=fx_px,
        fy_px=fy_px,
        cx_px=cx_px,
        cy_px=cy_px,
        p2_matrix=p2_matrix,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        inner_ratio=inner_ratio,
        min_points=min_points,
    )


def draw_projected_3d_boxes(image_bgr, boxes_3d, cv2_module):
    return draw_3d_boxes(image_bgr=image_bgr, boxes_3d=boxes_3d, cv2_module=cv2_module)
