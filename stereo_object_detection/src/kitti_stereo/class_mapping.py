from __future__ import annotations


def yolo_to_kitti_group(yolo_label: str) -> str | None:
    label = yolo_label.lower().strip()
    if label in {"car"}:
        return "Car"
    if label in {"truck"}:
        return "Truck"
    if label in {"bus"}:
        return "Tram"
    if label in {"person"}:
        return "Pedestrian"
    if label in {"bicycle", "motorcycle"}:
        return "Cyclist"
    return None


def is_detection_label_compatible(yolo_label: str, kitti_type: str) -> bool:
    mapped = yolo_to_kitti_group(yolo_label)
    if mapped is None:
        return False

    if mapped == "Car":
        return kitti_type in {"Car", "Van"}
    if mapped == "Truck":
        return kitti_type in {"Truck"}
    if mapped == "Tram":
        return kitti_type in {"Tram"}
    if mapped == "Pedestrian":
        return kitti_type in {"Pedestrian", "Person_sitting"}
    if mapped == "Cyclist":
        return kitti_type in {"Cyclist"}
    return False
