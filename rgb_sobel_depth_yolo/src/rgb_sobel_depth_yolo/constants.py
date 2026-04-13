from __future__ import annotations

KITTI_CLASSES = [
    "Car",
    "Van",
    "Truck",
    "Pedestrian",
    "Person_sitting",
    "Cyclist",
    "Tram",
    "Misc",
]

KITTI_CLASS_TO_ID = {name: idx for idx, name in enumerate(KITTI_CLASSES)}

DEFAULT_SGBM_CONFIG = {
    "minDisparity": 0,
    "numDisparities": 128,
    "blockSize": 7,
    "P1": 8 * 3 * 7 * 7,
    "P2": 32 * 3 * 7 * 7,
    "disp12MaxDiff": 1,
    "uniquenessRatio": 10,
    "speckleWindowSize": 100,
    "speckleRange": 32,
    "preFilterCap": 63,
    "mode": 1,
}
