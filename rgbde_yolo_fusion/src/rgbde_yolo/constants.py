KITTI_CLASSES = [
    "Car",
    "Van",
    "Truck",
    "Pedestrian",
    "Person_sitting",
    "Cyclist",
    "Tram",
]

KITTI_CLASS_TO_ID = {name: idx for idx, name in enumerate(KITTI_CLASSES)}

# Anchors are in pixels for strides [8, 16, 32].
DEFAULT_ANCHORS = {
    8: [(10, 13), (16, 30), (33, 23)],
    16: [(30, 61), (62, 45), (59, 119)],
    32: [(116, 90), (156, 198), (373, 326)],
}
