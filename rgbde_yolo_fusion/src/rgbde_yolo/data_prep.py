import random
import shutil
from pathlib import Path

from .constants import KITTI_CLASSES, KITTI_CLASS_TO_ID


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _kitti_to_yolo_line(line: str, img_w: int, img_h: int) -> str | None:
    parts = line.strip().split()
    if len(parts) < 15:
        return None
    cls_name = parts[0]
    if cls_name not in KITTI_CLASS_TO_ID:
        return None

    x1 = float(parts[4])
    y1 = float(parts[5])
    x2 = float(parts[6])
    y2 = float(parts[7])

    bw = max(1e-6, x2 - x1)
    bh = max(1e-6, y2 - y1)
    cx = x1 + 0.5 * bw
    cy = y1 + 0.5 * bh

    xc = cx / float(img_w)
    yc = cy / float(img_h)
    wn = bw / float(img_w)
    hn = bh / float(img_h)

    cls_id = KITTI_CLASS_TO_ID[cls_name]
    return f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"


def prepare_kitti_yolo_dataset(
    dataset_root: str | Path,
    out_dir: str | Path,
    val_fraction: float = 0.15,
    seed: int = 42,
    max_samples: int = 0,
) -> Path:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to prepare YOLO dataset.") from exc

    dataset_root = Path(dataset_root).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()

    label_dir = dataset_root / "data_object_label_2" / "training" / "label_2"
    img_dir = dataset_root / "data_object_image_2" / "training" / "image_2"

    sample_ids = sorted(p.stem for p in label_dir.glob("*.txt"))
    if max_samples > 0:
        sample_ids = sample_ids[:max_samples]

    rng = random.Random(seed)
    rng.shuffle(sample_ids)
    val_count = max(1, int(len(sample_ids) * val_fraction))
    val_ids = set(sample_ids[:val_count])
    train_ids = set(sample_ids[val_count:])

    train_img_dir = out_dir / "images" / "train"
    val_img_dir = out_dir / "images" / "val"
    train_lbl_dir = out_dir / "labels" / "train"
    val_lbl_dir = out_dir / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    def process_sid(sid: str, split: str) -> None:
        src_img = img_dir / f"{sid}.png"
        src_lbl = label_dir / f"{sid}.txt"
        if not src_img.exists() or not src_lbl.exists():
            return

        if split == "train":
            dst_img = train_img_dir / f"{sid}.png"
            dst_lbl = train_lbl_dir / f"{sid}.txt"
        else:
            dst_img = val_img_dir / f"{sid}.png"
            dst_lbl = val_lbl_dir / f"{sid}.txt"

        _link_or_copy(src_img, dst_img)

        img = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
        if img is None:
            return
        h, w = img.shape[:2]

        yolo_lines = []
        for line in src_lbl.read_text(encoding="utf-8").splitlines():
            out = _kitti_to_yolo_line(line, img_w=w, img_h=h)
            if out is not None:
                yolo_lines.append(out)
        dst_lbl.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

    for sid in sorted(train_ids):
        process_sid(sid, split="train")
    for sid in sorted(val_ids):
        process_sid(sid, split="val")

    yaml_path = out_dir / "kitti_rgb_yolov8.yaml"
    yaml_text = "\n".join(
        [
            f"path: {out_dir}",
            "train: images/train",
            "val: images/val",
            "names:",
        ]
        + [f"  {i}: {name}" for i, name in enumerate(KITTI_CLASSES)]
    )
    yaml_path.write_text(yaml_text + "\n", encoding="utf-8")
    return yaml_path
