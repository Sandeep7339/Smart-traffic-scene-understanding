
import numpy as np


def bbox_iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)


def _ap_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_map50(
    predictions: list[list[dict]],
    ground_truths: list[list[dict]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> dict:
    ap_per_class: list[float] = []

    for cls_id in range(num_classes):
        cls_preds = []
        npos = 0
        gt_by_img = {}

        for img_idx, gts in enumerate(ground_truths):
            gt_cls = [g for g in gts if int(g["class_id"]) == cls_id]
            gt_by_img[img_idx] = {
                "boxes": [g["bbox_xyxy"] for g in gt_cls],
                "matched": [False] * len(gt_cls),
            }
            npos += len(gt_cls)

        for img_idx, preds in enumerate(predictions):
            for p in preds:
                if int(p["class_id"]) != cls_id:
                    continue
                cls_preds.append((img_idx, float(p["score"]), p["bbox_xyxy"]))

        if npos == 0:
            ap_per_class.append(float("nan"))
            continue

        cls_preds.sort(key=lambda x: x[1], reverse=True)
        tp = np.zeros(len(cls_preds), dtype=np.float32)
        fp = np.zeros(len(cls_preds), dtype=np.float32)

        for i, (img_idx, _score, box) in enumerate(cls_preds):
            gt_info = gt_by_img[img_idx]
            gt_boxes = gt_info["boxes"]
            if not gt_boxes:
                fp[i] = 1.0
                continue

            ious = np.asarray([bbox_iou_xyxy(box, gt_box) for gt_box in gt_boxes], dtype=np.float32)
            best_idx = int(np.argmax(ious))
            best_iou = float(ious[best_idx])

            if best_iou >= iou_threshold and not gt_info["matched"][best_idx]:
                tp[i] = 1.0
                gt_info["matched"][best_idx] = True
            else:
                fp[i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
        recall = tp_cum / float(max(npos, 1))
        ap = _ap_from_pr(precision, recall)
        ap_per_class.append(ap)

    valid = [v for v in ap_per_class if np.isfinite(v)]
    map50 = float(np.mean(valid)) if valid else float("nan")
    return {
        "map50": map50,
        "ap50_per_class": ap_per_class,
    }
