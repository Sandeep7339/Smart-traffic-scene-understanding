
import random
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(-1)
    x1 = x - 0.5 * w
    y1 = y - 0.5 * h
    x2 = x + 0.5 * w
    y2 = y + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    a = box1[:, None, :]
    b = box2[None, :, :]

    inter_x1 = torch.maximum(a[..., 0], b[..., 0])
    inter_y1 = torch.maximum(a[..., 1], b[..., 1])
    inter_x2 = torch.minimum(a[..., 2], b[..., 2])
    inter_y2 = torch.minimum(a[..., 3], b[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area2 = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    union = area1 + area2 - inter + 1e-6
    return inter / union


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> list[int]:
    if boxes.numel() == 0:
        return []
    keep: list[int] = []
    order = torch.argsort(scores, descending=True)
    while order.numel() > 0:
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        iou = box_iou(boxes[i : i + 1], boxes[rest]).squeeze(0)
        order = rest[iou <= iou_threshold]
    return keep


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: list[float]
    score: float
    class_id: int


def decode_outputs(
    outputs: dict[int, torch.Tensor],
    anchors: dict[int, list[tuple[int, int]]],
    conf_threshold: float,
    num_classes: int,
    img_size: int,
) -> list[list[Detection]]:
    batch_size = next(iter(outputs.values())).shape[0]
    device = next(iter(outputs.values())).device
    all_batch: list[list[Detection]] = [[] for _ in range(batch_size)]

    for stride, pred in outputs.items():
        _, na, _, gh, gw = pred.shape
        anchor_tensor = torch.tensor(anchors[stride], dtype=pred.dtype, device=device)

        p = pred.permute(0, 1, 3, 4, 2).contiguous()  # B,A,H,W,C
        xy = torch.sigmoid(p[..., 0:2])
        wh = torch.exp(p[..., 2:4]).clamp(max=16.0)
        obj = torch.sigmoid(p[..., 4])
        cls = torch.sigmoid(p[..., 5 : 5 + num_classes])

        grid_y, grid_x = torch.meshgrid(
            torch.arange(gh, device=device),
            torch.arange(gw, device=device),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, gh, gw, 2).to(pred.dtype)

        xy_abs = (xy + grid) * stride
        wh_abs = wh * anchor_tensor.view(1, na, 1, 1, 2)
        boxes_xyxy = torch.cat([xy_abs - 0.5 * wh_abs, xy_abs + 0.5 * wh_abs], dim=-1)

        cls_score, cls_id = torch.max(cls, dim=-1)
        final_score = obj * cls_score
        mask = final_score >= conf_threshold

        for b in range(batch_size):
            idx = torch.where(mask[b])
            for a, y, x in zip(idx[0], idx[1], idx[2]):
                score = float(final_score[b, a, y, x].item())
                if score < conf_threshold:
                    continue
                cid = int(cls_id[b, a, y, x].item())
                box = boxes_xyxy[b, a, y, x]
                box = box.clamp(min=0.0, max=float(img_size))
                all_batch[b].append(
                    Detection(
                        bbox_xyxy=[float(v.item()) for v in box],
                        score=score,
                        class_id=cid,
                    )
                )

    # Per-class NMS.
    final_batch: list[list[Detection]] = []
    for detections in all_batch:
        kept: list[Detection] = []
        for cls_id in range(num_classes):
            cls_det = [d for d in detections if d.class_id == cls_id]
            if not cls_det:
                continue
            boxes = torch.tensor([d.bbox_xyxy for d in cls_det], dtype=torch.float32)
            scores = torch.tensor([d.score for d in cls_det], dtype=torch.float32)
            keep_idx = nms(boxes, scores, iou_threshold=0.5)
            kept.extend([cls_det[i] for i in keep_idx])
        final_batch.append(sorted(kept, key=lambda d: d.score, reverse=True))

    return final_batch
