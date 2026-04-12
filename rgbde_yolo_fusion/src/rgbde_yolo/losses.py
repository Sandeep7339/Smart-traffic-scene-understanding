
from typing import Any

import torch
import torch.nn.functional as F


def _wh_iou(wh1: torch.Tensor, wh2: torch.Tensor) -> torch.Tensor:
    # wh1: [N,2], wh2: [M,2]
    wh1 = wh1[:, None, :]
    wh2 = wh2[None, :, :]
    inter = torch.minimum(wh1[..., 0], wh2[..., 0]) * torch.minimum(wh1[..., 1], wh2[..., 1])
    area1 = wh1[..., 0] * wh1[..., 1]
    area2 = wh2[..., 0] * wh2[..., 1]
    union = area1 + area2 - inter + 1e-6
    return inter / union


def build_targets(
    outputs: dict[int, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    anchors: dict[int, list[tuple[int, int]]],
    num_classes: int,
    img_size: int,
) -> dict[int, dict[str, torch.Tensor]]:
    device = next(iter(outputs.values())).device
    built: dict[int, dict[str, torch.Tensor]] = {}

    for stride, pred in outputs.items():
        b, na, _, gh, gw = pred.shape
        built[stride] = {
            "obj": torch.zeros((b, na, gh, gw), device=device),
            "tx": torch.zeros((b, na, gh, gw), device=device),
            "ty": torch.zeros((b, na, gh, gw), device=device),
            "tw": torch.zeros((b, na, gh, gw), device=device),
            "th": torch.zeros((b, na, gh, gw), device=device),
            "cls": torch.zeros((b, na, gh, gw, num_classes), device=device),
            "mask": torch.zeros((b, na, gh, gw), dtype=torch.bool, device=device),
        }

    for bi, t in enumerate(targets):
        boxes = t["boxes"].to(device=device, dtype=torch.float32)
        labels = t["labels"].to(device=device, dtype=torch.long)
        if boxes.numel() == 0:
            continue

        # Normalized -> pixels
        gt_xy = boxes[:, 0:2] * img_size
        gt_wh = boxes[:, 2:4] * img_size

        for gi in range(boxes.shape[0]):
            wh = gt_wh[gi : gi + 1]
            best_stride = None
            best_anchor_idx = None
            best_anchor_iou = -1.0

            for stride, a_list in anchors.items():
                a_wh = torch.tensor(a_list, dtype=torch.float32, device=device)
                iou = _wh_iou(wh, a_wh).squeeze(0)
                v, idx = torch.max(iou, dim=0)
                if float(v.item()) > best_anchor_iou:
                    best_anchor_iou = float(v.item())
                    best_stride = stride
                    best_anchor_idx = int(idx.item())

            assert best_stride is not None
            stride = int(best_stride)
            a_idx = int(best_anchor_idx)

            out = outputs[stride]
            _, _, _, gh, gw = out.shape
            target_buf = built[stride]

            gx = gt_xy[gi, 0] / stride
            gy = gt_xy[gi, 1] / stride
            gw_gt = gt_wh[gi, 0]
            gh_gt = gt_wh[gi, 1]

            cx = int(torch.clamp(gx.floor(), min=0, max=gw - 1).item())
            cy = int(torch.clamp(gy.floor(), min=0, max=gh - 1).item())

            target_buf["obj"][bi, a_idx, cy, cx] = 1.0
            target_buf["tx"][bi, a_idx, cy, cx] = gx - cx
            target_buf["ty"][bi, a_idx, cy, cx] = gy - cy

            anchor_w, anchor_h = anchors[stride][a_idx]
            target_buf["tw"][bi, a_idx, cy, cx] = torch.log(gw_gt / (anchor_w + 1e-6) + 1e-6)
            target_buf["th"][bi, a_idx, cy, cx] = torch.log(gh_gt / (anchor_h + 1e-6) + 1e-6)
            target_buf["cls"][bi, a_idx, cy, cx, labels[gi].item()] = 1.0
            target_buf["mask"][bi, a_idx, cy, cx] = True

    return built


def yolo_fusion_loss(
    outputs: dict[int, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    anchors: dict[int, list[tuple[int, int]]],
    num_classes: int,
    img_size: int,
    lambda_box: float = 5.0,
    lambda_obj: float = 1.0,
    lambda_cls: float = 1.0,
) -> dict[str, Any]:
    built = build_targets(outputs, targets, anchors, num_classes, img_size)

    total_box = torch.tensor(0.0, device=next(iter(outputs.values())).device)
    total_obj = torch.tensor(0.0, device=next(iter(outputs.values())).device)
    total_cls = torch.tensor(0.0, device=next(iter(outputs.values())).device)

    for stride, pred in outputs.items():
        t = built[stride]

        p = pred.permute(0, 1, 3, 4, 2).contiguous()
        p_tx = p[..., 0]
        p_ty = p[..., 1]
        p_tw = p[..., 2]
        p_th = p[..., 3]
        p_obj = p[..., 4]
        p_cls = p[..., 5 : 5 + num_classes]

        mask = t["mask"]

        total_obj = total_obj + F.binary_cross_entropy_with_logits(p_obj, t["obj"], reduction="mean")

        if mask.any():
            total_box = total_box + F.binary_cross_entropy_with_logits(p_tx[mask], t["tx"][mask], reduction="mean")
            total_box = total_box + F.binary_cross_entropy_with_logits(p_ty[mask], t["ty"][mask], reduction="mean")
            total_box = total_box + F.smooth_l1_loss(p_tw[mask], t["tw"][mask], reduction="mean")
            total_box = total_box + F.smooth_l1_loss(p_th[mask], t["th"][mask], reduction="mean")
            total_cls = total_cls + F.binary_cross_entropy_with_logits(p_cls[mask], t["cls"][mask], reduction="mean")

    total = lambda_box * total_box + lambda_obj * total_obj + lambda_cls * total_cls
    return {
        "total": total,
        "box": total_box.detach(),
        "obj": total_obj.detach(),
        "cls": total_cls.detach(),
    }
