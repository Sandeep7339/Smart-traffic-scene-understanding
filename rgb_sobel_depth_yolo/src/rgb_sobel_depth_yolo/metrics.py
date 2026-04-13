from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.utils.nms import non_max_suppression

from .utils import move_batch_to_device, xywhn_to_xyxy


def _match_predictions(
    pred_classes: torch.Tensor,
    true_classes: torch.Tensor,
    iou: torch.Tensor,
    iouv: torch.Tensor,
) -> torch.Tensor:
    if pred_classes.numel() == 0:
        return torch.zeros((0, iouv.numel()), dtype=torch.bool, device=pred_classes.device)

    correct = np.zeros((pred_classes.shape[0], iouv.shape[0]), dtype=bool)
    if true_classes.numel() == 0:
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    correct_class = true_classes[:, None] == pred_classes[None, :]
    iou = iou * correct_class
    iou_np = iou.detach().cpu().numpy()

    for i, threshold in enumerate(iouv.detach().cpu().tolist()):
        matches = np.nonzero(iou_np >= threshold)
        matches = np.array(matches).T
        if matches.shape[0] == 0:
            continue

        if matches.shape[0] > 1:
            matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

        correct[matches[:, 1].astype(int), i] = True

    return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)


def evaluate_model(
    model,
    dataloader,
    device: torch.device,
    num_classes: int,
    class_names: list[str],
    conf_thres: float = 0.001,
    iou_thres: float = 0.6,
    max_det: int = 300,
    plot_curves: bool = False,
    plot_dir: str | Path | None = None,
    prefix: str = "",
) -> dict[str, Any]:
    model.eval()

    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    stats: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    predictions: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []

    loss_sum = 0.0
    loss_parts_sum = torch.zeros(3, dtype=torch.float32, device=device)
    seen_batches = 0

    names_map = {i: class_names[i] for i in range(len(class_names))}

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            bs = int(batch["img"].shape[0])
            h = int(batch["img"].shape[2])
            w = int(batch["img"].shape[3])

            preds = model(batch["img"])
            loss, loss_items = model.loss(batch, preds)
            loss_sum += float(loss.sum().item() / max(1, bs))
            loss_parts_sum += loss_items.detach()
            seen_batches += 1

            pred_tensor = preds[0] if isinstance(preds, (tuple, list)) else preds
            pred_nms = non_max_suppression(
                pred_tensor,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                nc=num_classes,
                max_det=max_det,
            )

            batch_idx = batch["batch_idx"]
            for si in range(bs):
                pred = pred_nms[si]
                pred = pred if pred is not None else torch.zeros((0, 6), device=device)

                gt_mask = batch_idx == si
                gt_cls = batch["cls"][gt_mask].view(-1).to(torch.int64)
                gt_boxes_xyxy = xywhn_to_xyxy(batch["bboxes"][gt_mask], width=w, height=h)

                if pred.shape[0] > 0:
                    pred_boxes = pred[:, :4]
                    pred_conf = pred[:, 4]
                    pred_cls = pred[:, 5].to(torch.int64)
                else:
                    pred_boxes = torch.zeros((0, 4), device=device)
                    pred_conf = torch.zeros((0,), device=device)
                    pred_cls = torch.zeros((0,), dtype=torch.int64, device=device)

                correct = torch.zeros((pred.shape[0], iouv.numel()), dtype=torch.bool, device=device)
                if gt_cls.numel() > 0 and pred.shape[0] > 0:
                    iou = box_iou(gt_boxes_xyxy, pred_boxes)
                    correct = _match_predictions(pred_cls, gt_cls, iou, iouv)

                stats.append(
                    (
                        correct.detach().cpu().numpy(),
                        pred_conf.detach().cpu().numpy(),
                        pred_cls.detach().cpu().numpy(),
                        gt_cls.detach().cpu().numpy(),
                    )
                )

                tp50 = int(correct[:, 0].sum().item()) if correct.numel() > 0 else 0
                pred_count = int(pred.shape[0])
                gt_count = int(gt_cls.shape[0])
                fp50 = max(0, pred_count - tp50)
                fn50 = max(0, gt_count - tp50)

                sid = batch["image_ids"][si]
                diff = batch["difficulties"][si]
                per_image.append(
                    {
                        "image_id": sid,
                        "gt_count": gt_count,
                        "pred_count": pred_count,
                        "tp50": tp50,
                        "fp50": fp50,
                        "fn50": fn50,
                        "has_occlusion": bool(diff.get("has_occlusion", False)),
                        "has_small_object": bool(diff.get("has_small_object", False)),
                    }
                )

                scale = torch.tensor([w, h, w, h], device=device, dtype=torch.float32)
                boxes_abs = pred_boxes.detach().cpu().tolist()
                boxes_norm = (pred_boxes / scale).detach().cpu().tolist() if pred_boxes.numel() > 0 else []
                scores = pred_conf.detach().cpu().tolist()
                class_ids = [int(x) for x in pred_cls.detach().cpu().tolist()]
                class_labels = [class_names[c] if 0 <= c < len(class_names) else str(c) for c in class_ids]

                predictions.append(
                    {
                        "image_id": sid,
                        "boxes_xyxy": boxes_abs,
                        "boxes_xyxy_norm": boxes_norm,
                        "scores": scores,
                        "classes": class_ids,
                        "class_names": class_labels,
                    }
                )

    loss_avg = loss_sum / max(1, seen_batches)
    loss_parts = (loss_parts_sum / max(1, seen_batches)).detach().cpu().tolist()

    if not stats:
        return {
            "loss": loss_avg,
            "loss_parts": {"box": 0.0, "cls": 0.0, "dfl": 0.0},
            "map50": 0.0,
            "map50_95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "per_class": {},
            "pr_curve": {"recall": [], "precision": []},
            "predictions": predictions,
            "per_image": per_image,
        }

    tp, conf, pred_cls, target_cls = [np.concatenate(x, 0) for x in zip(*stats)]

    if target_cls.size == 0:
        metrics = {
            "map50": 0.0,
            "map50_95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "per_class": {},
            "pr_curve": {"recall": [], "precision": []},
        }
    elif conf.size == 0:
        unique_classes = np.unique(target_cls).astype(int)
        per_class = {
            class_names[int(c)]: {
                "class_id": int(c),
                "map50": 0.0,
                "map50_95": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }
            for c in unique_classes
            if 0 <= int(c) < len(class_names)
        }
        metrics = {
            "map50": 0.0,
            "map50_95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "per_class": per_class,
            "pr_curve": {"recall": [0.0, 1.0], "precision": [1.0, 0.0]},
        }
    else:
        plot_path = Path(plot_dir) if plot_dir is not None else Path(".")
        out = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            plot=plot_curves,
            save_dir=plot_path,
            names=names_map,
            prefix=prefix,
        )
        tp_c, fp_c, p, r, f1, ap, unique_classes, p_curve, r_curve, f1_curve, x, prec_values = out

        map50 = float(ap[:, 0].mean()) if ap.size else 0.0
        map50_95 = float(ap.mean()) if ap.size else 0.0
        precision = float(p.mean()) if p.size else 0.0
        recall = float(r.mean()) if r.size else 0.0

        per_class = {}
        for idx, cls_id in enumerate(unique_classes.astype(int)):
            if 0 <= cls_id < len(class_names):
                per_class[class_names[cls_id]] = {
                    "class_id": int(cls_id),
                    "map50": float(ap[idx, 0]) if ap.ndim == 2 else 0.0,
                    "map50_95": float(ap[idx].mean()) if ap.ndim == 2 else 0.0,
                    "precision": float(p[idx]) if idx < len(p) else 0.0,
                    "recall": float(r[idx]) if idx < len(r) else 0.0,
                }

        mean_precision_curve = p_curve.mean(0).tolist() if hasattr(p_curve, "ndim") and p_curve.ndim == 2 else []
        recall_axis = x.tolist() if hasattr(x, "tolist") else []

        metrics = {
            "map50": map50,
            "map50_95": map50_95,
            "precision": precision,
            "recall": recall,
            "per_class": per_class,
            "pr_curve": {
                "recall": recall_axis,
                "precision": mean_precision_curve,
            },
        }

    return {
        "loss": loss_avg,
        "loss_parts": {
            "box": float(loss_parts[0]) if len(loss_parts) > 0 else 0.0,
            "cls": float(loss_parts[1]) if len(loss_parts) > 1 else 0.0,
            "dfl": float(loss_parts[2]) if len(loss_parts) > 2 else 0.0,
        },
        **metrics,
        "predictions": predictions,
        "per_image": per_image,
    }
