from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def resolve_device(device: str) -> torch.device:
    d = (device or "auto").strip().lower()
    if d == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if d.startswith("cuda") and torch.cuda.is_available():
        return torch.device(d)
    if d.isdigit() and torch.cuda.is_available():
        return torch.device(f"cuda:{d}")
    return torch.device("cpu")


def xyxy_to_xywhn(boxes_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    if boxes_xyxy.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]

    cx = ((x1 + x2) * 0.5) / float(width)
    cy = ((y1 + y2) * 0.5) / float(height)
    bw = (x2 - x1) / float(width)
    bh = (y2 - y1) / float(height)
    return np.stack([cx, cy, bw, bh], axis=1).astype(np.float32)


def xywhn_to_xyxy(boxes_xywhn: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if boxes_xywhn.numel() == 0:
        return boxes_xywhn.new_zeros((0, 4))
    cx = boxes_xywhn[:, 0] * width
    cy = boxes_xywhn[:, 1] * height
    bw = boxes_xywhn[:, 2] * width
    bh = boxes_xywhn[:, 3] * height
    x1 = cx - bw * 0.5
    y1 = cy - bh * 0.5
    x2 = cx + bw * 0.5
    y2 = cy + bh * 0.5
    return torch.stack([x1, y1, x2, y2], dim=1)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=device.type == "cuda")
        else:
            moved[key] = value
    return moved


def safe_pct_improvement(new_value: float, baseline_value: float) -> float:
    denom = abs(baseline_value) if abs(baseline_value) > 1e-12 else 1.0
    return 100.0 * (new_value - baseline_value) / denom
