from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from ultralytics.cfg import DEFAULT_CFG
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from .constants import KITTI_CLASSES


def _adapt_first_conv_for_fusion(
    fusion_model: DetectionModel,
    pretrained_model: DetectionModel,
    init_mode: str = "rgb_mean",
    noise_scale: float = 0.01,
) -> None:
    with torch.no_grad():
        conv_fusion = fusion_model.model[0].conv
        conv_pre = pretrained_model.model[0].conv

        if conv_pre.weight.shape[1] != 3:
            raise RuntimeError("Expected pretrained YOLO first conv to have 3 input channels.")
        if conv_fusion.weight.shape[1] < 5:
            raise RuntimeError("Fusion model must have at least 5 input channels.")

        conv_fusion.weight[:, :3].copy_(conv_pre.weight)

        mean_rgb = conv_pre.weight.mean(dim=1, keepdim=True)
        std_rgb = float(conv_pre.weight.std().item())
        eps = max(1e-6, noise_scale * std_rgb)

        for c in range(3, conv_fusion.weight.shape[1]):
            if init_mode == "random":
                conv_fusion.weight[:, c : c + 1].normal_(mean=0.0, std=eps)
            else:
                conv_fusion.weight[:, c : c + 1].copy_(mean_rgb + eps * torch.randn_like(mean_rgb))


def build_detection_model(
    pretrained_weights: str | Path,
    num_classes: int,
    in_channels: int,
    device: torch.device,
    depth_edge_init: str = "rgb_mean",
) -> DetectionModel:
    pretrained_weights = Path(pretrained_weights).expanduser().resolve()

    # Load official YOLOv8n checkpoint and create a new model with custom nc/ch.
    pretrained = YOLO(str(pretrained_weights)).model
    model = DetectionModel(cfg=pretrained.yaml, ch=in_channels, nc=num_classes, verbose=False)
    model.load(pretrained, verbose=False)

    if in_channels > 3:
        _adapt_first_conv_for_fusion(model, pretrained, init_mode=depth_edge_init)

    # Required for loss construction when training without Ultralytics trainer wrapper.
    model.args = DEFAULT_CFG
    model.names = {idx: name for idx, name in enumerate(KITTI_CLASSES[:num_classes])}
    model.to(device)
    return model


def save_checkpoint(
    path: str | Path,
    model: DetectionModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_map50_95: float,
    history: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> None:
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_map50_95": float(best_map50_95),
        "history": history,
        "model_yaml": model.yaml,
        "names": getattr(model, "names", {}),
    }
    if extra:
        payload.update(extra)

    torch.save(payload, str(ckpt_path))


def load_checkpoint(path: str | Path, model: DetectionModel, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    ckpt = torch.load(str(path), map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    names = ckpt.get("names")
    if names is not None:
        model.names = names
    return ckpt
