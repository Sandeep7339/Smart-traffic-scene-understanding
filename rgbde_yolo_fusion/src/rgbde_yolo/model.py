
from pathlib import Path

import torch
import torch.nn as nn


def _adapt_first_conv_to_channels(det_model: nn.Module, in_channels: int) -> None:
    first = det_model.model[0]
    old_conv = first.conv
    if old_conv.in_channels == in_channels:
        return

    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    with torch.no_grad():
        if in_channels >= old_conv.in_channels:
            new_conv.weight[:, : old_conv.in_channels] = old_conv.weight
            extra = in_channels - old_conv.in_channels
            if extra > 0:
                mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, old_conv.in_channels :] = mean_w.repeat(1, extra, 1, 1)
        else:
            new_conv.weight.copy_(old_conv.weight[:, :in_channels])

    first.conv = new_conv


def load_yolov8n_model(weights_path: str | Path, num_classes: int, in_channels: int = 3):
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    _adapt_first_conv_to_channels(model.model, in_channels=in_channels)

    if int(getattr(model.model.model[-1], "nc", num_classes)) != int(num_classes):
        model.model.model[-1].nc = int(num_classes)
        model.model.names = {i: str(i) for i in range(int(num_classes))}

    return model


class RGBDepthEdgeYOLO(nn.Module):
    """Compatibility wrapper: now delegates to pretrained YOLOv8n."""

    def __init__(self, num_classes: int, weights_path: str | Path = "yolov8n.pt", in_channels: int = 3) -> None:
        super().__init__()
        yolo = load_yolov8n_model(weights_path=weights_path, num_classes=num_classes, in_channels=in_channels)
        self.yolo = yolo
        self.model = yolo.model
        self.num_classes = int(num_classes)

    def forward(self, x: torch.Tensor):
        return self.model(x)
