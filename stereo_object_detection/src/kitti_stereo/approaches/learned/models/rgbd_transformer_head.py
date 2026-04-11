from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class _ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        return self.act(x)


class _ModalityEncoder(nn.Module):
    def __init__(self, in_channels: int, d_model: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            _ConvNormAct(in_channels, 32, stride=2),
            _ResidualBlock(32),
            _ConvNormAct(32, 64, stride=2),
            _ResidualBlock(64),
            _ConvNormAct(64, 128, stride=2),
            _ResidualBlock(128),
            nn.Conv2d(128, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class RGBDTransformer3DHead(nn.Module):
    """RGB-D transformer head for 3D center/dimension/yaw regression.

    This model uses:
    1. Separate CNN encoders for RGB and depth modalities.
    2. Token-level fusion via a Transformer encoder.
    3. Uncertainty-aware multi-task loss for stable training.
    """

    def __init__(
        self,
        roi_size: int = 96,
        geom_dim: int = 8,
        d_model: int = 192,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D sinusoidal positional encoding.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.roi_size = int(roi_size)
        self.geom_dim = int(geom_dim)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)

        self.rgb_encoder = _ModalityEncoder(in_channels=3, d_model=d_model)
        self.depth_encoder = _ModalityEncoder(in_channels=1, d_model=d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.rgb_modality_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.depth_modality_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.geom_proj = nn.Sequential(
            nn.Linear(self.geom_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.pred_head = nn.Linear(128, 8)  # x,y,z, log(h,w,l), sin(yaw), cos(yaw)
        self.logvar_head = nn.Linear(128, 3)  # task uncertainty for xyz, dims, yaw

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.rgb_modality_token, mean=0.0, std=0.02)
        nn.init.normal_(self.depth_modality_token, mean=0.0, std=0.02)

    @staticmethod
    def _build_2d_sincos_pos_embed(
        height: int,
        width: int,
        d_model: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4.")

        y = torch.linspace(0.0, 1.0, steps=height, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, steps=width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        yy = yy.reshape(-1, 1)
        xx = xx.reshape(-1, 1)

        quarter = d_model // 4
        omega = torch.arange(quarter, device=device, dtype=dtype)
        omega = 1.0 / (10000 ** (omega / max(1.0, float(quarter))))

        out_x = xx * omega.unsqueeze(0)
        out_y = yy * omega.unsqueeze(0)

        pos = torch.cat([
            torch.sin(out_x),
            torch.cos(out_x),
            torch.sin(out_y),
            torch.cos(out_y),
        ], dim=1)
        return pos.unsqueeze(0)

    @staticmethod
    def _to_tokens(feature_map: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = feature_map.shape
        return feature_map.permute(0, 2, 3, 1).reshape(bsz, height * width, channels)

    def forward(self, roi_tensor: torch.Tensor, geom_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if geom_features is None:
            raise ValueError("geom_features are required.")
        if geom_features.shape[1] != self.geom_dim:
            raise ValueError(f"Expected geom_features with dim={self.geom_dim}, got {geom_features.shape[1]}.")

        rgb = roi_tensor[:, :3, :, :]
        depth = roi_tensor[:, 3:4, :, :]

        rgb_feat = self.rgb_encoder(rgb)
        depth_feat = self.depth_encoder(depth)

        rgb_tokens = self._to_tokens(rgb_feat)
        depth_tokens = self._to_tokens(depth_feat)

        _, _, h, w = rgb_feat.shape
        pos = self._build_2d_sincos_pos_embed(
            height=h,
            width=w,
            d_model=self.d_model,
            device=roi_tensor.device,
            dtype=roi_tensor.dtype,
        )

        rgb_tokens = rgb_tokens + pos + self.rgb_modality_token
        depth_tokens = depth_tokens + pos + self.depth_modality_token

        geom_token = self.geom_proj(geom_features).unsqueeze(1)
        cls_token = self.cls_token.expand(roi_tensor.shape[0], -1, -1)

        tokens = torch.cat([cls_token, geom_token, rgb_tokens, depth_tokens], dim=1)
        fused = self.transformer(tokens)
        cls_fused = fused[:, 0, :]

        features = self.fusion_head(cls_fused)
        pred = self.pred_head(features)
        logvar = torch.clamp(self.logvar_head(features), min=-3.0, max=3.0)
        return pred, logvar

    @staticmethod
    def loss_dict(pred: torch.Tensor, target: torch.Tensor, logvar: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        pred_xyz = pred[:, 0:3]
        pred_dims_log = pred[:, 3:6]
        pred_yaw = pred[:, 6:8]

        tgt_xyz = target[:, 0:3]
        tgt_dims_log = target[:, 3:6]
        tgt_yaw = target[:, 6:8]

        xyz_per = F.smooth_l1_loss(pred_xyz, tgt_xyz, beta=0.5, reduction="none").mean(dim=1)
        dims_per = F.smooth_l1_loss(pred_dims_log, tgt_dims_log, beta=0.25, reduction="none").mean(dim=1)

        pred_yaw_norm = F.normalize(pred_yaw, dim=1, eps=1e-6)
        tgt_yaw_norm = F.normalize(tgt_yaw, dim=1, eps=1e-6)
        yaw_per = 1.0 - torch.sum(pred_yaw_norm * tgt_yaw_norm, dim=1).clamp(-1.0, 1.0)

        if logvar is not None:
            l_xyz = torch.exp(-logvar[:, 0]) * xyz_per + logvar[:, 0]
            l_dims = torch.exp(-logvar[:, 1]) * dims_per + logvar[:, 1]
            l_yaw = torch.exp(-logvar[:, 2]) * yaw_per + logvar[:, 2]
            total = (2.0 * l_xyz + 1.0 * l_dims + 1.0 * l_yaw).mean()
        else:
            total = (2.0 * xyz_per + 1.0 * dims_per + 1.0 * yaw_per).mean()

        return {
            "total": total,
            "xyz": xyz_per.mean(),
            "dims": dims_per.mean(),
            "yaw": yaw_per.mean(),
        }


def decode_rgbd_transformer_output(pred_row: torch.Tensor) -> Dict[str, float]:
    x_m = float(pred_row[0].item())
    y_m = float(pred_row[1].item())
    z_m = float(max(pred_row[2].item(), 0.1))

    h_m = float(math.exp(pred_row[3].item()))
    w_m = float(math.exp(pred_row[4].item()))
    l_m = float(math.exp(pred_row[5].item()))

    sin_y = float(pred_row[6].item())
    cos_y = float(pred_row[7].item())
    norm = math.sqrt(max(1e-8, sin_y * sin_y + cos_y * cos_y))
    sin_y /= norm
    cos_y /= norm
    yaw = float(math.atan2(sin_y, cos_y))

    return {
        "x_m": x_m,
        "y_m": y_m,
        "z_m": z_m,
        "h_m": h_m,
        "w_m": w_m,
        "l_m": l_m,
        "ry": yaw,
    }
