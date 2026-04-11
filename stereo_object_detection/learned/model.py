import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TinyEncoder(nn.Module):
    def __init__(self, in_channels, base_channels=32, out_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, base_channels, stride=2),
            ConvBlock(base_channels, base_channels, stride=1),
            ConvBlock(base_channels, base_channels * 2, stride=2),
            ConvBlock(base_channels * 2, base_channels * 2, stride=1),
            ConvBlock(base_channels * 2, out_channels, stride=2),
        )

    def forward(self, x):
        return self.net(x)


class StereoFusion3DNet(nn.Module):
    """Lightweight RGB-depth dual encoder + transformer fusion + MLP regressor.

    Output order: (dx, dy, dz, log(w), log(h), log(l), sin(theta), cos(theta))
    """

    def __init__(self, geom_dim=8, d_model=128, nhead=4, num_layers=2, patch_size=96):
        super().__init__()
        self.patch_size = int(patch_size)
        self.geom_dim = int(geom_dim)

        self.rgb_encoder = TinyEncoder(in_channels=3, base_channels=32, out_channels=128)
        self.depth_encoder = TinyEncoder(in_channels=1, base_channels=16, out_channels=128)

        self.rgb_proj = nn.Conv2d(128, d_model, kernel_size=1)
        self.depth_proj = nn.Conv2d(128, d_model, kernel_size=1)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.geom_proj = nn.Sequential(
            nn.Linear(geom_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 8),
        )

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, patch_tensor, geom_features):
        rgb = patch_tensor[:, :3, :, :]
        depth = patch_tensor[:, 3:4, :, :]

        rgb_map = self.rgb_proj(self.rgb_encoder(rgb))
        depth_map = self.depth_proj(self.depth_encoder(depth))

        b, c, h, w = rgb_map.shape
        rgb_tokens = rgb_map.permute(0, 2, 3, 1).reshape(b, h * w, c)
        depth_tokens = depth_map.permute(0, 2, 3, 1).reshape(b, h * w, c)
        geom_token = self.geom_proj(geom_features).unsqueeze(1)
        cls = self.cls_token.expand(b, -1, -1)

        tokens = torch.cat([cls, geom_token, rgb_tokens, depth_tokens], dim=1)
        fused = self.transformer(tokens)
        return self.head(fused[:, 0, :])

    @staticmethod
    def loss_dict(pred, target):
        # xyz residuals
        loss_xyz = F.smooth_l1_loss(pred[:, 0:3], target[:, 0:3], beta=0.5)
        # log dims
        loss_dims = F.smooth_l1_loss(pred[:, 3:6], target[:, 3:6], beta=0.25)
        # yaw via sin/cos consistency
        pred_sc = F.normalize(pred[:, 6:8], dim=1, eps=1e-6)
        tgt_sc = F.normalize(target[:, 6:8], dim=1, eps=1e-6)
        loss_theta = F.mse_loss(pred_sc, tgt_sc)

        total = 2.0 * loss_xyz + 1.0 * loss_dims + 1.0 * loss_theta
        return {
            "total": total,
            "xyz": loss_xyz,
            "dims": loss_dims,
            "theta": loss_theta,
        }
