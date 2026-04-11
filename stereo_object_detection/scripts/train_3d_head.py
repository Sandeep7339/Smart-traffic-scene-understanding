#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kitti_stereo.approaches.learned.dataset import KittiRoi3DDataset
from kitti_stereo.approaches.learned.models import RGBDTransformer3DHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RGB-D transformer 3D regression head on KITTI ROIs.")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "kitti-dataset")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--roi-size", type=int, default=96)
    parser.add_argument("--max-depth-m", type=float, default=80.0)
    parser.add_argument("--stereo-method", type=str, default="sgbm", choices=["sgbm", "bm"])
    parser.add_argument("--num-disparities", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--depth-cache-dir", type=Path, default=ROOT / "outputs" / "depth_cache")
    parser.add_argument("--save-path", type=Path, default=ROOT / "outputs" / "rgbd_transformer_3d_head.pt")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    return parser.parse_args()


def list_training_sample_ids(dataset_root: Path, max_frames: int | None = None) -> list[str]:
    label_dir = dataset_root / "data_object_label_2" / "training" / "label_2"
    sample_ids = sorted(p.stem for p in label_dir.glob("*.txt"))
    if max_frames is not None and max_frames > 0:
        sample_ids = sample_ids[:max_frames]
    return sample_ids


def run_epoch(model, loader, optimizer, device):
    model.train() if optimizer is not None else model.eval()
    running = {"total": 0.0, "xyz": 0.0, "dims": 0.0, "yaw": 0.0}
    n_batches = 0

    for batch in loader:
        roi = batch["roi"].to(device=device, dtype=torch.float32)
        geom = batch["geom"].to(device=device, dtype=torch.float32)
        target = batch["target"].to(device=device, dtype=torch.float32)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        pred, logvar = model(roi, geom)
        losses = RGBDTransformer3DHead.loss_dict(pred, target, logvar=logvar)

        if optimizer is not None:
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        for k in running.keys():
            running[k] += float(losses[k].detach().cpu().item())
        n_batches += 1

    if n_batches == 0:
        return {k: float("nan") for k in running}
    return {k: v / n_batches for k, v in running.items()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is required. Install with: pip install -r requirements.txt") from exc

    sample_ids = list_training_sample_ids(args.dataset_root, max_frames=args.max_frames)
    if not sample_ids:
        raise SystemExit(f"No training labels found under: {args.dataset_root}")

    dataset = KittiRoi3DDataset(
        dataset_root=args.dataset_root,
        sample_ids=sample_ids,
        cv2_module=cv2,
        roi_size=args.roi_size,
        max_depth_m=args.max_depth_m,
        stereo_method=args.stereo_method,
        num_disparities=args.num_disparities,
        block_size=args.block_size,
        depth_cache_dir=args.depth_cache_dir,
    )
    if len(dataset) < 10:
        raise SystemExit("Dataset is too small after filtering. Check labels and paths.")

    train_size = max(1, int(len(dataset) * args.train_fraction))
    val_size = len(dataset) - train_size
    if val_size == 0:
        val_size = 1
        train_size = len(dataset) - 1
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    geom_dim = int(dataset[0]["geom"].numel())
    model = RGBDTransformer3DHead(
        roi_size=args.roi_size,
        geom_dim=geom_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Training samples: {len(train_ds)} | Validation samples: {len(val_ds)} | Device: {device} | "
        f"Model: RGBDTransformer3DHead(d_model={args.d_model}, heads={args.num_heads}, layers={args.num_layers})"
    )
    for epoch in range(1, args.epochs + 1):
        train_losses = run_epoch(model, train_loader, optimizer=optimizer, device=device)
        with torch.no_grad():
            val_losses = run_epoch(model, val_loader, optimizer=None, device=device)

        print(
            f"Epoch {epoch:02d} | "
            f"train total={train_losses['total']:.4f} xyz={train_losses['xyz']:.4f} dims={train_losses['dims']:.4f} yaw={train_losses['yaw']:.4f} | "
            f"val total={val_losses['total']:.4f} xyz={val_losses['xyz']:.4f} dims={val_losses['dims']:.4f} yaw={val_losses['yaw']:.4f}"
        )

        if val_losses["total"] < best_val:
            best_val = val_losses["total"]
            ckpt = {
                "model_state_dict": model.state_dict(),
                "roi_size": args.roi_size,
                "max_depth_m": args.max_depth_m,
                "stereo_method": args.stereo_method,
                "num_disparities": args.num_disparities,
                "block_size": args.block_size,
                "best_val_total": best_val,
                "model_name": "RGBDTransformer3DHead",
                "geom_dim": geom_dim,
                "d_model": args.d_model,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
            }
            torch.save(ckpt, args.save_path)
            print(f"Saved best checkpoint: {args.save_path} (val_total={best_val:.4f})")

    print("Training completed.")


if __name__ == "__main__":
    main()
