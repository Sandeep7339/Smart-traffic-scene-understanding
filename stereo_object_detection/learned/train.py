#!/usr/bin/env python3
import argparse
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned.dataset import StereoYoloCropDataset
from learned.model import StereoFusion3DNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train learned 3D box fusion model (CNN + Transformer head).")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--max-depth-m", type=float, default=80.0)
    parser.add_argument("--stereo-method", type=str, default="sgbm", choices=["sgbm", "bm"])
    parser.add_argument("--num-disparities", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=7)

    parser.add_argument("--yolo-model", type=str, default=str(ROOT / "yolov8n.pt"))
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument("--max-frames", type=int, default=300)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)

    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-path", type=Path, default=ROOT / "outputs" / "learned_fusion_3d.pt")
    return parser.parse_args()


def list_training_sample_ids(dataset_root, max_frames=None):
    label_dir = Path(dataset_root) / "data_object_label_2" / "training" / "label_2"
    ids = sorted(p.stem for p in label_dir.glob("*.txt"))
    if max_frames is not None and max_frames > 0:
        ids = ids[: int(max_frames)]
    return ids


def run_epoch(model, loader, optimizer, device):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    agg = {"total": 0.0, "xyz": 0.0, "dims": 0.0, "theta": 0.0}
    steps = 0

    for batch in loader:
        patch = batch["patch"].to(device=device, dtype=torch.float32)
        geom = batch["geom"].to(device=device, dtype=torch.float32)
        target = batch["target"].to(device=device, dtype=torch.float32)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        pred = model(patch, geom)
        losses = StereoFusion3DNet.loss_dict(pred, target)

        if is_train:
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        for k in agg:
            agg[k] += float(losses[k].detach().cpu().item())
        steps += 1

    if steps == 0:
        return {k: float("nan") for k in agg}
    return {k: v / steps for k, v in agg.items()}


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    import cv2

    sample_ids = list_training_sample_ids(args.dataset_root, max_frames=args.max_frames)
    if not sample_ids:
        raise SystemExit("No training label files found.")

    dataset = StereoYoloCropDataset(
        dataset_root=args.dataset_root,
        sample_ids=sample_ids,
        cv2_module=cv2,
        patch_size=args.patch_size,
        max_depth_m=args.max_depth_m,
        stereo_method=args.stereo_method,
        num_disparities=args.num_disparities,
        block_size=args.block_size,
        yolo_model=args.yolo_model,
        match_iou=args.match_iou,
    )
    if len(dataset) < 10:
        raise SystemExit("Dataset too small after YOLO-GT matching. Try more frames or lower match IoU.")

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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StereoFusion3DNet(
        geom_dim=8,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        patch_size=args.patch_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    print(f"train={len(train_ds)} val={len(val_ds)} device={device} matched_samples={len(dataset)}")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer=optimizer, device=device)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, optimizer=None, device=device)

        print(
            f"epoch {epoch:02d} | "
            f"train total={train_loss['total']:.4f} xyz={train_loss['xyz']:.4f} dims={train_loss['dims']:.4f} th={train_loss['theta']:.4f} | "
            f"val total={val_loss['total']:.4f} xyz={val_loss['xyz']:.4f} dims={val_loss['dims']:.4f} th={val_loss['theta']:.4f}"
        )

        if val_loss["total"] < best_val:
            best_val = val_loss["total"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "patch_size": args.patch_size,
                    "geom_dim": 8,
                    "d_model": args.d_model,
                    "nhead": args.nhead,
                    "num_layers": args.num_layers,
                    "max_depth_m": args.max_depth_m,
                    "best_val": best_val,
                },
                args.save_path,
            )
            print(f"saved checkpoint: {args.save_path}")

    print("training complete")


if __name__ == "__main__":
    main()
