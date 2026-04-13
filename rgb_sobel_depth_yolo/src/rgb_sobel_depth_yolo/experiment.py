from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable

from .constants import KITTI_CLASSES
from .data import (
    KITTIMultiModalDataset,
    build_dataloader,
    build_train_val_split,
    summarize_label_quality,
)
from .metrics import evaluate_model
from .modeling import build_detection_model, load_checkpoint, save_checkpoint
from .utils import ensure_dir, move_batch_to_device, resolve_device, safe_pct_improvement, save_csv, save_json, set_seed
from .viz import save_side_by_side_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline YOLOv8 vs RGB+Depth+Sobel fusion YOLOv8 on KITTI")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--weights", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--output-root", type=Path, default=Path("rgb_sobel_depth_yolo/outputs"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.6)
    parser.add_argument("--vis-count", type=int, default=20)
    parser.add_argument("--qualitative-count", type=int, default=20)
    parser.add_argument(
        "--depth-edge-init",
        type=str,
        default="rgb_mean",
        choices=["rgb_mean", "random"],
        help="Initialization of extra channels in first conv for fusion model.",
    )
    return parser.parse_args()


def _train_one_epoch(
    model,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 10.0,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    running_parts = np.zeros(3, dtype=np.float64)
    n_batches = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        bs = int(batch["img"].shape[0])

        optimizer.zero_grad(set_to_none=True)
        preds = model(batch["img"])
        raw_loss, loss_items = model.loss(batch, preds)

        loss = raw_loss.sum() / max(1, bs)
        loss.backward()

        if grad_clip > 0:
            clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        running_loss += float(loss.item())
        running_parts += loss_items.detach().cpu().numpy()
        n_batches += 1

        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    if n_batches == 0:
        return {"loss": 0.0, "box": 0.0, "cls": 0.0, "dfl": 0.0}

    part_avg = running_parts / n_batches
    return {
        "loss": running_loss / n_batches,
        "box": float(part_avg[0]) if len(part_avg) > 0 else 0.0,
        "cls": float(part_avg[1]) if len(part_avg) > 1 else 0.0,
        "dfl": float(part_avg[2]) if len(part_avg) > 2 else 0.0,
    }


def _plot_training_curves(
    baseline_history: list[dict[str, Any]],
    fusion_history: list[dict[str, Any]],
    baseline_eval: dict[str, Any],
    fusion_eval: dict[str, Any],
    plots_dir: Path,
) -> dict[str, str]:
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs_b = [row["epoch"] for row in baseline_history]
    epochs_f = [row["epoch"] for row in fusion_history]

    loss_plot = plots_dir / "loss_curves_baseline_vs_fusion.png"
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_b, [row["train_loss"] for row in baseline_history], label="Baseline train loss", color="#cc2f2f")
    plt.plot(epochs_b, [row["val_loss"] for row in baseline_history], label="Baseline val loss", color="#cc2f2f", linestyle="--")
    plt.plot(epochs_f, [row["train_loss"] for row in fusion_history], label="Fusion train loss", color="#1e8a3a")
    plt.plot(epochs_f, [row["val_loss"] for row in fusion_history], label="Fusion val loss", color="#1e8a3a", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_plot, dpi=180)
    plt.close()

    map_plot = plots_dir / "map_curves_baseline_vs_fusion.png"
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_b, [row["map50"] for row in baseline_history], label="Baseline mAP@50", color="#cc2f2f")
    plt.plot(epochs_b, [row["map50_95"] for row in baseline_history], label="Baseline mAP@50:95", color="#cc2f2f", linestyle="--")
    plt.plot(epochs_f, [row["map50"] for row in fusion_history], label="Fusion mAP@50", color="#1e8a3a")
    plt.plot(epochs_f, [row["map50_95"] for row in fusion_history], label="Fusion mAP@50:95", color="#1e8a3a", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("mAP over Epochs")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(map_plot, dpi=180)
    plt.close()

    pr_plot = plots_dir / "precision_recall_baseline_vs_fusion.png"
    plt.figure(figsize=(8, 6))
    b_curve = baseline_eval.get("pr_curve", {})
    f_curve = fusion_eval.get("pr_curve", {})

    if b_curve.get("recall") and b_curve.get("precision"):
        plt.plot(b_curve["recall"], b_curve["precision"], label="Baseline PR", color="#cc2f2f", linewidth=2)
    if f_curve.get("recall") and f_curve.get("precision"):
        plt.plot(f_curve["recall"], f_curve["precision"], label="Fusion PR", color="#1e8a3a", linewidth=2)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pr_plot, dpi=180)
    plt.close()

    return {
        "loss_curve": str(loss_plot),
        "map_curve": str(map_plot),
        "pr_curve": str(pr_plot),
    }


def _prediction_map(predictions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["image_id"]: row for row in predictions}


def _build_qualitative_report(
    baseline_per_image: list[dict[str, Any]],
    fusion_per_image: list[dict[str, Any]],
    side_by_side_assets: list[dict[str, str]],
    out_dir: Path,
    top_k: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    base_map = {row["image_id"]: row for row in baseline_per_image}
    fusion_map = {row["image_id"]: row for row in fusion_per_image}
    vis_map = {row["image_id"]: row for row in side_by_side_assets}

    improved: list[dict[str, Any]] = []
    baseline_failures: list[dict[str, Any]] = []
    difficult_improved: list[dict[str, Any]] = []

    common_ids = sorted(set(base_map.keys()) & set(fusion_map.keys()))
    for sid in common_ids:
        b = base_map[sid]
        f = fusion_map[sid]

        tp_gain = int(f["tp50"]) - int(b["tp50"])
        fn_reduction = int(b["fn50"]) - int(f["fn50"])
        score = tp_gain + fn_reduction

        row = {
            "image_id": sid,
            "tp_gain": tp_gain,
            "fn_reduction": fn_reduction,
            "baseline": b,
            "fusion": f,
            "side_by_side": vis_map.get(sid, {}).get("side_by_side", ""),
        }

        if score > 0:
            improved.append(row)
            if bool(f.get("has_occlusion", False)) or bool(f.get("has_small_object", False)):
                difficult_improved.append(row)

        if int(b["gt_count"]) > 0 and int(b["tp50"]) == 0 and int(f["tp50"]) > 0:
            baseline_failures.append(row)

    improved = sorted(improved, key=lambda x: (x["tp_gain"] + x["fn_reduction"]), reverse=True)[:top_k]
    baseline_failures = sorted(baseline_failures, key=lambda x: x["tp_gain"], reverse=True)[:top_k]
    difficult_improved = sorted(difficult_improved, key=lambda x: (x["tp_gain"] + x["fn_reduction"]), reverse=True)[:top_k]

    # Copy top qualitative visuals to a compact folder for quick review.
    copied_assets: list[str] = []
    for idx, row in enumerate(improved[:top_k]):
        vis_path = row.get("side_by_side")
        if not vis_path:
            continue
        src = Path(vis_path)
        if not src.exists():
            continue
        dst = out_dir / f"improved_{idx:02d}_{row['image_id']}.png"
        shutil.copy2(src, dst)
        copied_assets.append(str(dst))

    report = {
        "improved_examples": improved,
        "baseline_failures_fixed": baseline_failures,
        "difficult_improved_examples": difficult_improved,
        "copied_assets": copied_assets,
    }

    save_json(out_dir / "qualitative_analysis.json", report)

    md_lines = [
        "# Qualitative Analysis",
        "",
        "## Improved Detections (Fusion > Baseline)",
        "",
    ]
    if improved:
        for row in improved:
            md_lines.append(
                f"- {row['image_id']}: tp_gain={row['tp_gain']}, fn_reduction={row['fn_reduction']}, side_by_side={row['side_by_side']}"
            )
    else:
        md_lines.append("- No improved examples found.")

    md_lines += ["", "## Baseline Failures Fixed by Fusion", ""]
    if baseline_failures:
        for row in baseline_failures:
            md_lines.append(f"- {row['image_id']}: baseline tp50=0, fusion tp50={row['fusion']['tp50']}")
    else:
        md_lines.append("- No baseline-only failures fixed in selected set.")

    md_lines += ["", "## Difficult Cases Improved (Occlusion/Small Objects)", ""]
    if difficult_improved:
        for row in difficult_improved:
            diff = row["fusion"]
            md_lines.append(
                f"- {row['image_id']}: occlusion={diff['has_occlusion']}, small_object={diff['has_small_object']}"
            )
    else:
        md_lines.append("- No difficult-case improvements found.")

    (out_dir / "qualitative_analysis.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return report


def _train_model(
    model_name: str,
    model,
    train_loader,
    val_loader,
    args: argparse.Namespace,
    device: torch.device,
    output_root: Path,
) -> dict[str, Any]:
    checkpoints_dir = ensure_dir(output_root / "checkpoints")
    results_dir = ensure_dir(output_root / "results")
    plots_dir = ensure_dir(output_root / "plots")
    predictions_dir = ensure_dir(output_root / "predictions")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=max(1e-6, args.lr * 0.05))

    history: list[dict[str, Any]] = []
    best_map = -1.0
    best_ckpt = checkpoints_dir / f"{model_name}_best.pt"
    last_ckpt = checkpoints_dir / f"{model_name}_last.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_stats = _train_one_epoch(model, train_loader, optimizer, device)

        val_stats = evaluate_model(
            model,
            val_loader,
            device=device,
            num_classes=len(KITTI_CLASSES),
            class_names=KITTI_CLASSES,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            plot_curves=False,
            plot_dir=plots_dir,
            prefix=f"{model_name}_",
        )

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_stats["loss"]),
            "train_box_loss": float(train_stats["box"]),
            "train_cls_loss": float(train_stats["cls"]),
            "train_dfl_loss": float(train_stats["dfl"]),
            "val_loss": float(val_stats["loss"]),
            "val_box_loss": float(val_stats["loss_parts"]["box"]),
            "val_cls_loss": float(val_stats["loss_parts"]["cls"]),
            "val_dfl_loss": float(val_stats["loss_parts"]["dfl"]),
            "map50": float(val_stats["map50"]),
            "map50_95": float(val_stats["map50_95"]),
            "precision": float(val_stats["precision"]),
            "recall": float(val_stats["recall"]),
            "epoch_seconds": float(time.time() - t0),
        }
        history.append(row)

        save_checkpoint(
            path=last_ckpt,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_map50_95=max(best_map, float(val_stats["map50_95"])),
            history=history,
            extra={"model_name": model_name},
        )

        if float(val_stats["map50_95"]) > best_map:
            best_map = float(val_stats["map50_95"])
            save_checkpoint(
                path=best_ckpt,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_map50_95=best_map,
                history=history,
                extra={"model_name": model_name},
            )

        scheduler.step()
        print(
            f"[{model_name}] epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"mAP50={row['map50']:.4f} mAP50:95={row['map50_95']:.4f} "
            f"P={row['precision']:.4f} R={row['recall']:.4f}"
        )

    # Final evaluation on best model, with PR plotting enabled.
    load_checkpoint(best_ckpt, model=model, map_location=device)
    final_eval = evaluate_model(
        model,
        val_loader,
        device=device,
        num_classes=len(KITTI_CLASSES),
        class_names=KITTI_CLASSES,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        plot_curves=True,
        plot_dir=plots_dir,
        prefix=f"{model_name}_",
    )

    final_metrics = {
        "model_name": model_name,
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
        "final": {
            "loss": float(final_eval["loss"]),
            "map50": float(final_eval["map50"]),
            "map50_95": float(final_eval["map50_95"]),
            "precision": float(final_eval["precision"]),
            "recall": float(final_eval["recall"]),
        },
    }

    save_json(results_dir / f"{model_name}_history.json", history)
    save_csv(results_dir / f"{model_name}_history.csv", history)
    save_json(results_dir / f"{model_name}_final_metrics.json", final_metrics)
    save_json(predictions_dir / f"{model_name}_predictions.json", final_eval["predictions"])
    save_json(results_dir / f"{model_name}_per_image_metrics.json", final_eval["per_image"])

    return {
        "history": history,
        "final_eval": final_eval,
        "final_metrics": final_metrics,
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
    }


def _write_comparison_summary(out_path_json: Path, out_path_md: Path, baseline: dict[str, Any], fusion: dict[str, Any]) -> dict[str, Any]:
    b = baseline["final_metrics"]["final"]
    f = fusion["final_metrics"]["final"]

    comparison = {
        "baseline": b,
        "fusion": f,
        "improvement_percent": {
            "map50": safe_pct_improvement(f["map50"], b["map50"]),
            "map50_95": safe_pct_improvement(f["map50_95"], b["map50_95"]),
            "precision": safe_pct_improvement(f["precision"], b["precision"]),
            "recall": safe_pct_improvement(f["recall"], b["recall"]),
        },
    }

    save_json(out_path_json, comparison)

    md_lines = [
        "# Baseline vs Modified Fusion YOLOv8",
        "",
        "| Metric | Baseline | Modified Fusion | Improvement % |",
        "|---|---:|---:|---:|",
        f"| mAP@50 | {b['map50']:.4f} | {f['map50']:.4f} | {comparison['improvement_percent']['map50']:.2f}% |",
        f"| mAP@50:95 | {b['map50_95']:.4f} | {f['map50_95']:.4f} | {comparison['improvement_percent']['map50_95']:.2f}% |",
        f"| Precision | {b['precision']:.4f} | {f['precision']:.4f} | {comparison['improvement_percent']['precision']:.2f}% |",
        f"| Recall | {b['recall']:.4f} | {f['recall']:.4f} | {comparison['improvement_percent']['recall']:.2f}% |",
        "",
        "## Checkpoints",
        "",
        f"- Baseline best: {baseline['best_checkpoint']}",
        f"- Fusion best: {fusion['best_checkpoint']}",
    ]
    out_path_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return comparison


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_root = ensure_dir(args.output_root)
    checkpoints_dir = ensure_dir(output_root / "checkpoints")
    results_dir = ensure_dir(output_root / "results")
    plots_dir = ensure_dir(output_root / "plots")
    visuals_root = ensure_dir(output_root / "visualizations")
    qualitative_dir = ensure_dir(output_root / "qualitative")

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    split = build_train_val_split(
        dataset_root=args.dataset_root,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_samples=args.max_samples,
    )

    split_json = {
        "train_ids": split.train_ids,
        "val_ids": split.val_ids,
        "seed": split.seed,
        "val_fraction": split.val_fraction,
    }
    save_json(results_dir / "split_info.json", split_json)

    label_quality = summarize_label_quality(args.dataset_root, split.train_ids + split.val_ids)
    save_json(results_dir / "label_quality.json", label_quality)
    print(
        "Label quality | "
        f"samples={label_quality['num_samples']} objects={label_quality['total_objects']} "
        f"empty_labels={label_quality['empty_label_files']}"
    )

    cache_root = ensure_dir(output_root / "cache")

    # Baseline RGB dataloaders
    train_rgb_ds = KITTIMultiModalDataset(
        dataset_root=args.dataset_root,
        sample_ids=split.train_ids,
        img_size=args.img_size,
        use_depth=False,
        use_edge=False,
        cache_dir=None,
    )
    val_rgb_ds = KITTIMultiModalDataset(
        dataset_root=args.dataset_root,
        sample_ids=split.val_ids,
        img_size=args.img_size,
        use_depth=False,
        use_edge=False,
        cache_dir=None,
    )

    train_rgb_loader = build_dataloader(
        train_rgb_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_rgb_loader = build_dataloader(
        val_rgb_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # Fusion (RGB + Depth + Edge) dataloaders
    train_fusion_ds = KITTIMultiModalDataset(
        dataset_root=args.dataset_root,
        sample_ids=split.train_ids,
        img_size=args.img_size,
        use_depth=True,
        use_edge=True,
        cache_dir=cache_root / "train_depth_edge",
    )
    val_fusion_ds = KITTIMultiModalDataset(
        dataset_root=args.dataset_root,
        sample_ids=split.val_ids,
        img_size=args.img_size,
        use_depth=True,
        use_edge=True,
        cache_dir=cache_root / "val_depth_edge",
    )

    train_fusion_loader = build_dataloader(
        train_fusion_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_fusion_loader = build_dataloader(
        val_fusion_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # Build models.
    baseline_model = build_detection_model(
        pretrained_weights=args.weights,
        num_classes=len(KITTI_CLASSES),
        in_channels=3,
        device=device,
    )
    fusion_model = build_detection_model(
        pretrained_weights=args.weights,
        num_classes=len(KITTI_CLASSES),
        in_channels=5,
        device=device,
        depth_edge_init=args.depth_edge_init,
    )

    # Train both models with identical hyperparameters.
    print("\nTraining baseline YOLOv8 (RGB)...")
    baseline_run = _train_model(
        model_name="baseline_yolov8_rgb",
        model=baseline_model,
        train_loader=train_rgb_loader,
        val_loader=val_rgb_loader,
        args=args,
        device=device,
        output_root=output_root,
    )

    print("\nTraining modified YOLOv8 fusion (RGB + Depth + Sobel Edge)...")
    fusion_run = _train_model(
        model_name="fusion_yolov8_rgb_depth_edge",
        model=fusion_model,
        train_loader=train_fusion_loader,
        val_loader=val_fusion_loader,
        args=args,
        device=device,
        output_root=output_root,
    )

    # Plot curves.
    plot_paths = _plot_training_curves(
        baseline_history=baseline_run["history"],
        fusion_history=fusion_run["history"],
        baseline_eval=baseline_run["final_eval"],
        fusion_eval=fusion_run["final_eval"],
        plots_dir=plots_dir,
    )
    save_json(results_dir / "plot_paths.json", plot_paths)

    # Quantitative summary.
    comparison = _write_comparison_summary(
        out_path_json=results_dir / "baseline_vs_fusion.json",
        out_path_md=results_dir / "baseline_vs_fusion.md",
        baseline=baseline_run,
        fusion=fusion_run,
    )

    # Visualization for same validation images.
    vis_ids = split.val_ids[: max(0, min(args.vis_count, len(split.val_ids)))]
    baseline_pred_map = _prediction_map(baseline_run["final_eval"]["predictions"])
    fusion_pred_map = _prediction_map(fusion_run["final_eval"]["predictions"])

    side_assets = save_side_by_side_predictions(
        dataset_root=args.dataset_root,
        image_ids=vis_ids,
        baseline_preds=baseline_pred_map,
        fusion_preds=fusion_pred_map,
        class_names=KITTI_CLASSES,
        img_size=args.img_size,
        baseline_dir=visuals_root / "baseline",
        fusion_dir=visuals_root / "modified",
        side_by_side_dir=visuals_root / "side_by_side",
    )
    save_json(results_dir / "visualization_assets.json", side_assets)

    # Qualitative analysis.
    qualitative = _build_qualitative_report(
        baseline_per_image=baseline_run["final_eval"]["per_image"],
        fusion_per_image=fusion_run["final_eval"]["per_image"],
        side_by_side_assets=side_assets,
        out_dir=qualitative_dir,
        top_k=args.qualitative_count,
    )

    final_report = {
        "device": str(device),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "weights": str(Path(args.weights).resolve()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "split": split_json,
        "label_quality": label_quality,
        "baseline": baseline_run["final_metrics"],
        "fusion": fusion_run["final_metrics"],
        "comparison": comparison,
        "plot_paths": plot_paths,
        "num_visualizations": len(side_assets),
        "qualitative": {
            "improved_examples": len(qualitative.get("improved_examples", [])),
            "baseline_failures_fixed": len(qualitative.get("baseline_failures_fixed", [])),
            "difficult_improved_examples": len(qualitative.get("difficult_improved_examples", [])),
        },
    }
    save_json(results_dir / "experiment_report.json", final_report)

    print("\nComparison summary (Baseline vs Modified Fusion):")
    print(
        json.dumps(
            {
                "mAP@50": {
                    "baseline": comparison["baseline"]["map50"],
                    "fusion": comparison["fusion"]["map50"],
                    "improvement_%": comparison["improvement_percent"]["map50"],
                },
                "mAP@50:95": {
                    "baseline": comparison["baseline"]["map50_95"],
                    "fusion": comparison["fusion"]["map50_95"],
                    "improvement_%": comparison["improvement_percent"]["map50_95"],
                },
                "Precision": {
                    "baseline": comparison["baseline"]["precision"],
                    "fusion": comparison["fusion"]["precision"],
                    "improvement_%": comparison["improvement_percent"]["precision"],
                },
                "Recall": {
                    "baseline": comparison["baseline"]["recall"],
                    "fusion": comparison["fusion"]["recall"],
                    "improvement_%": comparison["improvement_percent"]["recall"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
