# RGB + Sobel + Depth Fusion YOLOv8 (KITTI)

This directory contains a complete experiment pipeline that compares:

1. `baseline_yolov8_rgb`: standard YOLOv8 with RGB input (3 channels)
2. `fusion_yolov8_rgb_depth_edge`: modified YOLOv8 with early fusion input `[R, G, B, Depth, SobelEdge]` (5 channels)

The goal is to test whether adding stereo depth and edge cues improves KITTI object detection over a strong RGB baseline.

## 1) What This Directory Implements End-to-End

- KITTI parsing and split creation
- Multi-modal preprocessing:
  - Stereo depth proxy from left-right KITTI images (`StereoSGBM`)
  - Sobel edge magnitude from grayscale RGB
  - Per-map normalization to `[0, 1]`
- Early-fusion tensor construction (`3ch` vs `5ch`)
- YOLOv8 model construction and training without Ultralytics trainer wrapper
- First-conv adaptation from `3 -> 5` input channels for fusion model
- Validation metrics (mAP, precision, recall), per-image diagnostics, and prediction dumps
- Curve plotting, side-by-side visualizations, and qualitative analysis reports
- Baseline vs fusion comparison summary with percent improvements

## 2) High-Level Architecture

### 2.1 Experiment Flow

```text
KITTI dataset
  |- image_2 (left RGB)
  |- image_3 (right RGB)
  |- label_2 (KITTI labels)
       |
       v
split builder (seeded train/val split)
       |
       +--> Baseline dataloader: RGB only (3ch)
       |
       +--> Fusion dataloader: RGB + depth + edge (5ch)
                  |- depth from StereoSGBM(left,right)
                  |- sobel edge from left RGB
                  |- cached as .npz per image
       |
       v
YOLOv8n baseline (ch=3)         YOLOv8n fusion (ch=5)
                                   |- pretrained transfer
                                   |- first conv extra channels initialized
       |
       v
same training schedule + hyperparameters
       |
       v
validation + evaluation + artifacts
  |- metrics json/csv
  |- checkpoints
  |- prediction json
  |- plots
  |- visualizations
  |- qualitative report
```

### 2.2 Model Design

- Base detector: Ultralytics YOLOv8 detection model (`yolov8n.pt` by default).
- Baseline model input channels: `3`.
- Fusion model input channels: `5`.
- For fusion model, the first convolution is adapted:
  - RGB weights (`channels 0..2`) are copied from pretrained model.
  - Extra channels (`depth`, `edge`) are initialized using either:
    - `rgb_mean` (default): mean of RGB kernel weights + small noise
    - `random`: small random normal initialization

This is implemented in `src/rgb_sobel_depth_yolo/modeling.py`.

## 3) Dataset, Labels, and Split

### 3.1 Expected KITTI Layout

Under `dataset/` (or `--dataset-root`), the code expects:

```text
data_object_image_2/training/image_2/*.png   # left camera
data_object_image_3/training/image_3/*.png   # right camera
data_object_label_2/training/label_2/*.txt   # KITTI labels
```

### 3.2 Classes

From `constants.py`:

- `Car`
- `Van`
- `Truck`
- `Pedestrian`
- `Person_sitting`
- `Cyclist`
- `Tram`
- `Misc`

### 3.3 Split Used in Current Outputs

From `outputs/results/split_info.json`:

- Total labeled samples: `7481`
- Train samples: `6359`
- Validation samples: `1122`
- `val_fraction`: `0.15`
- `seed`: `42`

From `outputs/results/label_quality.json`:

- Total objects: `40570`
- Empty label files: `0`

## 4) Method Details

### 4.1 Depth Map Computation

In `data.py::compute_depth_map`:

1. Convert left and right BGR images to grayscale.
2. Compute disparity with `cv2.StereoSGBM_create` using:
   - `numDisparities=128`, `blockSize=7`, `uniquenessRatio=10`, etc.
3. Invalidate non-positive disparity.
4. Convert to relative depth proxy with inverse disparity (`1/disparity`).
5. Clip to 99th percentile of valid values.
6. Normalize to `[0,1]`.

Note: this is **relative depth**, not absolute metric depth (no calibrated intrinsics used here).

### 4.2 Sobel Edge Computation

In `data.py::compute_sobel_edge`:

1. Convert RGB to grayscale float `[0,1]`.
2. Compute Sobel x and y gradients (`ksize=3`).
3. Edge magnitude: `sqrt(gx^2 + gy^2)`.
4. Normalize to `[0,1]`.

### 4.3 Fusion Input Construction

For each sample (after resize to `img_size x img_size`):

- Baseline tensor: `[R,G,B]`
- Fusion tensor: `[R,G,B,Depth,Edge]`

Both are stored as `float32`, CHW format, and fed to YOLO model.

### 4.4 Caching Strategy

Fusion depth/edge maps are cached to compressed `.npz` files:

- `outputs/cache/train_depth_edge/` (one file per train image)
- `outputs/cache/val_depth_edge/` (one file per val image)

Current cache counts:

- train cache files: `6359`
- val cache files: `1122`

Current cache disk usage:

- train cache: `5.4G`
- val cache: `964M`

## 5) Training Methodology

Training loop is implemented in `src/rgb_sobel_depth_yolo/experiment.py`.

- Optimizer: `AdamW`
- Scheduler: `CosineAnnealingLR`
- Gradient clipping: `max_norm=10.0`
- Epochs: `20`
- Batch size: `8`
- Image size: `640`
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Device: `auto` (resolved to `cuda:0` in current run)

Both models are trained with the **same split and same hyperparameters**.

## 6) Evaluation Methodology and Metrics

Evaluation is in `src/rgb_sobel_depth_yolo/metrics.py`.

### 6.1 Inference and Matching

- NMS settings:
  - `conf_thres = 0.001`
  - `iou_thres = 0.6`
  - `max_det = 300`
- AP is computed using Ultralytics `ap_per_class` over IoU thresholds `0.50:0.05:0.95`.

### 6.2 Reported Metrics

- `mAP@50`: mean AP at IoU 0.50
- `mAP@50:95`: COCO-style mean AP averaged over IoU 0.50 to 0.95
- `Precision`, `Recall`: from AP computation outputs
- `Loss`: mean validation loss

### 6.3 Per-image Diagnostic Metrics

Saved per image in `*_per_image_metrics.json`:

- `gt_count`, `pred_count`
- `tp50`, `fp50`, `fn50` (IoU=0.5 matching)
- difficulty tags:
  - `has_occlusion`
  - `has_small_object`

### 6.4 Improvement % Formula

Used in `safe_pct_improvement`:

```text
improvement_% = 100 * (fusion - baseline) / abs(baseline)
```

## 7) Current Results (From This Directory)

Source: `outputs/results/baseline_vs_fusion.json`

| Metric | Baseline RGB | Fusion RGB+Depth+Edge | Absolute Delta (Fusion-Baseline) | Relative Delta |
|---|---:|---:|---:|---:|
| mAP@50 | 0.8137 | 0.7940 | -0.0198 | -2.43% |
| mAP@50:95 | 0.5685 | 0.5539 | -0.0146 | -2.57% |
| Precision | 0.7890 | 0.7866 | -0.0024 | -0.30% |
| Recall | 0.7754 | 0.7801 | +0.0046 | +0.60% |
| Val Loss | 2.4129 | 2.4143 | +0.0014 | n/a |

### 7.1 Best Epochs

From history files:

- Baseline best `mAP@50:95` at epoch `16`: `0.5685`
- Fusion best `mAP@50:95` at epoch `18`: `0.5539`

### 7.2 Training Time Notes

- Baseline avg epoch: `85.97s`
- Fusion avg epoch: `144.57s`
- Fusion epoch-1: `1074.76s` (heavy first-pass depth/edge cache generation)
- Fusion epochs 2-20 avg: `95.61s`

Interpretation: fusion preprocessing introduces a large initial overhead, but after caching, per-epoch cost becomes much closer to baseline.

## 8) Visualizations and Qualitative Comparison

### 8.1 Visualization Outputs

Current run generated `20` side-by-side examples. Representative samples:

| Sample ID | Side-by-Side Comparison |
|---|---|
| `000006` | ![Side by side 000006](outputs/visualizations/side_by_side/side_by_side_000006.png) |
| `000022` | ![Side by side 000022](outputs/visualizations/side_by_side/side_by_side_000022.png) |
| `000037` | ![Side by side 000037](outputs/visualizations/side_by_side/side_by_side_000037.png) |
| `000060` | ![Side by side 000060](outputs/visualizations/side_by_side/side_by_side_000060.png) |
| `000085` | ![Side by side 000085](outputs/visualizations/side_by_side/side_by_side_000085.png) |
| `000096` | ![Side by side 000096](outputs/visualizations/side_by_side/side_by_side_000096.png) |

You can find the full generated set in `outputs/visualizations/`.

### 8.2 Qualitative Analysis

`outputs/qualitative/qualitative_analysis.json` and `.md` include:

- `improved_examples`: `20`
- `baseline_failures_fixed`: `0`
- `difficult_improved_examples`: `20`

Top gain examples include image IDs such as `004796`, `005663`, `001119`.

Important note: many top improved examples do not have side-by-side image paths in the qualitative report because visualization assets are only generated for the first `vis_count` validation IDs.

## 9) Complete Output Artifact Guide

### 9.1 `outputs/checkpoints/`

- `baseline_yolov8_rgb_best.pt`
- `baseline_yolov8_rgb_last.pt`
- `fusion_yolov8_rgb_depth_edge_best.pt`
- `fusion_yolov8_rgb_depth_edge_last.pt`

### 9.2 `outputs/results/`

- `split_info.json`: train/val IDs and split params
- `label_quality.json`: object and empty-label stats
- `baseline_yolov8_rgb_history.json/csv`: epoch-wise training + validation logs
- `fusion_yolov8_rgb_depth_edge_history.json/csv`: same for fusion
- `*_final_metrics.json`: final summary metrics per model
- `*_per_image_metrics.json`: per-image TP/FP/FN diagnostics
- `baseline_vs_fusion.json/md`: quantitative comparison and improvement percentages
- `plot_paths.json`: paths to generated summary plots
- `visualization_assets.json`: mapping of image IDs to saved visualization files
- `visualization_threshold.json`: optional threshold metadata (`vis_conf_threshold: 0.1`)
- `experiment_report.json`: full top-level report with config + summary

### 9.3 `outputs/plots/`

Cross-model summary curves:

![Loss Curves](outputs/plots/loss_curves_baseline_vs_fusion.png)
![mAP Curves](outputs/plots/map_curves_baseline_vs_fusion.png)
![Precision Recall Curves](outputs/plots/precision_recall_baseline_vs_fusion.png)

Per-model PR/F1 curves:

![Baseline PR Curve](outputs/plots/baseline_yolov8_rgb_PR_curve.png)
![Fusion PR Curve](outputs/plots/fusion_yolov8_rgb_depth_edge_PR_curve.png)
![Baseline F1 Curve](outputs/plots/baseline_yolov8_rgb_F1_curve.png)
![Fusion F1 Curve](outputs/plots/fusion_yolov8_rgb_depth_edge_F1_curve.png)

### 9.4 `outputs/predictions/`

- `baseline_yolov8_rgb_predictions.json`
- `fusion_yolov8_rgb_depth_edge_predictions.json`

Each file has `1122` entries (one per validation image), with predicted boxes, scores, class IDs, and class names.

### 9.5 `outputs/cache/`

- Cached depth+edge maps for fusion training/validation
- One `.npz` per sample (`depth`, `edge` arrays)

### 9.6 `outputs/qualitative/`

- `qualitative_analysis.json`
- `qualitative_analysis.md`

## 10) Source Code Map

```text
rgb_sobel_depth_yolo/
├── scripts/run_experiment.py                # entrypoint
├── configs/experiment.yaml                  # parameter template (not auto-loaded by script)
├── src/rgb_sobel_depth_yolo/
│   ├── constants.py                         # classes + StereoSGBM defaults
│   ├── data.py                              # KITTI parsing, depth/edge, dataset, dataloaders
│   ├── modeling.py                          # YOLO build + first-conv adaptation + ckpt io
│   ├── metrics.py                           # evaluation + AP + per-image diagnostics
│   ├── viz.py                               # baseline/fusion/side-by-side visual outputs
│   ├── experiment.py                        # training loop, comparisons, reporting
│   └── utils.py                             # io, seed, device, helper math
├── outputs/                                 # generated artifacts
└── requirements.txt
```

## 11) How To Run

From repository root:

```bash
python3 -m pip install -r rgb_sobel_depth_yolo/requirements.txt
python3 rgb_sobel_depth_yolo/scripts/run_experiment.py \
  --dataset-root ./dataset \
  --weights ./yolov8n.pt \
  --epochs 20 \
  --batch-size 8 \
  --img-size 640 \
  --device auto
```

Useful optional flags:

- `--val-fraction 0.15`
- `--seed 42`
- `--max-samples 0` (0 means use all)
- `--conf-thres 0.001`
- `--iou-thres 0.6`
- `--vis-count 20`
- `--qualitative-count 20`
- `--depth-edge-init rgb_mean|random`

## 12) Reproducibility and Notes

- Seed is set for Python, NumPy, and Torch.
- Fusion and baseline are run under identical data split and optimizer settings.
- The first fusion epoch can be much slower due to uncached depth/edge computation.
- `configs/experiment.yaml` mirrors defaults but is not currently parsed automatically by the script; CLI args are the active config source.

## 13) Practical Interpretation of Current Run

For this specific run, RGB baseline remains better on global AP metrics (`mAP@50`, `mAP@50:95`) while fusion slightly improves recall. This suggests the added modalities may help recover some misses in difficult scenes, but overall ranking quality still needs tuning (depth quality, channel initialization strategy, and training schedule are likely leverage points).
