# RGB + Sobel + Depth Early-Fusion YOLOv8 (KITTI)

This module compares:

1. Baseline YOLOv8 (`RGB`, 3 channels)
2. Modified YOLOv8 early fusion (`RGB + Depth + Edge`, 5 channels)

## What It Implements

- Stereo depth from KITTI left/right images (`StereoSGBM`) and normalization to `[0, 1]`
- Sobel edge magnitude from grayscale RGB and normalization to `[0, 1]`
- 5-channel input construction: `[R, G, B, Depth, Edge]`
- YOLOv8 first-conv adaptation from `3 -> 5` channels with pretrained weight transfer
- Same training setup/hyperparameters for baseline and modified models
- Epoch logging (loss, mAP, precision, recall) to JSON and CSV
- Final metric comparison + percent improvements
- Training plots (loss, mAP, PR)
- Prediction dumps (`image_id, boxes, scores, classes`)
- Baseline/modified/side-by-side visualization outputs
- Qualitative analysis artifacts for improved/failure/difficult cases

## Directory Layout

```text
rgb_sobel_depth_yolo/
├── scripts/
│   └── run_experiment.py
├── src/rgb_sobel_depth_yolo/
│   ├── __init__.py
│   ├── constants.py
│   ├── utils.py
│   ├── data.py
│   ├── modeling.py
│   ├── metrics.py
│   ├── viz.py
│   └── experiment.py
├── outputs/
│   ├── checkpoints/
│   ├── results/
│   ├── plots/
│   ├── predictions/
│   ├── visualizations/
│   └── qualitative/
└── requirements.txt
```

## Run

From repo root:

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

## Key Outputs

- `rgb_sobel_depth_yolo/outputs/checkpoints/`
  - `baseline_yolov8_rgb_best.pt`, `baseline_yolov8_rgb_last.pt`
  - `fusion_yolov8_rgb_depth_edge_best.pt`, `fusion_yolov8_rgb_depth_edge_last.pt`
- `rgb_sobel_depth_yolo/outputs/results/`
  - per-epoch logs for both models (`*.json`, `*.csv`)
  - `baseline_vs_fusion.json`, `baseline_vs_fusion.md`
  - `experiment_report.json`
- `rgb_sobel_depth_yolo/outputs/plots/`
  - loss curves, mAP curves, precision-recall curves
- `rgb_sobel_depth_yolo/outputs/predictions/`
  - structured prediction files for both models
- `rgb_sobel_depth_yolo/outputs/visualizations/`
  - baseline predictions, modified predictions, side-by-side comparisons
- `rgb_sobel_depth_yolo/outputs/qualitative/`
  - improved examples, baseline failures fixed, difficult-case highlights

## Notes

- The two models are trained with identical train/val split and hyperparameters.
- KITTI labels are parsed directly and converted to YOLO training targets internally.
- Depth and edge maps are aligned with RGB via identical resizing in preprocessing.
