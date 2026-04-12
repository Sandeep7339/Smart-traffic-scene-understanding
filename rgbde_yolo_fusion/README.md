# RGB + Depth + Edge YOLO-Style Fusion (2D Detection)

This directory is fully isolated from the rest of the repository and implements a new detector with the exact pipeline:

1. Build 5-channel input: RGB (3) + normalized depth (1) + Sobel edge map (1)
2. Multi-branch feature extraction:
- Branch A: RGB CNN
- Branch B: Depth CNN
- Branch C: Edge CNN
3. Feature fusion:
- Concatenation + channel attention
4. Modified YOLO-style backbone
5. FPN/PAN neck
6. YOLO-style detection heads at strides 8/16/32
7. 2D boxes + class scores output

## Directory Layout

```text
rgbde_yolo_fusion/
├── configs/
│   └── train_rgbde_yolo.yaml
├── outputs/
│   ├── checkpoints/
│   ├── comparisons/
│   └── results/
├── scripts/
│   ├── train_rgbde_yolo.py
│   ├── eval_rgbde_yolo.py
│   ├── compare_baseline_vs_fusion.py
│   └── run_infer_sample.py
├── src/
│   └── rgbde_yolo/
│       ├── __init__.py
│       ├── constants.py
│       ├── dataset.py
│       ├── model.py
│       ├── losses.py
│       ├── metrics.py
│       ├── train.py
│       ├── evaluate.py
│       ├── compare_baseline.py
│       ├── infer.py
│       └── utils.py
├── requirements.txt
└── README.md
```

## Reused Existing Work (Read-Only)

The new code imports existing stereo/depth/KITTI helpers from:
- stereo_object_detection/src/kitti_stereo/depth_pipeline.py
- stereo_object_detection/src/kitti_stereo/kitti_data.py
- stereo_object_detection/src/kitti_stereo/kitti_labels.py
- stereo_object_detection/src/kitti_stereo/class_mapping.py

No existing files are modified.

## Install

From workspace root:

```bash
python3 -m pip install -r rgbde_yolo_fusion/requirements.txt
```

## Train

```bash
python3 rgbde_yolo_fusion/scripts/train_rgbde_yolo.py \
  --dataset-root ./dataset \
  --img-size 384 \
  --epochs 20 \
  --batch-size 8 \
  --checkpoint-dir rgbde_yolo_fusion/outputs/checkpoints \
  --results-dir rgbde_yolo_fusion/outputs/results
```

Main training artifact files:
- outputs/checkpoints/fusion_yolo_last.pt
- outputs/checkpoints/fusion_yolo_best.pt
- outputs/results/train_metrics.json

## Evaluate

```bash
python3 rgbde_yolo_fusion/scripts/eval_rgbde_yolo.py \
  --dataset-root ./dataset \
  --checkpoint rgbde_yolo_fusion/outputs/checkpoints/fusion_yolo_best.pt \
  --results-path rgbde_yolo_fusion/outputs/results/eval_metrics.json
```

## Compare Against Baseline RGB YOLO

```bash
python3 rgbde_yolo_fusion/scripts/compare_baseline_vs_fusion.py \
  --dataset-root ./dataset \
  --checkpoint rgbde_yolo_fusion/outputs/checkpoints/fusion_yolo_best.pt \
  --baseline-model ./yolov8n.pt \
  --vis-count 8 \
  --vis-dir rgbde_yolo_fusion/outputs/comparisons/side_by_side \
  --output-json rgbde_yolo_fusion/outputs/comparisons/baseline_vs_fusion.json \
  --output-md rgbde_yolo_fusion/outputs/comparisons/baseline_vs_fusion.md
```

Single-sample side-by-side visual comparison:

```bash
python3 rgbde_yolo_fusion/scripts/compare_baseline_vs_fusion.py \
  --dataset-root ./dataset \
  --checkpoint rgbde_yolo_fusion/outputs/checkpoints/fusion_yolo_best.pt \
  --baseline-model ./yolov8n.pt \
  --sample-id 000008 \
  --sample-split training \
  --vis-count 1 \
  --vis-dir rgbde_yolo_fusion/outputs/comparisons/side_by_side \
  --output-json rgbde_yolo_fusion/outputs/comparisons/baseline_vs_fusion.json \
  --output-md rgbde_yolo_fusion/outputs/comparisons/baseline_vs_fusion.md
```

Comparison artifacts:
- outputs/comparisons/baseline_vs_fusion.json
- outputs/comparisons/baseline_vs_fusion.md
- outputs/comparisons/side_by_side/side_by_side_*.png

## Single-Sample Inference

```bash
python3 rgbde_yolo_fusion/scripts/run_infer_sample.py \
  --dataset-root ./dataset \
  --checkpoint rgbde_yolo_fusion/outputs/checkpoints/fusion_yolo_best.pt \
  --sample-id 000008 \
  --save-json rgbde_yolo_fusion/outputs/results/infer_sample.json \
  --save-image rgbde_yolo_fusion/outputs/results/infer_sample.png
```

## Notes

- Depth map is produced from stereo pair and normalized to [0, 1].
- Edge map is Sobel magnitude from grayscale RGB.
- The model is intentionally YOLO-style and lightweight enough to run on a single GPU or CPU (slower).
- For fast iteration, use `--max-samples` during train/eval/compare.
