# Stereo Object Detection (3D)

Stereo + YOLO pipeline for estimating 3D bounding boxes in traffic scenes.

- Input: left/right RGB stereo images + camera calibration + YOLO 2D detections
- Output: 3D box parameters `(x, y, z, w, h, l, theta)`

## Repository Layout

```text
stereo_object_detection/
├── geometric/
│   ├── depth.py
│   ├── yolo_detect.py
│   ├── fusion_math.py
│   └── run_geo.py
├── learned/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── infer.py
│   └── utils.py
├── scripts/
├── src/
├── configs/
├── outputs/
├── arch.png
├── requirements.txt
└── .gitignore
```

## Methodology

### 1. Geometric Approach

`geometric/run_geo.py` executes the complete deterministic pipeline:

1. Stereo matching (`SGBM`/`BM`) to get disparity
2. Disparity to depth conversion
3. YOLO 2D object detection
4. Robust depth extraction per bbox (outlier filtering)
5. 3D box fitting in camera frame
6. Projection of 3D corners back onto the image

Properties:
- No training required
- Fast and interpretable
- Quality depends strongly on stereo depth quality

### 2. Learned Approach

`learned/train.py` + `learned/infer.py` implement a data-driven refinement pipeline:

1. Use YOLO detections as object proposals
2. Crop RGB and depth patches per bbox
3. Normalize and resize patches to fixed input size
4. Dual encoders (RGB encoder + depth encoder)
5. Transformer fusion head combines RGB/depth/geometry cues
6. Regress 3D parameters and decode to final 3D box

Properties:
- Requires training
- Better robustness potential in challenging scenes
- Depends on training data quality and domain match

## Architecture Diagram

![Architecture](arch.png)

## Outputs (Saved Artifacts)

The repository keeps representative outputs for sample `training_000000`:

### Geometric Outputs
- `outputs/training_000000_geo_depth_map.png`
- `outputs/training_000000_geo_yolo_2d.png`
- `outputs/training_000000_geo_3dbox.png`

### Learned Outputs
- `outputs/training_000000_learned_depth_map.png`
- `outputs/training_000000_learned_yolo_2d.png`
- `outputs/training_000000_learned_3dbox.png`

## Quick Run

### Geometric

```bash
python3 geometric/run_geo.py \
  --dataset-root ./dataset \
  --split training \
  --sample-id 000000 \
  --no-show
```

### Learned Training

```bash
python3 learned/train.py \
  --dataset-root ./dataset \
  --max-frames 300 \
  --epochs 8 \
  --batch-size 16
```

### Learned Inference

```bash
python3 learned/infer.py \
  --dataset-root ./dataset \
  --split training \
  --sample-id 000000 \
  --checkpoint ./outputs/learned_fusion_3d.pt \
  --no-show
```

## What Is Ignored (Unnecessary/Heavy Files)

See `.gitignore` for excluded data and heavy artifacts, including:
- `dataset/`
- `kitti-dataset.zip`
- checkpoints (`*.pt`, `*.pth`, `*.ckpt`)
- virtual environments (`.venv/`, `.venv311/`)
- generated output files except selected showcase images
