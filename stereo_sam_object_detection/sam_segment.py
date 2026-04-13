
from pathlib import Path
from typing import Sequence

import numpy as np


class SamSegmenter:
    def __init__(
        self,
        checkpoint_path: str | Path,
        model_type: str = "vit_b",
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {self.checkpoint_path}")

        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise RuntimeError(
                "segment-anything is required for SAM fusion. Install with: "
                "python3 -m pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from exc

        if model_type not in sam_model_registry:
            supported = ", ".join(sorted(sam_model_registry.keys()))
            raise ValueError(f"Unsupported SAM model_type '{model_type}'. Supported: {supported}")

        self.device = device
        sam = sam_model_registry[model_type](checkpoint=str(self.checkpoint_path))
        sam.to(device=self.device)
        sam.eval()

        self._torch = torch
        self.predictor = SamPredictor(sam)

    def predict_masks(self, image_bgr: np.ndarray, detections: Sequence) -> list[np.ndarray]:
        if len(detections) == 0:
            return []

        image_rgb = image_bgr[:, :, ::-1]
        self.predictor.set_image(image_rgb)

        masks: list[np.ndarray] = []
        with self._torch.inference_mode():
            for det in detections:
                box = np.array(det.bbox_xyxy, dtype=np.float32)
                pred_masks, _, _ = self.predictor.predict(
                    box=box,
                    multimask_output=False,
                )
                if pred_masks.ndim == 3:
                    mask = pred_masks[0]
                else:
                    mask = pred_masks
                masks.append(mask.astype(bool))
        return masks
