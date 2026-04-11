"""Compatibility wrapper around the newer RGB-D transformer 3D head."""

from __future__ import annotations

from ..approaches.learned.models import RGBDTransformer3DHead, decode_rgbd_transformer_output


# Backward-compatible aliases used by older scripts.
DepthAware3DHead = RGBDTransformer3DHead
decode_depth_aware_output = decode_rgbd_transformer_output

__all__ = [
    "RGBDTransformer3DHead",
    "decode_rgbd_transformer_output",
    "DepthAware3DHead",
    "decode_depth_aware_output",
]
