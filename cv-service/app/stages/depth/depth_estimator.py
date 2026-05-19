"""
Depth estimation stages — produce a per-pixel depth map.

Implementations
~~~~~~~~~~~~~~~
* ``DepthAnythingEstimator``    — DepthAnything v2 via HuggingFace.
* ``HeuristicDepthEstimator``   — no model; depth_map stays ``None``.
* ``MockDepthEstimator``        — testing stub.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from app.core.base import PipelineStage, StageContext
from app.core.config import settings
from app.core.logging import get_logger
from app.registry import depth_registry

logger = get_logger(__name__)


# ── DepthAnything v2 ─────────────────────────────────────────
@depth_registry.register("depth_anything_v2")
class DepthAnythingEstimator(PipelineStage):
    """Production depth estimator backed by DepthAnything v2."""

    def __init__(self) -> None:
        self._pipeline: Optional[Any] = None

    def load(self) -> None:
        try:
            from transformers import pipeline as hf_pipeline

            self._pipeline = hf_pipeline(
                "depth-estimation",
                model=settings.depth_model_name,
                device=0 if settings.device == "cuda" else -1,
            )
            logger.info("depth_model_loaded", model=settings.depth_model_name)
        except Exception as exc:
            logger.warning("depth_model_load_failed", error=str(exc))
            self._pipeline = None

    def process(self, ctx: StageContext) -> StageContext:
        if self._pipeline is None:
            return ctx

        try:
            out = self._pipeline(ctx.image)
            depth = np.array(out["depth"])
            d_min, d_max = depth.min(), depth.max()
            if d_max > d_min:
                depth = (depth - d_min) / (d_max - d_min)
            ctx.depth_map = depth
        except Exception as exc:
            logger.warning("depth_inference_failed", error=str(exc))

        return ctx


# ── Heuristic (no model) ────────────────────────────────────
@depth_registry.register("heuristic")
class HeuristicDepthEstimator(PipelineStage):
    """
    No depth model — leaves ``ctx.depth_map`` as ``None``.
    Postprocessor will fall back to bbox-area heuristic.
    """

    def load(self) -> None:
        pass

    def process(self, ctx: StageContext) -> StageContext:
        # depth_map stays None → postprocessor uses area heuristic
        return ctx


# ── Mock (testing) ───────────────────────────────────────────
@depth_registry.register("mock")
class MockDepthEstimator(PipelineStage):
    """Returns a uniform depth map — for unit tests."""

    def load(self) -> None:
        pass

    def process(self, ctx: StageContext) -> StageContext:
        w, h = ctx.image.size
        ctx.depth_map = np.ones((h, w), dtype=np.float32) * 0.5
        return ctx
