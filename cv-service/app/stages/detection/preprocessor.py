"""
Preprocessor stage — validates and normalises the input image.

This is the first stage in the pipeline.  It ensures the image
is in RGB mode and optionally resizes oversized images to keep
downstream inference fast.
"""
from __future__ import annotations

from app.core.base import PipelineStage, StageContext
from app.core.logging import get_logger

logger = get_logger(__name__)

MAX_DIM = 1920  # downscale if either side exceeds this


class PreprocessorStage(PipelineStage):
    """Validate + normalise the raw PIL image sitting in *ctx.image*."""

    def load(self) -> None:
        # Nothing to load — pure image transforms.
        pass

    def process(self, ctx: StageContext) -> StageContext:
        image = ctx.image

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            ctx.image = image

        # Downscale oversized images (preserve aspect ratio)
        w, h = image.size
        if max(w, h) > MAX_DIM:
            scale = MAX_DIM / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            ctx.image = image.resize(new_size)
            logger.info(
                "image_downscaled",
                original=(w, h),
                new=new_size,
                request_id=ctx.request_id,
            )

        logger.debug(
            "preprocess_done",
            size=ctx.image.size,
            request_id=ctx.request_id,
        )
        return ctx
