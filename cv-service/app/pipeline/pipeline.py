"""
CVPipeline — orchestrator that runs a list of stages sequentially.

The pipeline is **stateless per request**: each call to ``run()``
creates a fresh ``StageContext`` and passes it through every stage.
"""
from __future__ import annotations

import time
from typing import List

from PIL import Image

from app.core.base import PipelineStage, StageContext
from app.core.logging import get_logger

logger = get_logger(__name__)


class CVPipeline:
    """
    Orchestrator: holds an ordered list of ``PipelineStage`` instances
    and runs them sequentially on each incoming image.
    """

    def __init__(self, stages: List[PipelineStage]) -> None:
        self._stages = stages
        self._ready = False

    # ── lifecycle ────────────────────────────────────────────
    def load_all(self) -> None:
        """Call ``load()`` on every stage (once, at startup)."""
        for stage in self._stages:
            logger.info("loading_stage", stage=stage.name)
            stage.load()
        self._ready = True
        logger.info("pipeline_ready", n_stages=len(self._stages))

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── inference ────────────────────────────────────────────
    def run(self, image: Image.Image) -> StageContext:
        """
        Create a fresh context, run all stages, return context.

        The caller (router / worker) is responsible for calling
        ``nutrition_service`` afterwards — the pipeline only
        handles CV inference stages.
        """
        ctx = StageContext(image=image)
        t0 = time.perf_counter()

        logger.info("pipeline_start", request_id=ctx.request_id, size=image.size)

        for stage in self._stages:
            ts = time.perf_counter()
            ctx = stage.process(ctx)
            elapsed = (time.perf_counter() - ts) * 1000
            ctx.stage_timings[stage.name] = round(elapsed, 2)

        ctx.processing_time_ms = round((time.perf_counter() - t0) * 1000, 1)

        logger.info(
            "pipeline_done",
            request_id=ctx.request_id,
            elapsed_ms=ctx.processing_time_ms,
            stage_timings=ctx.stage_timings,
        )
        return ctx

    # ── introspection ────────────────────────────────────────
    @property
    def stage_names(self) -> List[str]:
        return [s.name for s in self._stages]

    def __repr__(self) -> str:
        names = " → ".join(self.stage_names)
        return f"CVPipeline([{names}])"
