"""
Post-processing stage — merge detection + classification + depth
into a final list of ``DetectedFood`` items.

Implements the gram-estimation heuristic (bbox area × depth).
"""
from __future__ import annotations

import numpy as np

from app.core.base import PipelineStage, StageContext
from app.core.logging import get_logger
from app.registry import postprocessor_registry
from app.schemas.cv_schemas import BoundingBox, DetectedFood

logger = get_logger(__name__)

# Average grams per food bbox area (pixels²) — rough heuristic baseline
GRAMS_PER_PIXEL_AREA = 0.05


def _estimate_grams(bbox: BoundingBox, depth_map: np.ndarray | None) -> float:
    """
    Estimate portion weight in grams.
    - If depth available: bbox area × mean_depth_value × calibration
    - Otherwise: fallback to bbox pixel area heuristic
    """
    area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)

    if depth_map is not None:
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
        region_depth = depth_map[y1:y2, x1:x2]
        mean_depth = float(np.mean(region_depth)) if region_depth.size > 0 else 1.0
        grams = area * mean_depth * 0.001  # calibrated constant
    else:
        grams = area * GRAMS_PER_PIXEL_AREA

    return max(10.0, min(grams, 800.0))  # clamp 10–800g


# ── Default postprocessor ───────────────────────────────────
@postprocessor_registry.register("default")
class DefaultPostprocessor(PipelineStage):
    """
    Merge classified detections with depth data to produce
    the final ``DetectedFood`` list in ``ctx.food_items``.
    """

    def load(self) -> None:
        pass

    def process(self, ctx: StageContext) -> StageContext:
        for cls_det in ctx.classifications:
            bbox = cls_det.raw.bbox
            grams = _estimate_grams(bbox, ctx.depth_map)

            # Confidence = average of detector + classifier
            combined_conf = round(
                (cls_det.raw.detector_confidence + cls_det.classify_confidence) / 2,
                3,
            )

            ctx.food_items.append(
                DetectedFood(
                    label=cls_det.label,
                    confidence=combined_conf,
                    bbox=bbox,
                    estimated_grams=round(grams, 1),
                )
            )

        logger.info(
            "postprocess_done",
            request_id=ctx.request_id,
            n_foods=len(ctx.food_items),
        )
        return ctx
