"""
Detection stages — find food regions in the image.

Implementations
~~~~~~~~~~~~~~~
* ``YoloV8Detector``  — production detector using Ultralytics YOLOv8.
* ``MockDetector``    — returns fixed bounding boxes for testing.

Register new detectors with::

    @detector_registry.register("yolov10")
    class YoloV10Detector(PipelineStage):
        def load(self): ...
        def process(self, ctx): ...
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from app.core.base import PipelineStage, RawDetection, StageContext
from app.core.config import settings
from app.core.logging import get_logger
from app.registry import detector_registry
from app.schemas.cv_schemas import BoundingBox

logger = get_logger(__name__)

# Minimum detection confidence to keep
_MIN_CONF = 0.35


# ── YOLOv8 ──────────────────────────────────────────────────
@detector_registry.register("yolov8")
class YoloV8Detector(PipelineStage):
    """Production food detector backed by Ultralytics YOLOv8."""

    def __init__(self) -> None:
        self._model: Optional[Any] = None

    def load(self) -> None:
        weights = settings.food_detection_weights
        if not Path(weights).exists():
            logger.warning(
                "yolo_weights_missing",
                path=weights,
                hint="Using pretrained yolov8n as fallback",
            )
            weights = "yolov8n.pt"

        from ultralytics import YOLO

        model = YOLO(weights)
        model.to(settings.device)
        self._model = model
        logger.info("yolo_loaded", weights=weights)

    def process(self, ctx: StageContext) -> StageContext:
        assert self._model is not None, "call load() before process()"
        results = self._model(ctx.image, verbose=False)
        boxes_raw = results[0].boxes

        for box in boxes_raw:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf.item())

            if conf < _MIN_CONF:
                continue

            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            crop = ctx.image.crop((x1, y1, x2, y2))

            ctx.detections.append(
                RawDetection(
                    bbox=bbox,
                    detector_confidence=conf,
                    crop=crop,
                )
            )

        logger.info(
            "detection_done",
            request_id=ctx.request_id,
            n_detections=len(ctx.detections),
        )
        return ctx


# ── Mock (testing) ───────────────────────────────────────────
@detector_registry.register("mock")
class MockDetector(PipelineStage):
    """Returns a single fixed bounding box — for unit tests."""

    def load(self) -> None:
        pass

    def process(self, ctx: StageContext) -> StageContext:
        bbox = BoundingBox(x1=10, y1=10, x2=200, y2=200)
        crop = ctx.image.crop((10, 10, 200, 200))
        ctx.detections.append(
            RawDetection(bbox=bbox, detector_confidence=0.95, crop=crop)
        )
        return ctx
