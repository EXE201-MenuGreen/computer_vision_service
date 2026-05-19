"""
Registry pattern — swap any stage via string key in config.

Usage::

    from app.registry import detector_registry

    @detector_registry.register("yolov8")
    class YoloV8Detector(PipelineStage):
        ...

    # Later, in factory:
    cls = detector_registry.get("yolov8")
    stage = cls()
    stage.load()
"""
from __future__ import annotations

from typing import Dict, Type

from app.core.base import PipelineStage
from app.core.logging import get_logger

logger = get_logger(__name__)


class StageRegistry:
    """
    A simple key → class mapping for one category of stages.

    Each stage category (detector, classifier, …) gets its own
    ``StageRegistry`` instance.
    """

    def __init__(self, category: str) -> None:
        self._category = category
        self._registry: Dict[str, Type[PipelineStage]] = {}

    # ── decorator ────────────────────────────────────────────
    def register(self, key: str):
        """Class decorator that registers a stage under *key*."""

        def decorator(cls: Type[PipelineStage]):
            if key in self._registry:
                logger.warning(
                    "registry_overwrite",
                    category=self._category,
                    key=key,
                    old=self._registry[key].__name__,
                    new=cls.__name__,
                )
            self._registry[key] = cls
            logger.debug(
                "stage_registered",
                category=self._category,
                key=key,
                cls=cls.__name__,
            )
            return cls

        return decorator

    # ── lookup ───────────────────────────────────────────────
    def get(self, key: str) -> Type[PipelineStage]:
        if key not in self._registry:
            available = ", ".join(sorted(self._registry)) or "(none)"
            raise KeyError(
                f"No '{self._category}' stage registered for key '{key}'. "
                f"Available: {available}"
            )
        return self._registry[key]

    def keys(self):
        return list(self._registry.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __repr__(self) -> str:
        entries = ", ".join(sorted(self._registry))
        return f"StageRegistry({self._category!r}, [{entries}])"


# ── Pre-built registries (one per stage category) ──────────
detector_registry = StageRegistry("detector")
classifier_registry = StageRegistry("classifier")
depth_registry = StageRegistry("depth")
postprocessor_registry = StageRegistry("postprocessor")
