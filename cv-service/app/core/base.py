"""
Core contracts for the CV pipeline.

StageContext  – shared data bag passed between stages.
PipelineStage – abstract base every stage must implement.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
from PIL import Image

from app.schemas.cv_schemas import BoundingBox, DetectedFood


# ── Intermediate detection data ────────────────────────────
@dataclass
class RawDetection:
    """One bounding-box output from the detector stage."""
    bbox: BoundingBox
    detector_confidence: float
    crop: Optional[Image.Image] = None  # filled by detector


@dataclass
class ClassifiedDetection:
    """Detection enriched with classification label."""
    raw: RawDetection
    label: str
    classify_confidence: float


# ── Stage context (shared data bag) ────────────────────────
@dataclass
class StageContext:
    """
    Mutable bag of data flowing through the pipeline.

    Each stage reads what it needs and writes its output here.
    Adding a new stage only requires adding new fields — zero
    impact on existing stages.
    """
    # Input
    image: Image.Image

    # Metadata
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Stage outputs (populated progressively)
    detections: List[RawDetection] = field(default_factory=list)
    classifications: List[ClassifiedDetection] = field(default_factory=list)
    depth_map: Optional[np.ndarray] = None
    food_items: List[DetectedFood] = field(default_factory=list)

    # Timing / diagnostics
    processing_time_ms: float = 0.0
    stage_timings: dict = field(default_factory=dict)

    # Extensible metadata bucket
    extras: dict = field(default_factory=dict)


# ── Pipeline stage contract ────────────────────────────────
class PipelineStage(ABC):
    """
    Every pipeline stage must implement ``process(ctx)``.

    Stages are **stateful singletons** — they may hold loaded
    model weights, pre-computed transforms, etc.  ``load()``
    is called once at startup; ``process()`` is called per request.
    """

    @abstractmethod
    def load(self) -> None:
        """One-time initialisation (load weights, warm up, …)."""
        ...

    @abstractmethod
    def process(self, ctx: StageContext) -> StageContext:
        """Run inference and mutate *ctx* in-place.  Return *ctx*."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
