from app.registry.stage_registry import (
    StageRegistry,
    detector_registry,
    classifier_registry,
    depth_registry,
    postprocessor_registry,
)

__all__ = [
    "StageRegistry",
    "detector_registry",
    "classifier_registry",
    "depth_registry",
    "postprocessor_registry",
]
