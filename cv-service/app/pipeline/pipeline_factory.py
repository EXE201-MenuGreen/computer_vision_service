"""
PipelineFactory — build a ``CVPipeline`` from a config dict.

Config example::

    {
        "detector":      "yolov8",
        "classifier":    "efficientnet_b4",
        "depth":         "depth_anything_v2",
        "postprocessor": "default",
    }

Each key maps to a ``StageRegistry``; the value is the registered
key inside that registry.  Change one line in ``.env`` to swap a
model — no code changes, no redeploy.
"""
from __future__ import annotations

from typing import Dict

from app.core.logging import get_logger
from app.pipeline.pipeline import CVPipeline
from app.registry import (
    classifier_registry,
    depth_registry,
    detector_registry,
    postprocessor_registry,
)
from app.stages.detection.preprocessor import PreprocessorStage

logger = get_logger(__name__)


class PipelineFactory:
    """Assemble a ``CVPipeline`` from a flat config dict."""

    @staticmethod
    def build(config: Dict[str, str]) -> CVPipeline:
        """
        Build and return a ``CVPipeline`` with stages resolved
        from the registries.

        Parameters
        ----------
        config : dict
            Keys: ``detector``, ``classifier``, ``depth``,
            ``postprocessor``.  Values: registered string keys.

        Returns
        -------
        CVPipeline
            Ready-to-``load_all()`` pipeline instance.
        """
        # 1. Preprocessor is always first (not configurable)
        preprocessor = PreprocessorStage()

        # 2. Resolve configurable stages
        detector_cls = detector_registry.get(config["detector"])
        classifier_cls = classifier_registry.get(config["classifier"])
        depth_cls = depth_registry.get(config["depth"])
        postprocessor_cls = postprocessor_registry.get(config["postprocessor"])

        stages = [
            preprocessor,
            detector_cls(),
            classifier_cls(),
            depth_cls(),
            postprocessor_cls(),
        ]

        pipeline = CVPipeline(stages)

        logger.info(
            "pipeline_built",
            stages=[s.name for s in stages],
            config=config,
        )
        return pipeline
