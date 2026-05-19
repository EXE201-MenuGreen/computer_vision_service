"""Auto-import all stage modules so they self-register with registries."""

# Detection
from app.stages.detection import preprocessor  # noqa: F401
from app.stages.detection import detector       # noqa: F401

# Classification
from app.stages.classification import classifier       # noqa: F401
from app.stages.classification import clip_classifier  # noqa: F401

# Depth
from app.stages.depth import depth_estimator  # noqa: F401

# Post-processing
from app.stages.postprocess import postprocessor  # noqa: F401
