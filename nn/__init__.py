# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .tasks import (
    BaseModel,
    ClassificationModel,
    DetectionModel,
    SegmentationModel,
    guess_model_scale,
    guess_model_task,
    load_checkpoint,
    parse_model,
    torch_safe_load,
    yaml_model_load,
)

from .attention.CA import CoordAtt
from .attention.CA_CBAM import CBAM
globals()['CoordAtt'] = CoordAtt
globals()['CBAM'] = CBAM
__all__ = (
    "load_checkpoint",
    "parse_model",
    "yaml_model_load",
    "guess_model_task",
    "guess_model_scale",
    "torch_safe_load",
    "DetectionModel",
    "SegmentationModel",
    "ClassificationModel",
    "BaseModel",
    "CoordAtt",
    "CBAM",
)
