from __future__ import annotations

from . import recommendation_exact_runtime as _exact_runtime
from . import recommendation_models as _models
from . import recommendation_service_finalize as _finalize
from . import recommendation_service_runtime as _runtime
from .recommendation_auto_gate import _prediction_shape_gate, evaluate_recommendation_policy


def _reexport(module) -> None:
    for name in dir(module):
        if name.startswith("__"):
            continue
        globals()[name] = getattr(module, name)


for _module in (_models, _exact_runtime, _finalize, _runtime):
    _reexport(_module)

