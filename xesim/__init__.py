"""xeSim: procedural-mechanistic Xenium tissue simulator (plan3).

Three tasks:

* ``XesimModel.fit(bundle, annotations, out_dir)`` — fit priors + train renderer
* ``model.explain(canonical_dir, …)`` — render real tiles
* ``model.generate(num_scenes=…, guide=…)`` — sample new scenes

CLI mirrors these as ``xesim fit-model``, ``xesim explain``, ``xesim generate``.

Quick start:

>>> from xesim import XesimModel
>>> m = XesimModel.load("path/to/model_dir")
>>> scenes = m.generate(num_scenes=2, seed=42)
>>> for s in scenes:
...     m.write(s, "out/")
"""

from .mechanistic_scene import MechanisticCell, MechanisticParams, MechanisticScene
from .model import XesimModel
from .torch_utils import get_device

__all__ = [
    "XesimModel",
    "MechanisticScene",
    "MechanisticCell",
    "MechanisticParams",
    "get_device",
]

__version__ = "0.1.0"
