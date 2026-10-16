"""
dsu: Convenience alias and backward-compatibility interface for datascienceutils
"""
import sys
import datascienceutils as _dsu
from datascienceutils import *

# Explicitly re-export all submodules
from datascienceutils import (
    analyze,
    dimensionAnalyze,
    plotter,
    timeSeriesUtils,
    statsutils,
    sklearnUtils,
    sampling,
    outliers,
    dists,
    clusteringModels,
    datautils,
    modelValidators,
    predictiveModels,
    utils,
)

# Re-export all public attributes from datascienceutils onto dsu
for _attr in dir(_dsu):
    if not _attr.startswith("_") or _attr in ("_RUST_AVAILABLE", "__version__"):
        globals()[_attr] = getattr(_dsu, _attr)

# Register submodules under sys.modules so `from dsu import ...` works universally
sys.modules["dsu.analyze"] = analyze
sys.modules["dsu.dimensionAnalyze"] = dimensionAnalyze
sys.modules["dsu.plotter"] = plotter
sys.modules["dsu.timeSeriesUtils"] = timeSeriesUtils
sys.modules["dsu.statsutils"] = statsutils
sys.modules["dsu.sklearnUtils"] = sklearnUtils
sys.modules["dsu.sampling"] = sampling
sys.modules["dsu.outliers"] = outliers
sys.modules["dsu.dists"] = dists
sys.modules["dsu.clusteringModels"] = clusteringModels
sys.modules["dsu.datautils"] = datautils
sys.modules["dsu.modelValidators"] = modelValidators
sys.modules["dsu.predictiveModels"] = predictiveModels
sys.modules["dsu.utils"] = utils

__all__ = list(_dsu.__all__)
