"""供热曲线。

曲线按室外温度给出一次网目标供水温度：室外越冷，供温越高。曲线是"按气象更新"
的对象，每次更新都要先校验室外温度是否落在适用区间内，校验通过才允许覆盖旧值；
超过新鲜度窗口的曲线一律视为过期，热源不得据其点火。

曲线还可以按运行攒下来的（室外温度, 实际供温）采样**分段拟合换代**（见
:mod:`heatnet.temp.fitting`）：分段口径定死为等室外温度，换代后版本号加一，
新版本管新下发的调质、已下发的调质单仍按冻结的老版本结算。
"""

from __future__ import annotations

from .curve import HeatCurve
from .fitting import (
    CURVE_SEGMENTS,
    EMPIRICAL,
    REUSE_PREVIOUS,
    SEGMENTING,
    CurveSample,
    SegmentFit,
    RefitResult,
    empirical_supply,
    knot_outdoor_temps,
    normalize_samples,
    refit_curve,
)
from .forecast import ForecastPoint, build_forecast, describe_forecast
from .state import CurveSnapshot
from .table import CurveTable
from .version import CurveVersion, QualityOrder

__all__ = [
    "CURVE_SEGMENTS",
    "EMPIRICAL",
    "REUSE_PREVIOUS",
    "SEGMENTING",
    "CurveSnapshot",
    "CurveSample",
    "CurveTable",
    "CurveVersion",
    "ForecastPoint",
    "HeatCurve",
    "QualityOrder",
    "RefitResult",
    "SegmentFit",
    "build_forecast",
    "describe_forecast",
    "empirical_supply",
    "knot_outdoor_temps",
    "normalize_samples",
    "refit_curve",
]
