"""供热曲线。

曲线按室外温度给出一次网目标供水温度：室外越冷，供温越高。曲线是"按气象更新"
的对象，每次更新都要先校验室外温度是否落在适用区间内，校验通过才允许覆盖旧值；
超过新鲜度窗口的曲线一律视为过期，热源不得据其点火。

运行攒下的"室外温度 + 实际供温"由 :class:`~heatnet.temp.tuning.CurveTuningService`
按管段做等室外温度分段拟合，版本链与已下发调质固化在
:class:`~heatnet.temp.tuning.CurveCatalog` 里。
"""

from __future__ import annotations

from .curve import HeatCurve
from .forecast import ForecastPoint, build_forecast, describe_forecast
from .segments import SEGMENT_BASIS, CurveSample, SegmentGrid
from .state import CurveSnapshot
from .table import CurveTable
from .tuning import (
    CurveCatalog,
    CurveTuningService,
    CurveVersion,
    QualityOrder,
)

__all__ = [
    "SEGMENT_BASIS",
    "CurveCatalog",
    "CurveSample",
    "CurveSnapshot",
    "CurveTable",
    "CurveTuningService",
    "CurveVersion",
    "ForecastPoint",
    "HeatCurve",
    "QualityOrder",
    "SegmentGrid",
    "build_forecast",
    "describe_forecast",
]
