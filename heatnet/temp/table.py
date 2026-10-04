"""供热曲线分段表。

曲线是一条按室外温度插值的折线。出厂的调试版用三个断点描述：室外最冷点对应最高
供温、室外最暖点对应最低供温、区间中点落在两者之间；按运行数据换代后的新版可以
是更多等距断点（见 :mod:`heatnet.temp.fitting`），插值口径不变。断点之间线性
插值，区间之外按端点取值——室外温度超出断点时曲线只负责夹取，是否越界由
:class:`~heatnet.temp.curve.HeatCurve` 的校验判定。
"""

from __future__ import annotations

from ..config import CurveLimits


class CurveTable:
    """按室外温度插值的一次网供温表。"""

    def __init__(self, limits: CurveLimits, points: tuple[tuple[float, float], ...] | None = None) -> None:
        self._limits = limits
        if points is None:
            mid = (limits.outdoor_min_c + limits.outdoor_max_c) / 2.0
            points = (
                (limits.outdoor_min_c, limits.supply_high_c),
                (mid, (limits.supply_high_c + limits.supply_low_c) / 2.0),
                (limits.outdoor_max_c, limits.supply_low_c),
            )
        points = tuple((float(outdoor), float(supply)) for outdoor, supply in points)
        if len(points) < 2:
            raise ValueError("曲线至少需要两个断点")
        if any(left[0] >= right[0] for left, right in zip(points, points[1:])):
            raise ValueError("曲线断点的室外温度必须严格递增")
        self._points = points

    @classmethod
    def from_points(cls, limits: CurveLimits, points: tuple[tuple[float, float], ...]) -> "CurveTable":
        """用一组拟合出来的断点构造新版本曲线表。"""

        return cls(limits, points)

    @property
    def limits(self) -> CurveLimits:
        return self._limits

    @property
    def points_tuple(self) -> tuple[tuple[float, float], ...]:
        """以元组形式返回断点，供换代拟合沿用上一版取值。"""

        return self._points

    def points(self) -> list[dict]:
        """返回断点列表。"""

        return [{"outdoor_c": round(o, 2), "supply_c": round(s, 2)} for o, s in self._points]

    def interpolate(self, outdoor_c: float) -> float:
        """按室外温度插值出供温。"""

        value = float(outdoor_c)
        if value <= self._points[0][0]:
            return self._points[0][1]
        if value >= self._points[-1][0]:
            return self._points[-1][1]
        for (left_x, left_y), (right_x, right_y) in zip(self._points, self._points[1:]):
            if left_x <= value <= right_x:
                ratio = (value - left_x) / (right_x - left_x)
                return left_y + (right_y - left_y) * ratio
        return self._points[-1][1]
