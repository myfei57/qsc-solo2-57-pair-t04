"""等室外温度分段网格与拟合样本。

分段方式在设计上定死为「等室外温度」：把曲线适用的室外温度区间
``[outdoor_min_c, outdoor_max_c]`` 等分成固定段数，断点只由配置决定，不随
某一轮采样分布漂移。按等负荷切分会让每轮断点跟着负荷估计走，换代之后断点
对不上，"这一版改的是哪一段"既无法逐段对比，也无法落审计。

曲线在断点之间线性插值、区间之外夹取到端点；分段（段）是相邻两个断点之间
的区间，版本换代的差异按段汇报。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import CurveLimits

SEGMENT_BASIS = "equal_outdoor"


@dataclass(frozen=True)
class CurveSample:
    """一个运行攒下来的采样点：室外温度与当时实际供水温度。

    ``ok`` 为假（采样丢点、仪表判坏、数值缺失）的点属于部分失败，清洗阶段
    必须逐点剔除并记录原因，任何情况下都不得进入拟合。
    """

    ts: float
    outdoor_c: float
    supply_c: float
    ok: bool = True
    reason: str = ""

    @classmethod
    def from_mapping(cls, raw: dict) -> "CurveSample":
        """从接口/文件字段还原一个采样点，字段缺失或非数值即判坏点。"""

        try:
            ts = float(raw.get("ts", 0.0))
            outdoor = float(raw["outdoor_c"])
            supply = float(raw["supply_c"])
        except (TypeError, ValueError, KeyError):
            return cls(
                ts=float(raw.get("ts", 0.0) or 0.0),
                outdoor_c=0.0,
                supply_c=0.0,
                ok=False,
                reason="missing_or_non_numeric",
            )
        ok = bool(raw.get("ok", True))
        reason = str(raw.get("reason", ""))
        if not ok and not reason:
            reason = "dropped"
        return cls(ts=ts, outdoor_c=outdoor, supply_c=supply, ok=ok, reason=reason)

    def as_dict(self) -> dict:
        return {
            "ts": round(self.ts, 3),
            "outdoor_c": round(self.outdoor_c, 2),
            "supply_c": round(self.supply_c, 2),
            "ok": self.ok,
            "reason": self.reason,
        }


class SegmentGrid:
    """等室外温度断点网格与段/插值工具。"""

    def __init__(self, limits: CurveLimits, segment_count: int) -> None:
        count = int(segment_count)
        if count < 2:
            raise ValueError("分段数至少为 2")
        self._limits = limits
        self._count = count
        span = limits.outdoor_max_c - limits.outdoor_min_c
        self._knots = tuple(
            limits.outdoor_min_c + span * index / count for index in range(count + 1)
        )

    @property
    def limits(self) -> CurveLimits:
        return self._limits

    @property
    def segment_count(self) -> int:
        return self._count

    def knots(self) -> tuple[float, ...]:
        """返回固定断点（室外温度，升序）。"""

        return self._knots

    def segments(self) -> list[dict]:
        """返回分段区间描述，供控制台展示与审计落盘。"""

        return [
            {
                "index": index,
                "outdoor_min_c": round(left, 2),
                "outdoor_max_c": round(right, 2),
                "basis": SEGMENT_BASIS,
            }
            for index, (left, right) in enumerate(zip(self._knots, self._knots[1:]))
        ]

    def segment_index(self, outdoor_c: float) -> int:
        """室外温度落在哪一段；区间外夹取到首段/末段。"""

        value = float(outdoor_c)
        if value <= self._knots[0]:
            return 0
        if value >= self._knots[-1]:
            return self._count - 1
        span = self._knots[-1] - self._knots[0]
        index = int((value - self._knots[0]) / span * self._count)
        return min(index, self._count - 1)

    def weights(self, outdoor_c: float) -> list[float]:
        """ Hat 基函数权重：只有包围该温度的两个断点非零。 """

        value = min(max(float(outdoor_c), self._knots[0]), self._knots[-1])
        index = self.segment_index(value)
        left, right = self._knots[index], self._knots[index + 1]
        ratio = 0.0 if right == left else (value - left) / (right - left)
        weights = [0.0] * (self._count + 1)
        weights[index] = 1.0 - ratio
        weights[index + 1] = ratio
        return weights

    def interpolate(self, outdoor_c: float, supplies: tuple[float, ...]) -> float:
        """按给定断点供温线性插值出目标供温。"""

        weights = self.weights(outdoor_c)
        return sum(weight * supply for weight, supply in zip(weights, supplies))

    def segment_values(self, supplies: tuple[float, ...]) -> list[dict]:
        """把断点供温折成逐段端点供温，换代差异按这份口径对比。"""

        return [
            {
                "index": index,
                "outdoor_min_c": round(left, 2),
                "outdoor_max_c": round(right, 2),
                "supply_left_c": round(supplies[index], 2),
                "supply_right_c": round(supplies[index + 1], 2),
                "basis": SEGMENT_BASIS,
            }
            for index, (left, right) in enumerate(zip(self._knots, self._knots[1:]))
        ]

    def describe(self) -> dict:
        """网格摘要：分段方式、断点数、断点室外温度与分段区间。"""

        return {
            "basis": SEGMENT_BASIS,
            "segment_count": self._count,
            "knots_outdoor_c": [round(knot, 2) for knot in self._knots],
            "segments": self.segments(),
        }
