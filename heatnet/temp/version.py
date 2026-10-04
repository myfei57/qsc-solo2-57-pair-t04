"""曲线版本与调质单。

曲线换版是**版本变化**：每换代一次产生一个不可变的 :class:`CurveVersion`，版本号
只增不减。换代之后新下发的调质按新版插值计算，已经下发的调质单冻结在它自己的版本
上——:class:`QualityOrder` 记下发布那一刻的曲线版本号，应用时按那个版本取供温，
不允许中途换表，否则现场就会出现"单子上写 85℃、换代后悄悄变成 79℃"的对不上账。

拟合结果、控制台展示的分段与审计落盘的内容都从同一份 :class:`CurveVersion` 取，
保证"拟合、曲线与审计一致"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .fitting import SEGMENTING

METHOD_COMMISSIONING = "commissioning"


@dataclass(frozen=True)
class CurveVersion:
    """一版不可变的供热曲线：版本号、断点、分段明细与换代来源。"""

    version: int
    points: tuple[tuple[float, float], ...]
    segments: tuple[dict, ...]
    segmenting: str
    created_at: float
    source: str
    used_samples: int
    dropped_total: int
    fitted_segments: int
    fallback_segments: int

    def table_points(self) -> list[dict]:
        """断点列表（控制台与审计共用的口径）。"""

        return [{"outdoor_c": round(o, 2), "supply_c": round(s, 2)} for o, s in self.points]

    def describe_segments(self) -> list[dict]:
        """分段明细（拟合/兜底方式、样本数、夹取与保序痕迹）。"""

        return [dict(segment) for segment in self.segments]

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "segmenting": self.segmenting,
            "segments": self.describe_segments(),
            "points": self.table_points(),
            "created_at": round(self.created_at, 3),
            "source": self.source,
            "used_samples": self.used_samples,
            "dropped_total": self.dropped_total,
            "fitted_segments": self.fitted_segments,
            "fallback_segments": self.fallback_segments,
        }


def commissioning_segments(points: tuple[tuple[float, float], ...]) -> tuple[dict, ...]:
    """出厂调试版的分段明细：三点折线即两段，方法记为 commissioning。"""

    segments: list[dict] = []
    for index, ((left_o, left_s), (right_o, right_s)) in enumerate(zip(points, points[1:])):
        segments.append(
            {
                "index": index,
                "outdoor_range_c": [round(left_o, 2), round(right_o, 2)],
                "sample_count": 0,
                "dropped_count": 0,
                "method": METHOD_COMMISSIONING,
                "fallback_policy": "",
                "knots": [
                    {
                        "outdoor_c": round(left_o, 2),
                        "supply_c": round(left_s, 2),
                        "raw_supply_c": round(left_s, 2),
                        "band_clamped": False,
                        "monotone_clamped": False,
                    },
                    {
                        "outdoor_c": round(right_o, 2),
                        "supply_c": round(right_s, 2),
                        "raw_supply_c": round(right_s, 2),
                        "band_clamped": False,
                        "monotone_clamped": False,
                    },
                ],
            }
        )
    return tuple(segments)


# 调质单状态。
ORDER_PENDING = "pending"
ORDER_APPLIED = "applied"
ORDER_CANCELLED = "cancelled"


@dataclass
class QualityOrder:
    """一单已下发的调质：下发时冻结曲线版本，应用时按冻结版本取供温。"""

    order_id: str
    outdoor_c: float
    curve_version: int
    target_supply_c: float
    issued_at: float
    origin: str = "console"
    status: str = ORDER_PENDING

    def as_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "outdoor_c": round(self.outdoor_c, 2),
            "curve_version": self.curve_version,
            "target_supply_c": round(self.target_supply_c, 2),
            "issued_at": round(self.issued_at, 3),
            "origin": self.origin,
            "status": self.status,
        }
