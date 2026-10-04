"""按运行数据把供热曲线分段重新拟合。

分段口径**定死为等室外温度分段**：把曲线适用区间 ``[outdoor_min_c, outdoor_max_c]``
按室外温度等分成 :data:`CURVE_SEGMENTS` 段，段间断点等距分布，不允许这一轮按室外
温度、下一轮按负荷换口径——否则两次换代出来的曲线没有可比性，审计也对不上。

每一段独立拿该段箱内的（室外温度, 实际供温）采样做一元线性回归，再把段两端断点
上的回归预测值拼成新的折线：相邻两段共用断点、预测值取两段回归的平均，所以拼出来
仍然是一条连续折线。某段数据不够（少于 :data:`MIN_SAMPLES_PER_SEGMENT` 个点）时
不做回归，按该管段在换代请求里指定的兜底策略取值——``reuse_previous`` 沿用上一版
曲线在该断点上的值，``empirical`` 按配置高/低温端点连成的经验线兜底。

采样丢点属于**部分失败**：缺字段、非数值、室外温度越界、供温非正的点只记进丢点
计数，这一轮不参与拟合，其余点照常拟合。只有一个可用点都没有时才算整轮失败，抛
:class:`~heatnet.errors.CurveFitError`，换代由调用方整条放弃，老版本原样保留。

断点回归值先夹回供温硬边界，再做一次保序（室外越冷、供温只能越高或相等），保序
采用 PAVA 整体融段，避免逐点夹取把误差传播成一条平线。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import CurveLimits
from ..errors import CurveFitError

# 分段方式定死：等室外温度分段（不是等负荷）。
SEGMENTING = "equal_outdoor"
# 曲线适用区间等分成四段，产生五个等距断点。
CURVE_SEGMENTS = 4
# 每个管段最少需要多少个采样点才做回归；不足即走兜底。
MIN_SAMPLES_PER_SEGMENT = 2

REUSE_PREVIOUS = "reuse_previous"
EMPIRICAL = "empirical"
FALLBACK_POLICIES = (REUSE_PREVIOUS, EMPIRICAL)

METHOD_FIT = "least_squares"
METHOD_REUSE = "reuse_previous"
METHOD_EMPIRICAL = "empirical"

# 丢点原因。
DROP_MISSING = "missing"
DROP_NOT_NUMBER = "not_number"
DROP_OUTDOOR_OUT_OF_RANGE = "outdoor_out_of_range"
DROP_SUPPLY_INVALID = "supply_invalid"


@dataclass(frozen=True)
class CurveSample:
    """一个运行采样：某时刻的室外温度与当时实际下发的供温。"""

    outdoor_c: float
    supply_c: float


@dataclass(frozen=True)
class SegmentFit:
    """一段管段的拟合结果。

    ``method`` 标明这一段最终用的是回归值、上一版沿用值还是经验兜底值；保序融段
    可能同时改动相邻几个断点，因此 ``band_clamped`` / ``monotone_clamped`` 按断点
    给出，控制台展示与审计落的都是这一份结构。
    """

    index: int
    outdoor_start_c: float
    outdoor_end_c: float
    sample_count: int
    dropped_count: int
    method: str
    fallback_policy: str
    points: tuple[tuple[float, float], ...]
    raw_supply_c: tuple[float, ...]
    band_clamped: tuple[bool, ...]
    monotone_clamped: tuple[bool, ...]

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "outdoor_range_c": [round(self.outdoor_start_c, 2), round(self.outdoor_end_c, 2)],
            "sample_count": self.sample_count,
            "dropped_count": self.dropped_count,
            "method": self.method,
            "fallback_policy": self.fallback_policy,
            "knots": [
                {
                    "outdoor_c": round(outdoor, 2),
                    "supply_c": round(supply, 2),
                    "raw_supply_c": round(raw, 2),
                    "band_clamped": band,
                    "monotone_clamped": mono,
                }
                for (outdoor, supply), raw, band, mono in zip(
                    self.points, self.raw_supply_c, self.band_clamped, self.monotone_clamped
                )
            ],
        }


@dataclass(frozen=True)
class RefitResult:
    """一轮换代的完整产物：新版本断点、分段明细与丢点计数。"""

    points: tuple[tuple[float, float], ...]
    segments: tuple[SegmentFit, ...]
    used_samples: int
    dropped: dict[str, int]
    dropped_total: int
    fitted_segments: int
    fallback_segments: int

    def segment_dicts(self) -> list[dict]:
        return [segment.as_dict() for segment in self.segments]


def knot_outdoor_temps(limits: CurveLimits) -> tuple[float, ...]:
    """等室外温度断点：四段五断点，首尾固定在适用区间两端。"""

    span = limits.outdoor_max_c - limits.outdoor_min_c
    return tuple(
        limits.outdoor_min_c + span * index / CURVE_SEGMENTS for index in range(CURVE_SEGMENTS + 1)
    )


def empirical_supply(limits: CurveLimits, outdoor_c: float) -> float:
    """经验兜底线：高/低温端点之间线性取值。"""

    ratio = (outdoor_c - limits.outdoor_min_c) / (limits.outdoor_max_c - limits.outdoor_min_c)
    return limits.supply_high_c + (limits.supply_low_c - limits.supply_high_c) * ratio


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def normalize_samples(
    limits: CurveLimits, raw_samples: list[dict]
) -> tuple[list[CurveSample], dict[str, int]]:
    """把接口进来的原始采样清洗成拟合输入。

    缺字段、非数值、室外温度越界、供温非正都算丢点，分类计数返回；清洗后的样本
    顺序与输入一致。
    """

    dropped = {
        DROP_MISSING: 0,
        DROP_NOT_NUMBER: 0,
        DROP_OUTDOOR_OUT_OF_RANGE: 0,
        DROP_SUPPLY_INVALID: 0,
    }
    samples: list[CurveSample] = []
    for raw in raw_samples:
        if not isinstance(raw, dict) or "outdoor_c" not in raw or "supply_c" not in raw:
            dropped[DROP_MISSING] += 1
            continue
        outdoor = _as_float(raw.get("outdoor_c"))
        supply = _as_float(raw.get("supply_c"))
        if outdoor is None or supply is None:
            dropped[DROP_NOT_NUMBER] += 1
            continue
        if not limits.outdoor_in_range(outdoor):
            dropped[DROP_OUTDOOR_OUT_OF_RANGE] += 1
            continue
        if supply <= 0.0:
            dropped[DROP_SUPPLY_INVALID] += 1
            continue
        samples.append(CurveSample(outdoor, supply))
    return samples, dropped


def _regress(samples: list[CurveSample]) -> tuple[float, float] | None:
    """一元线性回归 supply = slope * outdoor + intercept；样本不足或方差为零返回 None。"""

    if len(samples) < MIN_SAMPLES_PER_SEGMENT:
        return None
    count = len(samples)
    mean_x = sum(item.outdoor_c for item in samples) / count
    mean_y = sum(item.supply_c for item in samples) / count
    variance = sum((item.outdoor_c - mean_x) ** 2 for item in samples)
    if variance <= 1e-12:
        return None
    covariance = sum((item.outdoor_c - mean_x) * (item.supply_c - mean_y) for item in samples)
    slope = covariance / variance
    return slope, mean_y - slope * mean_x


def _resolve_policies(policies: object) -> tuple[str, ...]:
    """把换代请求里的兜底策略归一化成每段一个，非法值直接拒绝。"""

    if policies is None:
        return (REUSE_PREVIOUS,) * CURVE_SEGMENTS
    if isinstance(policies, str):
        if policies not in FALLBACK_POLICIES:
            raise CurveFitError(f"未知兜底策略 {policies!r}", component="temp")
        return (policies,) * CURVE_SEGMENTS
    if not isinstance(policies, (list, tuple)) or len(policies) != CURVE_SEGMENTS:
        raise CurveFitError(
            f"兜底策略必须是 {CURVE_SEGMENTS} 段各一个（或统一一个字符串）", component="temp"
        )
    resolved = tuple(str(item) for item in policies)
    for policy in resolved:
        if policy not in FALLBACK_POLICIES:
            raise CurveFitError(f"未知兜底策略 {policy!r}", component="temp")
    return resolved


def refit_curve(
    limits: CurveLimits,
    samples: list[CurveSample],
    dropped: dict[str, int] | None = None,
    previous_points: tuple[tuple[float, float], ...] | None = None,
    fallback_policies: object = None,
) -> RefitResult:
    """按等室外温度分段重新拟合一版曲线。

    全部采样都被丢掉时抛 :class:`CurveFitError`，调用方不得换代；只要还有可用点，
    数据不够的管段就按各自的兜底策略补齐，整轮算部分成功。
    """

    dropped = dict(dropped or {})
    for reason in (DROP_MISSING, DROP_NOT_NUMBER, DROP_OUTDOOR_OUT_OF_RANGE, DROP_SUPPLY_INVALID):
        dropped.setdefault(reason, 0)
    policies = _resolve_policies(fallback_policies)

    if not samples:
        raise CurveFitError("没有可用于拟合的采样，本轮不换代", component="temp")

    knots = knot_outdoor_temps(limits)
    # 段 i 的箱界取相邻断点的中点，保证每个样本恰好落进一段（首尾段向外敞开）。
    mids = [(knots[index] + knots[index + 1]) / 2.0 for index in range(CURVE_SEGMENTS - 1)]
    bounds = (
        [(-math.inf, mids[0])]
        + [(mids[index - 1], mids[index]) for index in range(1, CURVE_SEGMENTS - 1)]
        + [(mids[-1], math.inf)]
    )

    bins: list[list[CurveSample]] = [[] for _ in range(CURVE_SEGMENTS)]
    for sample in samples:
        for index, (left, right) in enumerate(bounds):
            if left <= sample.outdoor_c < right or (
                index == CURVE_SEGMENTS - 1 and sample.outdoor_c == right
            ):
                bins[index].append(sample)
                break

    if previous_points is None:
        previous_table = None
    else:
        previous_table = tuple((float(o), float(s)) for o, s in previous_points)

    # 先按段回归/兜底，得到每个断点的"原始候选值"：断点 k 是段 k-1 与段 k 的共用点，
    # 取两段在该断点预测的平均；首断点只属于首段，末断点只属于末段。
    regressions: list[tuple[float, float] | None] = []
    methods: list[str] = []
    for index, bucket in enumerate(bins):
        line = _regress(bucket)
        if line is not None:
            regressions.append(line)
            methods.append(METHOD_FIT)
            continue
        policy = policies[index]
        regressions.append(None)
        methods.append(METHOD_REUSE if policy == REUSE_PREVIOUS else METHOD_EMPIRICAL)

    raw_values: list[list[float]] = [[] for _ in knots]
    for index, line in enumerate(regressions):
        for knot_index in (index, index + 1):
            if line is not None:
                slope, intercept = line
                value = slope * knots[knot_index] + intercept
            elif methods[index] == METHOD_EMPIRICAL:
                value = empirical_supply(limits, knots[knot_index])
            else:
                value = _interpolate_previous(previous_table, limits, knots[knot_index])
            raw_values[knot_index].append(value)

    raw_knots = [sum(values) / len(values) for values in raw_values]

    # 夹回供温硬边界。
    band_clamped = [False] * len(knots)
    clamped: list[float] = []
    for knot_index, value in enumerate(raw_knots):
        if value > limits.supply_high_c:
            clamped.append(limits.supply_high_c)
            band_clamped[knot_index] = True
        elif value < limits.supply_low_c:
            clamped.append(limits.supply_low_c)
            band_clamped[knot_index] = True
        else:
            clamped.append(value)

    # PAVA 保序：断点按室外温度从低到高排列，供温必须非递增。
    fitted_values, mono_clamped = _isotonic_nonincreasing(clamped)

    segments: list[SegmentFit] = []
    for index in range(CURVE_SEGMENTS):
        segments.append(
            SegmentFit(
                index=index,
                outdoor_start_c=knots[index],
                outdoor_end_c=knots[index + 1],
                sample_count=len(bins[index]),
                dropped_count=0,
                method=methods[index],
                fallback_policy=policies[index],
                points=((knots[index], fitted_values[index]), (knots[index + 1], fitted_values[index + 1])),
                raw_supply_c=(raw_knots[index], raw_knots[index + 1]),
                band_clamped=(band_clamped[index], band_clamped[index + 1]),
                monotone_clamped=(mono_clamped[index], mono_clamped[index + 1]),
            )
        )

    dropped_total = sum(dropped.values())
    return RefitResult(
        points=tuple(zip(knots, fitted_values)),
        segments=tuple(segments),
        used_samples=len(samples),
        dropped=dropped,
        dropped_total=dropped_total,
        fitted_segments=sum(1 for method in methods if method == METHOD_FIT),
        fallback_segments=sum(1 for method in methods if method != METHOD_FIT),
    )


def _isotonic_nonincreasing(values: list[float]) -> tuple[list[float], list[bool]]:
    """PAVA 求一条非递增的最近逼近，返回逼近值与每个位置是否被融段改动。"""

    blocks: list[list[int]] = [[index] for index in range(len(values))]
    block_values: list[float] = list(values)
    index = 0
    while index < len(blocks) - 1:
        if block_values[index] >= block_values[index + 1]:
            index += 1
            continue
        merged = blocks[index] + blocks.pop(index + 1)
        blocks[index] = merged
        block_values[index] = sum(values[pos] for pos in merged) / len(merged)
        block_values.pop(index + 1)
        if index > 0:
            index -= 1
    fitted = [0.0] * len(values)
    changed = [False] * len(values)
    for positions, value in zip(blocks, block_values):
        for pos in positions:
            fitted[pos] = value
            changed[pos] = abs(value - values[pos]) > 1e-9
    return fitted, changed


def _interpolate_previous(
    previous: tuple[tuple[float, float], ...] | None,
    limits: CurveLimits,
    outdoor_c: float,
) -> float:
    """沿用上一版曲线在某断点上的插值；没有上一版时退回经验线。"""

    if not previous:
        return empirical_supply(limits, outdoor_c)
    if outdoor_c <= previous[0][0]:
        return previous[0][1]
    if outdoor_c >= previous[-1][0]:
        return previous[-1][1]
    for (left_x, left_y), (right_x, right_y) in zip(previous, previous[1:]):
        if left_x <= outdoor_c <= right_x:
            ratio = (outdoor_c - left_x) / (right_x - left_x)
            return left_y + (right_y - left_y) * ratio
    return previous[-1][1]
