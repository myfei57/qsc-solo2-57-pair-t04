"""分段拟合：样本清洗、连续最小二乘、经验兜底与逐段换代差异。

拟合只认等室外温度断点网格（见 :mod:`heatnet.temp.segments`）：每个采样点按
其室外温度映射到相邻两个断点的 hat 基函数上，对全部断点供温做一次连续分段
线性最小二乘，因此段与段在断点处天然连续，不会出现折线错位。

采样丢点属于**部分失败**而不是整轮作废：``clean_samples`` 把坏点逐点剔出并
保留剔除原因，坏点一个都不进设计矩阵；但只要某一段有效点数不足门限（或总
点数不足），该管段本轮数据就视为不可拟合，整管段走兜底，不允许只换其中几段，
否则未覆盖段会被噪声解带着走。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import CurveFitLimits
from .segments import CurveSample, SegmentGrid

OUTDOOR_OUT_OF_RANGE = "outdoor_out_of_range"
FITTED = "fitted"
INHERIT = "inherit"
EMPIRICAL = "empirical"


@dataclass(frozen=True)
class SegmentChange:
    """换代时一段的新旧对比。"""

    index: int
    changed: bool
    old_supply_left_c: float
    old_supply_right_c: float
    new_supply_left_c: float
    new_supply_right_c: float
    max_delta_c: float

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "changed": self.changed,
            "old_supply_left_c": round(self.old_supply_left_c, 2),
            "old_supply_right_c": round(self.old_supply_right_c, 2),
            "new_supply_left_c": round(self.new_supply_left_c, 2),
            "new_supply_right_c": round(self.new_supply_right_c, 2),
            "max_delta_c": round(self.max_delta_c, 3),
        }


@dataclass(frozen=True)
class FitResult:
    """一轮最小二乘的内部结果：覆盖门限、断点供温与逐段残差。"""

    covered: bool
    supplies: tuple[float, ...]
    segment_counts: tuple[int, ...]
    segment_rmse: tuple[float, ...]
    total: int

    def as_dict(self) -> dict:
        return {
            "covered": self.covered,
            "knot_supply_c": [round(value, 2) for value in self.supplies],
            "segment_samples": list(self.segment_counts),
            "segment_rmse_c": [round(value, 3) for value in self.segment_rmse],
            "total": self.total,
        }


def empirical_supplies(grid: SegmentGrid) -> tuple[float, ...]:
    """经验兜底曲线：最冷点供高、最暖点供低，断点之间线性取值。"""

    limits = grid.limits
    knots = grid.knots()
    span = knots[-1] - knots[0]
    supplies = [
        limits.supply_high_c
        + (limits.supply_low_c - limits.supply_high_c) * (knot - knots[0]) / span
        for knot in knots
    ]
    return tuple(supplies)


def clean_samples(
    raw_samples: list[CurveSample], grid: SegmentGrid
) -> tuple[list[CurveSample], list[dict]]:
    """逐点清洗采样：坏点与越界点剔出（部分失败），返回「有效点, 丢弃点」。"""

    accepted: list[CurveSample] = []
    dropped: list[dict] = []
    for sample in raw_samples:
        if not sample.ok:
            dropped.append({**sample.as_dict(), "reason": sample.reason or "dropped"})
            continue
        if not grid.limits.outdoor_in_range(sample.outdoor_c):
            dropped.append({**sample.as_dict(), "ok": False, "reason": OUTDOOR_OUT_OF_RANGE})
            continue
        accepted.append(sample)
    return accepted, dropped


def fit_curve(
    grid: SegmentGrid,
    samples: list[CurveSample],
    limits: CurveFitLimits,
) -> FitResult:
    """在固定断点网格上拟合连续分段线性曲线。"""

    dimension = grid.segment_count + 1
    counts = [0] * grid.segment_count
    for sample in samples:
        counts[grid.segment_index(sample.outdoor_c)] += 1
    segment_counts = tuple(counts)

    covered = len(samples) >= limits.min_samples_total and all(
        count >= limits.min_samples_per_segment for count in segment_counts
    )
    placeholder = tuple(0.0 for _ in range(dimension))
    empty = tuple(0.0 for _ in range(grid.segment_count))
    if not covered:
        return FitResult(False, placeholder, segment_counts, empty, len(samples))

    matrix = [[0.0] * dimension for _ in range(dimension)]
    vector = [0.0] * dimension
    for sample in samples:
        weights = grid.weights(sample.outdoor_c)
        for row, row_weight in enumerate(weights):
            if row_weight == 0.0:
                continue
            vector[row] += row_weight * sample.supply_c
            for column, column_weight in enumerate(weights):
                if column_weight != 0.0:
                    matrix[row][column] += row_weight * column_weight
    try:
        solved = _solve_symmetric(matrix, vector)
    except ArithmeticError:
        return FitResult(False, placeholder, segment_counts, empty, len(samples))

    low, high = grid.limits.supply_low_c, grid.limits.supply_high_c
    supplies = tuple(min(max(value, low), high) for value in solved)

    residuals: list[list[float]] = [[] for _ in range(grid.segment_count)]
    for sample in samples:
        predicted = grid.interpolate(sample.outdoor_c, supplies)
        residuals[grid.segment_index(sample.outdoor_c)].append((predicted - sample.supply_c) ** 2)
    rmse = tuple(
        (sum(values) / len(values)) ** 0.5 if values else 0.0 for values in residuals
    )
    return FitResult(True, supplies, segment_counts, rmse, len(samples))


def diff_segments(
    grid: SegmentGrid,
    old_supplies: tuple[float, ...],
    new_supplies: tuple[float, ...],
    tolerance_c: float,
) -> list[SegmentChange]:
    """逐段对比新旧曲线，段内（两端断点 + 中点插值）供温差超门限即记为改动段。"""

    changes: list[SegmentChange] = []
    for index, (left, right) in enumerate(zip(grid.knots(), grid.knots()[1:])):
        midpoint = (left + right) / 2.0
        deltas = (
            abs(new_supplies[index] - old_supplies[index]),
            abs(new_supplies[index + 1] - old_supplies[index + 1]),
            abs(
                grid.interpolate(midpoint, new_supplies)
                - grid.interpolate(midpoint, old_supplies)
            ),
        )
        max_delta = max(deltas)
        changes.append(
            SegmentChange(
                index=index,
                changed=max_delta > tolerance_c,
                old_supply_left_c=old_supplies[index],
                old_supply_right_c=old_supplies[index + 1],
                new_supply_left_c=new_supplies[index],
                new_supply_right_c=new_supplies[index + 1],
                max_delta_c=max_delta,
            )
        )
    return changes


def _solve_symmetric(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """高斯消元解正定法方程（带列主元），奇异即抛出。"""

    size = len(vector)
    work = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for pivot in range(size):
        pivot_row = max(range(pivot, size), key=lambda row: abs(work[row][pivot]))
        if abs(work[pivot_row][pivot]) < 1e-10:
            raise ArithmeticError("法方程奇异")
        work[pivot], work[pivot_row] = work[pivot_row], work[pivot]
        pivot_value = work[pivot][pivot]
        for row in range(pivot + 1, size):
            factor = work[row][pivot] / pivot_value
            if factor == 0.0:
                continue
            for column in range(pivot, size + 1):
                work[row][column] -= factor * work[pivot][column]
    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        remainder = work[row][size] - sum(work[row][column] * solution[column] for column in range(row + 1, size))
        solution[row] = remainder / work[row][row]
    return solution
