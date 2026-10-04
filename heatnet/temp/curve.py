"""供热曲线与气象校验。

一次气象推送只有在室外温度落在适用区间内时才允许覆盖曲线。校验不通过时抛出
:class:`~heatnet.errors.ForecastError`，调用方必须把这条推送整条作废——把错误
吞掉继续点火，热源就会按上一版曲线运行，越界的室外温度根本进不到曲线里，现场
看到的现象是"曲线还是老的，供热没跟着天气走"。

曲线本身有**版本**：出厂是调试版（version 1，三点两段），按运行数据分段拟合换代
后版本号加一，旧版本保留一份供已下发的调质单结算。换代在同一把锁里换表，换表前后
读不到半新半旧的断点。
"""

from __future__ import annotations

import threading

from ..config import CurveLimits
from ..errors import CurveFitError, ForecastError
from .fitting import (
    SEGMENTING,
    CurveSample,
    RefitResult,
    refit_curve,
)
from .state import CurveSnapshot
from .table import CurveTable
from .version import (
    ORDER_APPLIED,
    ORDER_CANCELLED,
    ORDER_PENDING,
    CurveVersion,
    QualityOrder,
    commissioning_segments,
)


class HeatCurve:
    """按气象更新、可按运行数据换代的供热曲线。"""

    def __init__(self, limits: CurveLimits) -> None:
        self._limits = limits
        self._guard = threading.RLock()
        initial_table = CurveTable(limits)
        initial_points = initial_table.points_tuple
        self._tables: dict[int, CurveTable] = {1: initial_table}
        self._version = CurveVersion(
            version=1,
            points=initial_points,
            segments=commissioning_segments(initial_points),
            segmenting=SEGMENTING,
            created_at=0.0,
            source="commissioning",
            used_samples=0,
            dropped_total=0,
            fitted_segments=0,
            fallback_segments=0,
        )
        self._versions: dict[int, CurveVersion] = {1: self._version}
        self._table = initial_table
        self._outdoor_c = (limits.outdoor_min_c + limits.outdoor_max_c) / 2.0
        self._supply_c = self._table.interpolate(self._outdoor_c)
        self._updated_at = 0.0
        self._epoch = 0
        self._validated = 0
        self._rejected = 0
        self._orders: dict[str, QualityOrder] = {}
        self._order_seq = 0

    @property
    def limits(self) -> CurveLimits:
        return self._limits

    @property
    def table(self) -> CurveTable:
        with self._guard:
            return self._table

    def outdoor_c(self) -> float:
        """返回当前曲线对应的室外温度。"""

        return self._outdoor_c

    def supply_c(self) -> float:
        """返回当前曲线的目标供水温度。"""

        return self._supply_c

    def epoch(self) -> int:
        """返回气象推送代际，每次成功推送加一（与曲线版本号不是一回事）。"""

        return self._epoch

    def version(self) -> int:
        """返回当前生效的曲线版本号。"""

        with self._guard:
            return self._version.version

    def version_info(self, version: int | None = None) -> CurveVersion:
        """返回某一版（默认当前版）的完整版本信息。"""

        with self._guard:
            key = self._version.version if version is None else version
            try:
                return self._versions[key]
            except KeyError:
                raise CurveFitError(f"曲线版本 {key} 不存在", component="temp") from None

    def segmenting(self) -> str:
        """返回分段口径（定死为等室外温度）。"""

        return SEGMENTING

    def updated_at(self) -> float:
        """返回最近一次成功更新的时刻。"""

        return self._updated_at

    def validated(self) -> int:
        """返回累计通过的校验次数。"""

        return self._validated

    def rejected(self) -> int:
        """返回累计被拒绝的校验次数。"""

        return self._rejected

    def validate(self, outdoor_c: float) -> float:
        """校验室外温度是否落在适用区间内，越界即抛出。"""

        value = float(outdoor_c)
        if not self._limits.outdoor_in_range(value):
            self._rejected += 1
            raise ForecastError(
                f"室外温度 {value:.1f}℃ 超出曲线适用区间 "
                f"[{self._limits.outdoor_min_c:.1f}, {self._limits.outdoor_max_c:.1f}]",
                component="temp",
            )
        self._validated += 1
        return value

    def apply_forecast(self, outdoor_c: float, now: float) -> dict:
        """校验后覆盖当前气象点，返回更新后的摘要（不换版本）。"""

        value = self.validate(outdoor_c)
        with self._guard:
            self._outdoor_c = value
            self._supply_c = self._table.interpolate(value)
            self._updated_at = float(now)
            self._epoch += 1
            return {
                "outdoor_c": round(self._outdoor_c, 2),
                "supply_c": round(self._supply_c, 2),
                "epoch": self._epoch,
                "curve_version": self._version.version,
                "updated_at": round(self._updated_at, 3),
            }

    def refit(
        self,
        samples: list[CurveSample],
        dropped: dict[str, int],
        now: float,
        *,
        source: str = "runtime",
        fallback_policies: object = None,
    ) -> RefitResult:
        """按运行采样分段拟合并换版，返回拟合明细。

        拟合抛错时不触碰当前版本；成功才在同一把锁里换表、存旧版。
        """

        with self._guard:
            result = refit_curve(
                self._limits,
                samples,
                dropped=dropped,
                previous_points=self._version.points,
                fallback_policies=fallback_policies,
            )
            new_number = self._version.version + 1
            version = CurveVersion(
                version=new_number,
                points=result.points,
                segments=tuple(result.segment_dicts()),
                segmenting=SEGMENTING,
                created_at=float(now),
                source=str(source),
                used_samples=result.used_samples,
                dropped_total=result.dropped_total,
                fitted_segments=result.fitted_segments,
                fallback_segments=result.fallback_segments,
            )
            self._versions[new_number] = version
            self._tables[new_number] = CurveTable.from_points(self._limits, result.points)
            self._version = version
            self._table = self._tables[new_number]
            self._supply_c = self._table.interpolate(self._outdoor_c)
            return result

    def supply_for_version(self, version: int, outdoor_c: float) -> float:
        """按指定版本插值供温；版本不存在即拒绝。"""

        with self._guard:
            try:
                table = self._tables[version]
            except KeyError:
                raise CurveFitError(f"曲线版本 {version} 不存在", component="temp") from None
            value = float(outdoor_c)
            if not self._limits.outdoor_in_range(value):
                raise ForecastError(
                    f"室外温度 {value:.1f}℃ 超出曲线适用区间 "
                    f"[{self._limits.outdoor_min_c:.1f}, {self._limits.outdoor_max_c:.1f}]",
                    component="temp",
                )
            return table.interpolate(value)

    # ---- 调质单：换代后新单按新版、已下发单按冻结的老版 ----------------

    def issue_quality_order(self, outdoor_c: float, now: float, origin: str = "console") -> QualityOrder:
        """下发一单调质：校验室外温度并冻结当前曲线版本。"""

        value = self.validate(outdoor_c)
        with self._guard:
            self._order_seq += 1
            order = QualityOrder(
                order_id=f"qo-{self._order_seq}",
                outdoor_c=value,
                curve_version=self._version.version,
                target_supply_c=self._table.interpolate(value),
                issued_at=float(now),
                origin=str(origin),
                status=ORDER_PENDING,
            )
            self._orders[order.order_id] = order
            return order

    def quality_order(self, order_id: str) -> QualityOrder:
        """取一单调质。"""

        with self._guard:
            try:
                return self._orders[str(order_id)]
            except KeyError:
                raise CurveFitError(f"调质单 {order_id!r} 不存在", component="temp") from None

    def orders(self) -> list[QualityOrder]:
        """返回全部调质单（按下发顺序）。"""

        with self._guard:
            return list(self._orders.values())

    def resolve_quality_order(self, order_id: str) -> QualityOrder:
        """应用一单调质：按单子冻结的版本取供温，单子只能应用一次。"""

        with self._guard:
            order = self.quality_order(order_id)
            if order.status != ORDER_PENDING:
                raise CurveFitError(
                    f"调质单 {order.order_id} 当前状态 {order.status}，不能应用", component="temp"
                )
            table = self._tables[order.curve_version]
            order.status = ORDER_APPLIED
            order.target_supply_c = table.interpolate(order.outdoor_c)
            return order

    def cancel_quality_order(self, order_id: str) -> QualityOrder:
        """撤销一单尚未应用的调质。"""

        with self._guard:
            order = self.quality_order(order_id)
            if order.status != ORDER_PENDING:
                raise CurveFitError(
                    f"调质单 {order.order_id} 当前状态 {order.status}，不能撤销", component="temp"
                )
            order.status = ORDER_CANCELLED
            return order

    def is_stale(self, now: float) -> bool:
        """曲线是否已经超过新鲜度窗口。"""

        return (float(now) - self._updated_at) > self._limits.stale_after_s

    def snapshot(self, now: float) -> CurveSnapshot:
        """返回曲线状态快照。"""

        return CurveSnapshot(
            outdoor_c=self._outdoor_c,
            supply_c=self._supply_c,
            epoch=self._epoch,
            updated_at=self._updated_at,
            stale=self.is_stale(now),
            validated=self._validated,
            rejected=self._rejected,
            curve_version=self._version.version,
            segmenting=SEGMENTING,
        )
