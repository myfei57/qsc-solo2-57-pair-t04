"""供热曲线版本链、调质绑定与按管段拟合编排。

曲线换版属于**版本变化**，版本按管段各自成链：拟合出的新参数发布为该管段的
下一版，旧版本不可变、永不就地改写。换代以后新下发的调质按新版插值；已经
下发的调质在下发那一刻就把版本号与目标供温固化，换代不追溯、不重算。

某根管段的拟合数据不够时按该管段登记的兜底策略处理（``inherit`` 沿用上一版、
``empirical`` 用经验曲线发一版兜底版），策略由调用方在装配期按管段定，不随
每轮数据改口径。采样丢点是部分失败：逐点剔除、记审计，绝不参与拟合。

拟合结果（:class:`~heatnet.temp.fitting.FitResult`）、版本里落的分段与审计
事件载荷都从同一份逐段差异对象折出，保证"拟合、曲线、审计"三处口径一致。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ..config import CurveFitLimits, CurveLimits
from ..eventbus import Event, EventBus
from ..store.jsonfile import JsonStore
from .fitting import (
    EMPIRICAL,
    FITTED,
    INHERIT,
    FitResult,
    clean_samples,
    diff_segments,
    empirical_supplies,
    fit_curve,
)
from .segments import CurveSample, SegmentGrid

SEED_VERSION = 1


@dataclass(frozen=True)
class CurveVersion:
    """一版不可变的管段供热曲线。"""

    pipe: str
    version: int
    source: str
    created_ts: float
    grid_basis: str
    knot_supply_c: tuple[float, ...]
    changed_segments: tuple[int, ...]
    segment_rmse_c: tuple[float, ...]
    origin: str = ""
    note: str = ""

    def interpolate(self, grid: SegmentGrid, outdoor_c: float) -> float:
        """按本版曲线插值出目标供温。"""

        return grid.interpolate(outdoor_c, self.knot_supply_c)

    def segments(self, grid: SegmentGrid) -> list[dict]:
        """本版的分段端点供温，供控制台展示。"""

        return grid.segment_values(self.knot_supply_c)

    def as_dict(self) -> dict:
        return {
            "pipe": self.pipe,
            "version": self.version,
            "source": self.source,
            "created_ts": round(self.created_ts, 3),
            "grid_basis": self.grid_basis,
            "knot_supply_c": [round(value, 2) for value in self.knot_supply_c],
            "changed_segments": list(self.changed_segments),
            "segment_rmse_c": [round(value, 3) for value in self.segment_rmse_c],
            "origin": self.origin,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "CurveVersion":
        return cls(
            pipe=str(raw.get("pipe", "")),
            version=int(raw.get("version", SEED_VERSION)),
            source=str(raw.get("source", EMPIRICAL)),
            created_ts=float(raw.get("created_ts", 0.0)),
            grid_basis=str(raw.get("grid_basis", "")),
            knot_supply_c=tuple(float(value) for value in raw.get("knot_supply_c", ())),
            changed_segments=tuple(int(value) for value in raw.get("changed_segments", ())),
            segment_rmse_c=tuple(float(value) for value in raw.get("segment_rmse_c", ())),
            origin=str(raw.get("origin", "")),
            note=str(raw.get("note", "")),
        )


@dataclass(frozen=True)
class QualityOrder:
    """一次已下发调质：下发时固化版本号与目标供温。"""

    order_id: str
    pipe: str
    outdoor_c: float
    target_supply_c: float
    curve_version: int
    issued_ts: float
    origin: str

    def as_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "pipe": self.pipe,
            "outdoor_c": round(self.outdoor_c, 2),
            "target_supply_c": round(self.target_supply_c, 2),
            "curve_version": self.curve_version,
            "issued_ts": round(self.issued_ts, 3),
            "origin": self.origin,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "QualityOrder":
        return cls(
            order_id=str(raw.get("order_id", "")),
            pipe=str(raw.get("pipe", "")),
            outdoor_c=float(raw.get("outdoor_c", 0.0)),
            target_supply_c=float(raw.get("target_supply_c", 0.0)),
            curve_version=int(raw.get("curve_version", 0)),
            issued_ts=float(raw.get("issued_ts", 0.0)),
            origin=str(raw.get("origin", "")),
        )


class CurveCatalog:
    """按管段保存曲线版本链与已下发调质，版本落盘、重启可回放。"""

    def __init__(
        self,
        grid: SegmentGrid,
        store: JsonStore,
        policies: dict[str, str] | None = None,
        now: float = 0.0,
    ) -> None:
        self._grid = grid
        self._store = store
        self._lock = threading.RLock()
        self._versions: dict[str, list[CurveVersion]] = {}
        self._orders: dict[str, list[QualityOrder]] = {}
        self._policies: dict[str, str] = {}
        self._rounds = 0
        for pipe, policy in (policies or {}).items():
            self.register_pipe(pipe, policy)
            self._ensure_seed(pipe, now)
        self._restore()

    @property
    def grid(self) -> SegmentGrid:
        return self._grid

    def _doc_id(self, pipe: str) -> str:
        return f"curve_catalog/{pipe}"

    def register_pipe(self, pipe: str, fallback: str = INHERIT) -> str:
        """登记一根管段的数据不足兜底策略，重名时策略必须一致。"""

        if fallback not in (INHERIT, EMPIRICAL):
            raise ValueError("兜底策略必须是 inherit 或 empirical")
        with self._lock:
            existing = self._policies.get(pipe)
            if existing is not None and existing != fallback:
                raise ValueError(f"管段 {pipe} 兜底策略已固定为 {existing}")
            self._pipes_set(pipe, fallback)
            return fallback

    def _pipes_set(self, pipe: str, fallback: str) -> None:
        self._policies[pipe] = fallback
        self._versions.setdefault(pipe, [])
        self._orders.setdefault(pipe, [])

    def _ensure_seed(self, pipe: str, now: float) -> CurveVersion:
        versions = self._versions.setdefault(pipe, [])
        if versions:
            return versions[-1]
        seed = CurveVersion(
            pipe=pipe,
            version=SEED_VERSION,
            source=EMPIRICAL,
            created_ts=float(now),
            grid_basis=self._grid.segments()[0]["basis"],
            knot_supply_c=empirical_supplies(self._grid),
            changed_segments=tuple(range(self._grid.segment_count)),
            segment_rmse_c=tuple(0.0 for _ in range(self._grid.segment_count)),
            origin="seed",
            note="初始经验曲线",
        )
        versions.append(seed)
        return seed

    def _restore(self) -> None:
        """从落盘文档回放每根管段的版本链与已下发调质。"""

        for doc_id in self._store.list_ids():
            if not doc_id.startswith("curve_catalog/"):
                continue
            pipe = doc_id.split("/", 1)[1]
            raw = self._store.read(doc_id)
            if not isinstance(raw, dict):
                continue
            self._pipes_set(pipe, str(raw.get("fallback", INHERIT)))
            versions = [CurveVersion.from_dict(item) for item in raw.get("versions", ())]
            orders = [QualityOrder.from_dict(item) for item in raw.get("orders", ())]
            if versions:
                self._versions[pipe] = versions
            # 文档缺版本块时保留装配期播下的种子版，不按当前配置重算，
            # 否则配置一旦改过，重启会把历史曲线悄悄换成另一条。
            self._orders[pipe] = orders
            self._rounds = max(self._rounds, int(raw.get("rounds", 0)))

    def _persist(self, pipe: str) -> None:
        versions = self._versions[pipe]
        self._store.write(
            self._doc_id(pipe),
            {
                "pipe": pipe,
                "fallback": self._policies[pipe],
                "rounds": self._rounds,
                "versions": [version.as_dict() for version in versions],
                "orders": [order.as_dict() for order in self._orders[pipe]],
            },
        )

    def pipes(self) -> list[str]:
        """返回已登记管段（排序后）。"""

        with self._lock:
            return sorted(self._versions)

    def policy(self, pipe: str) -> str:
        """返回某管段的数据不足兜底策略。"""

        return self._policies[pipe]

    def ensure_pipe(self, pipe: str, fallback: str, now: float) -> CurveVersion:
        """取管段当前版；未登记时按给定策略登记并播经验种子版。"""

        with self._lock:
            if pipe not in self._policies:
                self.register_pipe(pipe, fallback)
            seed = self._ensure_seed(pipe, now)
            self._persist(pipe)
            return seed

    def current_version(self, pipe: str) -> CurveVersion:
        """返回某管段的当前（最新）版本。"""

        with self._lock:
            versions = self._versions.get(pipe)
            if not versions:
                raise KeyError(pipe)
            return versions[-1]

    def version(self, pipe: str, number: int) -> CurveVersion:
        """按版本号取某管段的历史版本。"""

        with self._lock:
            for item in self._versions.get(pipe, ()):
                if item.version == number:
                    return item
            raise KeyError(number)

    def versions(self, pipe: str) -> list[CurveVersion]:
        """返回某管段的全部版本（按版本升序）。"""

        with self._lock:
            return list(self._versions.get(pipe, ()))

    def rounds(self) -> int:
        """返回累计拟合轮次。"""

        return self._rounds

    def publish_fitted(
        self,
        pipe: str,
        result: FitResult,
        now: float,
        origin: str,
        tolerance_c: float,
    ) -> CurveVersion:
        """把拟合结果发布为该管段下一版，改动段由新旧逐段对比得出。"""

        with self._lock:
            previous = self.current_version(pipe)
            changes = diff_segments(self._grid, previous.knot_supply_c, result.supplies, tolerance_c)
            version = CurveVersion(
                pipe=pipe,
                version=previous.version + 1,
                source=FITTED,
                created_ts=float(now),
                grid_basis=self._grid.segments()[0]["basis"],
                knot_supply_c=result.supplies,
                changed_segments=tuple(change.index for change in changes if change.changed),
                segment_rmse_c=result.segment_rmse,
                origin=origin,
                note="",
            )
            self._versions[pipe].append(version)
            self._persist(pipe)
            return version

    def publish_empirical_fallback(self, pipe: str, now: float, origin: str) -> CurveVersion:
        """数据不足且策略为经验兜底时，发一版经验曲线。"""

        with self._lock:
            previous = self.current_version(pipe)
            supplies = empirical_supplies(self._grid)
            changes = diff_segments(self._grid, previous.knot_supply_c, supplies, 0.0)
            version = CurveVersion(
                pipe=pipe,
                version=previous.version + 1,
                source=EMPIRICAL,
                created_ts=float(now),
                grid_basis=self._grid.segments()[0]["basis"],
                knot_supply_c=supplies,
                changed_segments=tuple(change.index for change in changes if change.changed),
                segment_rmse_c=tuple(0.0 for _ in range(self._grid.segment_count)),
                origin=origin,
                note="拟合数据不足，按经验曲线兜底",
            )
            self._versions[pipe].append(version)
            self._persist(pipe)
            return version

    def issue_order(self, pipe: str, outdoor_c: float, order_id: str, now: float, origin: str) -> QualityOrder:
        """按管段当前版下发一次调质，版本号与目标供温在这一刻固化。"""

        with self._lock:
            current = self.current_version(pipe)
            order = QualityOrder(
                order_id=order_id,
                pipe=pipe,
                outdoor_c=float(outdoor_c),
                target_supply_c=current.interpolate(self._grid, outdoor_c),
                curve_version=current.version,
                issued_ts=float(now),
                origin=origin,
            )
            self._orders[pipe].append(order)
            self._persist(pipe)
            return order

    def orders(self, pipe: str) -> list[QualityOrder]:
        """返回某管段已下发调质（按下发顺序）。"""

        with self._lock:
            return list(self._orders.get(pipe, ()))

    def mark_round(self) -> int:
        """记一轮拟合并落盘全部管段文档。"""

        with self._lock:
            self._rounds += 1
            for pipe in self.pipes():
                self._persist(pipe)
            return self._rounds

    def describe(self) -> dict:
        """汇总版本链，供控制台与状态快照读取。"""

        with self._lock:
            return {
                "grid": self._grid.describe(),
                "rounds": self._rounds,
                "pipes": [
                    {
                        "pipe": pipe,
                        "fallback": self._policies[pipe],
                        "current_version": self.current_version(pipe).version,
                        "versions": len(self.versions(pipe)),
                        "orders_issued": len(self.orders(pipe)),
                    }
                    for pipe in self.pipes()
                ],
            }


class CurveTuningService:
    """按轮次对各管段跑分段拟合并发布版本，事件直接进审计流水。"""

    def __init__(
        self,
        catalog: CurveCatalog,
        limits: CurveLimits,
        fit_limits: CurveFitLimits,
        bus: EventBus,
        default_fallback: str = INHERIT,
    ) -> None:
        self._catalog = catalog
        self._limits = limits
        self._fit_limits = fit_limits
        self._bus = bus
        self._default_fallback = default_fallback

    def refit_round(
        self,
        samples_by_pipe: dict[str, list[CurveSample]],
        now: float,
        origin: str = "runtime",
    ) -> dict:
        """对每根管段清洗、拟合、按策略兜底并发布，返回整轮报告。"""

        catalog = self._catalog
        grid = catalog.grid
        round_number = catalog.rounds() + 1
        round_id = f"curve-fit-{round_number}"
        pipe_reports: dict[str, dict] = {}
        dropped_all: list[dict] = []

        for pipe in sorted(samples_by_pipe):
            catalog.ensure_pipe(pipe, self._default_fallback, now)
            policy = catalog.policy(pipe)
            previous = catalog.current_version(pipe)
            accepted, dropped = clean_samples(list(samples_by_pipe[pipe]), grid)
            for item in dropped:
                dropped_all.append({"pipe": pipe, **item})
            result = fit_curve(grid, accepted, self._fit_limits)

            if dropped:
                self._bus.publish(
                    Event.of(
                        "temp",
                        "temp",
                        "curve_sample_dropped",
                        asset=pipe,
                        severity="warning",
                        message=f"管段 {pipe} 本轮采样丢点，已逐点剔除不参与拟合",
                        round_id=round_id,
                        pipe=pipe,
                        dropped=dropped,
                        dropped_count=len(dropped),
                        accepted_count=len(accepted),
                    )
                )

            if not result.covered:
                if policy == EMPIRICAL:
                    published = catalog.publish_empirical_fallback(pipe, now, origin)
                    action = "curve_fallback_empirical"
                    message = f"管段 {pipe} 拟合数据不足，按经验曲线兜底发布 v{published.version}"
                else:
                    published = previous
                    action = "curve_fallback_inherit"
                    message = f"管段 {pipe} 拟合数据不足，沿用上一版 v{previous.version}"
                self._bus.publish(
                    Event.of(
                        "temp",
                        "temp",
                        action,
                        asset=pipe,
                        severity="warning",
                        message=message,
                        round_id=round_id,
                        pipe=pipe,
                        policy=policy,
                        version=published.version,
                        source=published.source,
                        fit=result.as_dict(),
                        segments=published.segments(grid),
                    )
                )
                pipe_reports[pipe] = {
                    "pipe": pipe,
                    "published": policy == EMPIRICAL,
                    "policy": policy,
                    "previous_version": previous.version,
                    "version": published.version,
                    "source": published.source,
                    "changed_segments": list(published.changed_segments),
                    "segments": published.segments(grid),
                    "fit": result.as_dict(),
                    "accepted_samples": len(accepted),
                    "dropped_samples": len(dropped),
                }
                continue

            published = catalog.publish_fitted(pipe, result, now, origin, self._fit_limits.change_tolerance_c)
            changes = diff_segments(grid, previous.knot_supply_c, published.knot_supply_c, self._fit_limits.change_tolerance_c)
            self._bus.publish(
                Event.of(
                    "temp",
                    "temp",
                    "curve_version_published",
                    asset=pipe,
                    severity="notice",
                    message=(
                        f"管段 {pipe} 供热曲线换代 v{previous.version}→v{published.version}，"
                        f"改动段 {list(published.changed_segments) or '无'}"
                    ),
                    round_id=round_id,
                    pipe=pipe,
                    previous_version=previous.version,
                    version=published.version,
                    changed_segments=list(published.changed_segments),
                    changes=[change.as_dict() for change in changes],
                    fit=result.as_dict(),
                    segments=published.segments(grid),
                )
            )
            pipe_reports[pipe] = {
                "pipe": pipe,
                "published": True,
                "policy": policy,
                "previous_version": previous.version,
                "version": published.version,
                "source": FITTED,
                "changed_segments": list(published.changed_segments),
                "changes": [change.as_dict() for change in changes],
                "segments": published.segments(grid),
                "fit": result.as_dict(),
                "accepted_samples": len(accepted),
                "dropped_samples": len(dropped),
            }

        catalog.mark_round()
        report = {
            "round_id": round_id,
            "round": round_number,
            "ts": round(float(now), 3),
            "origin": origin,
            "grid": grid.describe(),
            "dropped": dropped_all,
            "dropped_count": len(dropped_all),
            "pipes": pipe_reports,
            "ok": True,
        }
        self._bus.publish(
            Event.of(
                "temp",
                "temp",
                "curve_refit_round",
                severity="notice",
                message=f"供热曲线第 {round_number} 轮分段拟合完成",
                round_id=round_id,
                grid=grid.describe(),
                pipes=list(pipe_reports),
                dropped_count=len(dropped_all),
            )
        )
        return report
