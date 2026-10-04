"""曲线分段拟合、换代版本与调质单冻结。

这些用例钉死几条交付约定：

* 分段口径定死为**等室外温度**四段五断点，不允许按等负荷；
* 采样丢点是部分失败——只计数、不参与拟合，这一轮照样换代；一个可用点都没有
  才算整轮失败，老版本原样保留；
* 数据不够的管段按管段自己的兜底策略取值（沿用上一版 / 经验线）；
* 曲线换代是版本变化：换代后新调质按新版算，已下发的调质单按冻结的老版算；
* 拟合结果、控制台分段与审计落的是同一份版本信息。
"""

from __future__ import annotations

import pytest

from heatnet.config import default_config
from heatnet.errors import CurveFitError, ForecastError
from heatnet.temp.curve import HeatCurve
from heatnet.temp.fitting import (
    CURVE_SEGMENTS,
    EMPIRICAL,
    REUSE_PREVIOUS,
    SEGMENTING,
    empirical_supply,
    knot_outdoor_temps,
    normalize_samples,
)


@pytest.fixture()
def limits():
    return default_config().curve


@pytest.fixture()
def curve(limits):
    return HeatCurve(limits)


def _samples(limits, count_per_segment=6, *, cold_shift=0.25):
    """沿整段室外温度区间造样本，真实供温相对经验线冷抬暖压。"""

    samples = []
    total = count_per_segment * CURVE_SEGMENTS
    for index in range(total):
        outdoor = limits.outdoor_min_c + index * (limits.outdoor_max_c - limits.outdoor_min_c) / (
            total - 1
        )
        samples.append({"outdoor_c": outdoor, "supply_c": _actual_supply(limits, outdoor, cold_shift)})
    return samples


def _actual_supply(limits, outdoor_c, cold_shift):
    span = (outdoor_c - limits.outdoor_min_c) / (limits.outdoor_max_c - limits.outdoor_min_c)
    empirical = limits.supply_high_c + (limits.supply_low_c - limits.supply_high_c) * span
    return empirical - cold_shift * (outdoor_c + 4.0)


# ---- 分段口径 ---------------------------------------------------------


def test_segmentation_is_fixed_equal_outdoor_four_segments(limits):
    knots = knot_outdoor_temps(limits)
    assert len(knots) == CURVE_SEGMENTS + 1
    assert knots[0] == pytest.approx(limits.outdoor_min_c)
    assert knots[-1] == pytest.approx(limits.outdoor_max_c)
    step = (limits.outdoor_max_c - limits.outdoor_min_c) / CURVE_SEGMENTS
    for index, knot in enumerate(knots):
        assert knot == pytest.approx(limits.outdoor_min_c + step * index)
    assert SEGMENTING == "equal_outdoor"


# ---- 拟合与换代 -------------------------------------------------------


def test_refit_moves_cold_side_up_and_warm_side_down(limits, curve):
    samples, dropped = normalize_samples(limits, _samples(limits))
    assert dropped == {
        "missing": 0,
        "not_number": 0,
        "outdoor_out_of_range": 0,
        "supply_invalid": 0,
    }
    old_table = curve.table
    result = curve.refit(samples, dropped, 1000.0)

    assert curve.version() == 2
    assert result.fitted_segments == 4
    assert result.fallback_segments == 0
    new_table = curve.table
    # 冷的时候老曲线供温偏低 -> 新版内部冷断点抬高；暖的时候老曲线偏高 -> 压低。
    assert new_table.interpolate(-15.0) > old_table.interpolate(-15.0)
    assert new_table.interpolate(7.0) < old_table.interpolate(7.0)
    # 单调：室外越冷供温越高。
    supplies = [new_table.interpolate(knot) for knot in knot_outdoor_temps(limits)]
    assert all(left >= right for left, right in zip(supplies, supplies[1:]))
    # 不越供温硬边界。
    assert limits.supply_low_c <= min(supplies) and max(supplies) <= limits.supply_high_c


def test_refit_is_atomic_when_no_usable_sample(limits, curve):
    samples, dropped = normalize_samples(
        limits,
        [
            {"supply_c": 80.0},
            {"outdoor_c": "x", "supply_c": 80.0},
            {"outdoor_c": 50.0, "supply_c": 80.0},
            {"outdoor_c": -5.0, "supply_c": -2.0},
        ],
    )
    assert samples == []
    assert sum(dropped.values()) == 4
    with pytest.raises(CurveFitError):
        curve.refit(samples, dropped, 1000.0)
    with pytest.raises(CurveFitError):
        curve.refit([], {}, 1000.0)
    assert curve.version() == 1


def test_dropped_samples_are_partial_failure_and_do_not_join_fit(limits, curve):
    raw = _samples(limits, count_per_segment=6)
    raw += [
        {"supply_c": 80.0},
        {"outdoor_c": "oops", "supply_c": 80.0},
        {"outdoor_c": 60.0, "supply_c": 80.0},
        {"outdoor_c": 0.0, "supply_c": -1.0},
    ]
    samples, dropped = normalize_samples(limits, raw)
    result = curve.refit(samples, dropped, 1000.0)

    assert curve.version() == 2
    assert result.used_samples == 24
    assert result.dropped_total == 4
    assert result.dropped == {
        "missing": 1,
        "not_number": 1,
        "outdoor_out_of_range": 1,
        "supply_invalid": 1,
    }
    # 每个采样恰好落进一段，丢点不进任何段。
    assert sum(segment.sample_count for segment in result.segments) == 24


# ---- 兜底策略按管段定 -------------------------------------------------


def test_sparse_segments_fall_back_per_segment(limits, curve):
    # 箱界是相邻断点中点：段0 覆盖 (-inf,-20.5]，段3 覆盖 [12.5,+inf)，
    # 两点都落在同一段才够回归；段1、段2 一个点都没有，必须兜底。
    raw = [
        {"outdoor_c": -23.0, "supply_c": _actual_supply(limits, -23.0, 0.25)},
        {"outdoor_c": -22.0, "supply_c": _actual_supply(limits, -22.0, 0.25)},
        {"outdoor_c": 13.0, "supply_c": _actual_supply(limits, 13.0, 0.25)},
        {"outdoor_c": 14.0, "supply_c": _actual_supply(limits, 14.0, 0.25)},
    ]
    samples, dropped = normalize_samples(limits, raw)
    result = curve.refit(samples, dropped, 1000.0, fallback_policies=REUSE_PREVIOUS)

    methods = [segment.method for segment in result.segments]
    assert methods == ["least_squares", "reuse_previous", "reuse_previous", "least_squares"]
    assert result.fitted_segments == 2
    assert result.fallback_segments == 2
    for segment in result.segments:
        if segment.method == "reuse_previous":
            assert segment.fallback_policy == REUSE_PREVIOUS
            assert segment.sample_count == 0


def test_empirical_fallback_uses_experience_line(limits, curve):
    # 两个点室外温度相同（方差为零）回归不成立 -> 全部管段走经验兜底，
    # 断点值必须正好落在经验线上。
    samples, dropped = normalize_samples(
        limits,
        [
            {"outdoor_c": -10.0, "supply_c": empirical_supply(limits, -10.0)},
            {"outdoor_c": -10.0, "supply_c": empirical_supply(limits, -10.0) + 1.0},
        ],
    )
    result = curve.refit(samples, dropped, 1000.0, fallback_policies=EMPIRICAL)
    assert result.fitted_segments == 0
    for segment in result.segments:
        assert segment.method == "empirical"
        for outdoor, supply in segment.points:
            assert supply == pytest.approx(empirical_supply(limits, outdoor))


def test_mixed_fallback_policies_apply_segment_by_segment(limits, curve):
    raw = [{"outdoor_c": 10.0, "supply_c": 60.0}, {"outdoor_c": 11.0, "supply_c": 61.0}]
    samples, dropped = normalize_samples(limits, raw)
    policies = (REUSE_PREVIOUS, EMPIRICAL, EMPIRICAL, REUSE_PREVIOUS)
    result = curve.refit(samples, dropped, 1000.0, fallback_policies=list(policies))
    assert result.segments[0].fallback_policy == REUSE_PREVIOUS
    assert result.segments[3].fallback_policy == REUSE_PREVIOUS


def test_invalid_fallback_policy_is_rejected(limits, curve):
    samples, dropped = normalize_samples(limits, _samples(limits))
    with pytest.raises(CurveFitError):
        curve.refit(samples, dropped, 1000.0, fallback_policies="by_load")
    with pytest.raises(CurveFitError):
        curve.refit(samples, dropped, 1000.0, fallback_policies=[REUSE_PREVIOUS] * 3)
    assert curve.version() == 1


# ---- 版本与调质单 -----------------------------------------------------


def test_new_orders_use_new_version_existing_orders_keep_old(limits, curve):
    old_order = curve.issue_quality_order(-10.0, 10.0)
    assert old_order.curve_version == 1
    old_target = old_order.target_supply_c

    samples, dropped = normalize_samples(limits, _samples(limits))
    curve.refit(samples, dropped, 20.0)
    assert curve.version() == 2

    applied = curve.resolve_quality_order(old_order.order_id)
    assert applied.status == "applied"
    assert applied.target_supply_c == pytest.approx(old_target)

    new_order = curve.issue_quality_order(-10.0, 30.0)
    assert new_order.curve_version == 2
    assert new_order.target_supply_c != pytest.approx(old_target)


def test_quality_order_can_only_be_resolved_once(limits, curve):
    order = curve.issue_quality_order(-6.0, 1.0)
    curve.resolve_quality_order(order.order_id)
    with pytest.raises(CurveFitError):
        curve.resolve_quality_order(order.order_id)


def test_unknown_order_and_version_are_rejected(limits, curve):
    with pytest.raises(CurveFitError):
        curve.resolve_quality_order("nope")
    with pytest.raises(CurveFitError):
        curve.supply_for_version(9, -6.0)


def test_order_issue_validates_outdoor_range(limits, curve):
    with pytest.raises(ForecastError):
        curve.issue_quality_order(60.0, 1.0)


# ---- 拟合 / 曲线 / 审计口径一致 ---------------------------------------


def test_version_info_is_the_single_source_for_segments(limits, curve):
    samples, dropped = normalize_samples(limits, _samples(limits))
    result = curve.refit(samples, dropped, 1234.0, source="runtime-refit")
    info = curve.version_info()

    assert info.version == 2
    assert info.segmenting == SEGMENTING
    assert info.created_at == 1234.0
    assert info.used_samples == result.used_samples
    # 版本断点即曲线表断点，版本分段即拟合分段。
    assert info.table_points() == curve.table.points()
    assert info.describe_segments() == result.segment_dicts()
    assert [(o, s) for o, s in info.points] == list(result.points)


def test_commissioning_v1_has_three_points_two_segments(limits, curve):
    info = curve.version_info(1)
    assert info.version == 1
    assert len(info.points) == 3
    assert [segment["method"] for segment in info.describe_segments()] == [
        "commissioning",
        "commissioning",
    ]
    assert curve.table.points() == info.table_points()


# ---- 平台编排 / 控制台 / 审计 -----------------------------------------


def _runtime_samples(limits, count=120):
    return [
        {"outdoor_c": limits.outdoor_min_c + i * (limits.outdoor_max_c - limits.outdoor_min_c) / (count - 1),
         "supply_c": _actual_supply(
             limits,
             limits.outdoor_min_c + i * (limits.outdoor_max_c - limits.outdoor_min_c) / (count - 1),
             0.25,
         )}
        for i in range(count)
    ]


def test_platform_refit_publishes_audit_and_serves_segments(platform):
    limits = platform.config.curve
    report = platform.refit_curve(_runtime_samples(limits))

    assert report["ok"] is True
    assert report["version"] == 2
    assert report["segmenting"] == "equal_outdoor"
    assert len(report["segments"]) == CURVE_SEGMENTS

    # 曲线、报告、审计是同一份版本信息。
    info = platform.curve.version_info()
    assert report["segments"] == info.describe_segments()
    status_curve = platform.curve_report()
    assert status_curve["segments"] == info.describe_segments()
    assert platform.curve.table.points() == info.table_points()

    entries = [item for item in platform.audit.records() if item.action == "curve_refit"]
    assert entries
    payload = entries[-1].payload
    assert payload["version"] == 2
    assert payload["segments"] == info.describe_segments()
    assert entries[-1].severity == "notice"  # 这一轮没有丢点


def test_platform_dropped_samples_are_warning_but_version_still_advances(platform):
    limits = platform.config.curve
    samples = _runtime_samples(limits) + [{"outdoor_c": 60.0, "supply_c": 80.0}]
    report = platform.refit_curve(samples)

    assert report["version"] == 2
    assert report["dropped_total"] == 1
    entries = [item for item in platform.audit.records() if item.action == "curve_refit"]
    assert entries[-1].severity == "warning"


def test_platform_failed_refit_keeps_old_version_and_raises_alarm(platform):
    from heatnet.errors import CurveFitError

    with pytest.raises(CurveFitError):
        platform.refit_curve([{"outdoor_c": 60.0, "supply_c": 80.0}])
    assert platform.curve.version() == 1
    rejected = [item for item in platform.audit.records() if item.action == "curve_refit_rejected"]
    assert rejected[-1].severity == "alarm"


def test_platform_orders_freeze_version_across_refit(platform):
    limits = platform.config.curve
    old_order = platform.issue_quality_order(-10.0, "op-old")
    assert old_order["curve_version"] == 1

    platform.refit_curve(_runtime_samples(limits))

    applied = platform.apply_quality_order(old_order["order_id"])
    assert applied["curve_version"] == 1
    assert applied["target_supply_c"] == pytest.approx(old_order["target_supply_c"])

    listing = platform.quality_orders()["orders"]
    assert [item["order_id"] for item in listing] == [old_order["order_id"]]
    assert listing[0]["status"] == "applied"

    preview = platform.curve_supply(1, -10.0)
    assert preview["supply_c"] == pytest.approx(old_order["target_supply_c"])


def test_console_endpoints_cover_refit_and_orders(platform):
    from heatnet.console.api import Api

    limits = platform.config.curve
    api = Api(platform)

    status, curve = api.handle("GET", "/api/curve", {}, {})
    assert status == 200
    assert curve["segmenting"] == "equal_outdoor"
    assert curve["curve_version"] == 1

    status, refit = api.handle("POST", "/api/curve/refit", {}, {"samples": _runtime_samples(limits)})
    assert status == 200
    assert refit["version"] == 2
    assert len(refit["segments"]) == 4

    status, order = api.handle("POST", "/api/curve/order", {}, {"outdoor_c": -12.0})
    assert status == 200 and order["curve_version"] == 2

    status, supply = api.handle("GET", "/api/curve/supply", {"version": "1", "outdoor_c": "-12"}, {})
    assert status == 200 and supply["version"] == 1

    status, applied = api.handle("POST", "/api/curve/order/apply", {}, {"order_id": order["order_id"]})
    assert status == 200 and applied["status"] == "applied"

    status, cancelled = api.handle(
        "POST", "/api/curve/order/cancel", {}, {"order_id": order["order_id"]}
    )
    # 已应用的单子不能再撤销。
    assert status == 409
    assert cancelled["component"] == "temp"

    status, orders = api.handle("GET", "/api/curve/orders", {}, {})
    assert status == 200 and len(orders["orders"]) == 1

