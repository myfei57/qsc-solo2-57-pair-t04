"""供热曲线分段拟合、版本换代、兜底策略与审计一致性。"""

from __future__ import annotations

import pytest

from heatnet.app import build_platform
from heatnet.config import ConfigError, config_to_mapping, default_config
from heatnet.console.api import Api
from heatnet.errors import NamespaceError
from heatnet.eventbus import EventBus
from heatnet.store.jsonfile import JsonStore
from heatnet.temp.fitting import EMPIRICAL, INHERIT, clean_samples, empirical_supplies, fit_curve
from heatnet.temp.segments import SEGMENT_BASIS, CurveSample, SegmentGrid
from heatnet.temp.tuning import CurveCatalog, CurveTuningService


def _samples(grid, supplies, count=80, jitter=0.0):
    """在整个室外温度区间上均匀造一批落在真实曲线上的采样。"""

    lo, hi = grid.limits.outdoor_min_c, grid.limits.outdoor_max_c
    points = []
    for index in range(count):
        outdoor = lo + (hi - lo) * index / (count - 1)
        supply = grid.interpolate(outdoor, supplies) + jitter * ((index % 3) - 1)
        points.append(CurveSample(ts=float(index), outdoor_c=outdoor, supply_c=supply))
    return points


def _service(tmp_path, clock, policy=INHERIT, pipe="p1"):
    cfg = default_config()
    grid = SegmentGrid(cfg.curve, cfg.curve_fit.segment_count)
    store = JsonStore(str(tmp_path / "curves"))
    catalog = CurveCatalog(grid, store, policies={pipe: policy}, now=clock.now())
    service = CurveTuningService(catalog, cfg.curve, cfg.curve_fit, EventBus(), default_fallback=policy)
    return cfg, grid, catalog, service


# ---- 分段方式：定死为等室外温度 --------------------------------------


def test_grid_is_equal_outdoor_and_fixed():
    cfg = default_config()
    grid = SegmentGrid(cfg.curve, cfg.curve_fit.segment_count)
    assert grid.segment_count == 4
    assert grid.knots() == pytest.approx([-26.0, -15.0, -4.0, 7.0, 18.0])
    assert all(segment["basis"] == SEGMENT_BASIS == "equal_outdoor" for segment in grid.segments())
    widths = [
        segment["outdoor_max_c"] - segment["outdoor_min_c"] for segment in grid.segments()
    ]
    assert widths == pytest.approx([11.0, 11.0, 11.0, 11.0])
    assert grid.segment_index(-26.0) == 0
    assert grid.segment_index(18.0) == 3
    weights = grid.weights(-15.0)
    assert weights[1] == pytest.approx(1.0)
    assert sum(weights) == pytest.approx(1.0)


def test_grid_knots_do_not_drift_between_rounds(platform):
    samples = _samples(platform.curve_grid, empirical_supplies(platform.curve_grid))
    first = platform.refit_curves({"north": samples})
    second = platform.refit_curves({"north": samples})
    assert first["grid"]["knots_outdoor_c"] == second["grid"]["knots_outdoor_c"]
    assert first["grid"]["basis"] == "equal_outdoor"


# ---- 采样丢点：部分失败，逐点剔除、绝不进拟合 --------------------------


def test_dropped_and_out_of_range_samples_never_enter_fit(platform):
    grid = platform.curve_grid
    clean = _samples(grid, (95, 86, 78, 68, 55), count=80)
    accepted, dropped = clean_samples(clean, grid)
    assert len(dropped) == 0 and len(accepted) == 80

    polluted = clean + [
        CurveSample(ts=9001, outdoor_c=-10.0, supply_c=9999.0, ok=False, reason="gap"),
        CurveSample(ts=9002, outdoor_c=60.0, supply_c=60.0, ok=True),
    ]
    accepted, dropped = clean_samples(polluted, grid)
    assert len(accepted) == 80 and len(dropped) == 2
    assert {item["reason"] for item in dropped} == {"gap", "outdoor_out_of_range"}

    limits = platform.config.curve_fit
    good = fit_curve(grid, clean, limits)
    with_bad = fit_curve(grid, accepted, limits)
    assert good.supplies == pytest.approx(with_bad.supplies, abs=1e-6)


def test_malformed_sample_payload_is_partial_failure(platform):
    report = platform.refit_curves(
        {"north": [
            {"ts": 1.0, "outdoor_c": "bad", "supply_c": 90.0},
            {"ts": 2.0, "supply_c": 90.0},
        ]}
    )
    assert report["dropped_count"] == 2
    assert report["pipes"]["north"]["published"] is False
    warnings = [item for item in platform.audit.entries(limit=50) if item["action"] == "curve_sample_dropped"]
    assert warnings and warnings[0]["severity"] == "warning"
    assert warnings[0]["payload"]["dropped_count"] == 2
    assert warnings[0]["payload"]["accepted_count"] == 0


# ---- 拟合换代与逐段改动 ----------------------------------------------


def test_fit_publishes_version_and_names_changed_segments(platform):
    # 冷端两段不变，暖端两段整体抬高 4℃。
    samples = _samples(platform.curve_grid, (95, 85, 79, 69, 55), jitter=0.2)
    report = platform.refit_curves({"north": samples}, origin="winter-batch")
    pipe = report["pipes"]["north"]
    assert pipe["published"] is True
    assert pipe["previous_version"] == 1 and pipe["version"] == 2
    assert pipe["source"] == "fitted"
    assert pipe["changed_segments"] == [1, 2, 3]
    unchanged, changed = pipe["changes"][0], pipe["changes"][1]
    assert unchanged["changed"] is False and changed["changed"] is True
    assert changed["new_supply_right_c"] - changed["old_supply_right_c"] == pytest.approx(4.0, abs=0.6)
    assert pipe["fit"]["segment_samples"] == [20, 20, 20, 20]
    assert all(rmse < 0.3 for rmse in pipe["fit"]["segment_rmse_c"])


def test_insufficient_samples_inherit_keeps_previous_version(platform):
    report = platform.refit_curves(
        {"south": [CurveSample(ts=1.0, outdoor_c=-10.0, supply_c=90.0)]}
    )
    pipe = report["pipes"]["south"]
    assert pipe["published"] is False
    assert pipe["policy"] == "inherit"
    assert pipe["version"] == pipe["previous_version"] == 1
    assert platform.curve_catalog.current_version("south").version == 1
    entries = [item for item in platform.audit.entries(limit=20) if item["action"] == "curve_fallback_inherit"]
    assert len(entries) == 1 and entries[0]["severity"] == "warning"


def test_uncovered_single_segment_triggers_pipe_level_fallback(platform):
    # 总点数够但全挤在冷段：其它段没有覆盖，整管段不得只换几段。
    samples = [
        CurveSample(ts=float(i), outdoor_c=-25.0 + i * 0.05, supply_c=94.0)
        for i in range(40)
    ]
    report = platform.refit_curves({"east": samples})
    pipe = report["pipes"]["east"]
    assert pipe["published"] is False and pipe["policy"] == "inherit"
    assert pipe["fit"]["segment_samples"][0] == 40
    assert pipe["fit"]["covered"] is False


def test_empirical_fallback_policy_publishes_new_version(tmp_path, clock):
    cfg, grid, catalog, service = _service(tmp_path, clock, policy=EMPIRICAL)
    report = service.refit_round({"p1": [CurveSample(1.0, -10.0, 90.0)]}, clock.now())
    pipe = report["pipes"]["p1"]
    assert pipe["published"] is True and pipe["policy"] == EMPIRICAL
    assert pipe["version"] == 2 and pipe["source"] == EMPIRICAL
    current = catalog.current_version("p1")
    assert current.version == 2 and current.source == EMPIRICAL
    assert current.knot_supply_c == pytest.approx(empirical_supplies(grid), abs=1e-9)


def test_policy_is_fixed_per_pipe(tmp_path, clock):
    _cfg, _grid, catalog, _svc = _service(tmp_path, clock, policy=INHERIT)
    assert catalog.policy("p1") == INHERIT
    with pytest.raises(ValueError):
        catalog.register_pipe("p1", EMPIRICAL)
    with pytest.raises(ValueError):
        catalog.register_pipe("p2", "per-round-whim")


# ---- 版本绑定：换代后新指令按新版、已下发按老版 ------------------------


def test_issued_quality_stays_bound_to_old_version(platform):
    samples_v2 = _samples(platform.curve_grid, (95, 85, 79, 69, 55))
    platform.refit_curves({"north": samples_v2})
    order = platform.issue_quality("north", -20.0, origin="dispatch-a")
    assert order["curve_version"] == 2
    target_old = order["target_supply_c"]

    samples_v3 = _samples(platform.curve_grid, (99, 89, 79, 69, 55))
    platform.refit_curves({"north": samples_v3})
    rebound = platform.quality_order(order["order_id"])
    assert rebound["curve_version"] == 2
    assert rebound["target_supply_c"] == target_old
    assert rebound["current_version"] == 3
    assert rebound["retroactive"] is False

    new_order = platform.issue_quality("north", -20.0, origin="dispatch-b")
    assert new_order["curve_version"] == 3
    assert new_order["target_supply_c"] > target_old


def test_issue_quality_rejects_unknown_pipe(platform):
    with pytest.raises(NamespaceError):
        platform.issue_quality("branch-unknown", -5.0)


# ---- 控制台展示与审计一致性 -------------------------------------------


def test_console_segments_versions_refit_and_issue(platform):
    api = Api(platform)
    samples = _samples(platform.curve_grid, (95, 85, 79, 69, 55))
    status, payload = api.handle(
        "POST", "/api/curve/refit", {}, {"origin": "console", "samples": {"north": [s.as_dict() for s in samples]}}
    )
    assert status == 200
    assert payload["pipes"]["north"]["version"] == 2

    status, segments = api.handle("GET", "/api/curve/segments", {"pipe": "north"}, {})
    assert status == 200
    assert segments["grid"]["basis"] == "equal_outdoor"
    assert len(segments["pipes"]["north"]["segments"]) == 4

    status, versions = api.handle("GET", "/api/curve/versions/north", {}, {})
    assert status == 200
    assert [item["version"] for item in versions["versions"]] == [1, 2]
    assert versions["versions"][-1]["segments"] == segments["pipes"]["north"]["segments"]

    status, issued = api.handle("POST", "/api/curve/issue", {}, {"pipe": "north", "outdoor_c": -12.0})
    assert status == 200 and issued["curve_version"] == 2
    status, fetched = api.handle("GET", f"/api/curve/orders/{issued['order_id']}", {}, {})
    assert status == 200
    assert fetched["target_supply_c"] == issued["target_supply_c"]

    status, bad = api.handle("POST", "/api/curve/refit", {}, {"samples": []})
    assert status == 409 and bad["component"] == "console"


def test_fit_curve_and_audit_share_one_segment_payload(platform):
    samples = _samples(platform.curve_grid, (95, 85, 79, 69, 55)) + [
        CurveSample(ts=9000.0, outdoor_c=-5.0, supply_c=0.0, ok=False, reason="gap")
    ]
    platform.refit_curves({"north": samples})
    published = [
        item for item in platform.audit.entries(limit=50) if item["action"] == "curve_version_published"
    ]
    assert len(published) == 1
    record = published[0]
    assert record["payload"]["version"] == 2
    assert record["payload"]["changed_segments"] == [1, 2, 3]
    live = platform.curve_segments("north")["pipes"]["north"]["segments"]
    assert record["payload"]["segments"] == live
    rounds = [item for item in platform.audit.entries(limit=50) if item["action"] == "curve_refit_round"]
    assert rounds and rounds[0]["payload"]["grid"]["basis"] == "equal_outdoor"


def test_status_includes_curve_tuning_catalog(platform):
    platform.refit_curves({"north": _samples(platform.curve_grid, empirical_supplies(platform.curve_grid))})
    status = platform.status()
    pipes = {item["pipe"]: item for item in status["curve_tuning"]["pipes"]}
    assert pipes["north"]["current_version"] == 2
    assert pipes["south"]["fallback"] == "inherit"


# ---- 持久化：跨重启版本链与已下发绑定不丢 ------------------------------


def test_versions_and_orders_survive_restart(platform, tmp_path, clock):
    platform.refit_curves({"north": _samples(platform.curve_grid, (95, 85, 79, 69, 55))})
    order = platform.issue_quality("north", -18.0)

    rebuilt = build_platform(  # 同存储目录重建平台
        config=platform.config,
        clock=clock,
        storage_root=platform.store.root,
    )
    versions = rebuilt.curve_catalog.versions("north")
    assert [item.version for item in versions] == [1, 2]
    assert rebuilt.curve_catalog.rounds() == 1
    rebound = rebuilt.quality_order(order["order_id"])
    assert rebound["curve_version"] == 2
    assert rebound["target_supply_c"] == order["target_supply_c"]


# ---- 配置：分段网格与门限在装配期定死 ----------------------------------


def test_curve_fit_config_roundtrips_and_validates():
    cfg = default_config()
    restored = type(cfg).from_mapping(config_to_mapping(cfg))
    assert restored.curve_fit == cfg.curve_fit
    assert restored == cfg

    mapping = config_to_mapping(cfg)
    mapping["limits"]["curve_fit"]["min_samples_total"] = 2
    with pytest.raises(ConfigError):
        type(cfg).from_mapping(mapping)
