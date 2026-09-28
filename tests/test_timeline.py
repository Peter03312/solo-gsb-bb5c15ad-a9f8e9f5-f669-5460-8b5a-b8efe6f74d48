"""时间线合并与预算算法单元测试。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.timeline import (
    DEVIATION_OVER,
    DEVIATION_UNDER,
    DeviationAnomaly,
    Segment,
    attribute_exposure,
    clip_segments,
    coverage_gaps,
    first_exceed_time,
    integrate,
    integrate_within,
    merge_intervals,
    reconcile_illuminance,
    simplify_intervals,
    timelines_equal,
    year_bounds,
)


def _dt(h: float, day: int = 1) -> datetime:
    return datetime(2026, 3, day, tzinfo=timezone.utc) + timedelta(hours=h)


def test_disjoint_intervals_kept_separate():
    segs = merge_intervals([(_dt(0), _dt(2), 50), (_dt(4), _dt(6), 100)])
    assert [(s.start, s.end, s.lux) for s in segs] == [
        (_dt(0), _dt(2), 50),
        (_dt(4), _dt(6), 100),
    ]


def test_overlap_takes_max_lux_not_sum():
    # 两盏灯在 10:00-12:00 重叠：有效照度取 max(50, 200)=200
    segs = merge_intervals([(_dt(8), _dt(12), 50), (_dt(10), _dt(14), 200)])
    assert segs == [
        Segment(_dt(8), _dt(10), 50),
        Segment(_dt(10), _dt(14), 200),
    ]
    assert integrate(segs) == 50 * 2 + 200 * 4


def test_triple_overlap_and_coalescing():
    segs = merge_intervals(
        [(_dt(0), _dt(3), 50), (_dt(2), _dt(5), 80), (_dt(4), _dt(6), 80)]
    )
    # 0-2:50, 2-3:max(50,80)=80, 3-4:80, 4-5:max(80,80)=80, 5-6:80
    # 相邻同照度合并：2-6 连成一段 80
    assert segs == [Segment(_dt(0), _dt(2), 50), Segment(_dt(2), _dt(6), 80)]


def test_touching_intervals_are_not_overlap():
    # 首尾相接不算重叠，但同照度会被合并为一个片段
    segs = merge_intervals([(_dt(0), _dt(2), 50), (_dt(2), _dt(4), 50)])
    assert segs == [Segment(_dt(0), _dt(4), 50)]


def test_zero_effective_lux_dropped_when_brighter_lamp_active_then_ends():
    # 0 lux 区间与有灯区间重叠时不产生影响
    segs = merge_intervals([(_dt(0), _dt(4), 0), (_dt(1), _dt(3), 50)])
    assert segs == [Segment(_dt(1), _dt(3), 50)]


def test_clip_to_year_window():
    start, end = year_bounds(2026)
    segs = merge_intervals(
        [
            (datetime(2025, 12, 31, 20, tzinfo=timezone.utc),
             datetime(2026, 1, 1, 4, tzinfo=timezone.utc), 100)
        ]
    )
    clipped = clip_segments(segs, start, end)
    assert clipped == [
        Segment(start, datetime(2026, 1, 1, 4, tzinfo=timezone.utc), 100)
    ]
    assert integrate_within(segs, start, end) == 400


def test_first_exceed_time_interpolated():
    # 恒定 100 lux，上限 250 lux·h → 2.5 小时处超标
    segs = merge_intervals([(_dt(0), _dt(10), 100)])
    exceed = first_exceed_time(segs, 250.0, year_bounds(2026)[0])
    assert exceed == _dt(2.5)


def test_exactly_at_limit_is_not_exceeded():
    segs = merge_intervals([(_dt(0), _dt(5), 50)])  # 恰好 250
    assert first_exceed_time(segs, 250.0, year_bounds(2026)[0]) is None


def test_first_exceed_skips_low_exposure_segment():
    # 第一段 2h*50=100，第二段从 2h 起 200lux，还需 150 → 0.75h
    segs = merge_intervals([(_dt(0), _dt(2), 50), (_dt(2), _dt(6), 200)])
    exceed = first_exceed_time(segs, 250.0, year_bounds(2026)[0])
    assert exceed == _dt(2.75)


def test_coverage_gaps_merges_dark_and_dim_periods():
    # 08-12 @50 达标；12-14 无灯、14-16 @30 不达标、16-18 无灯 → 连成单一缺口
    segs = merge_intervals([(_dt(8), _dt(12), 50), (_dt(14), _dt(16), 30)])
    gaps, compliant = coverage_gaps(segs, _dt(8), _dt(18), 50)
    assert gaps == [(_dt(12), _dt(18))]
    assert compliant == 4.0


def test_coverage_gaps_exact_threshold_is_compliant():
    # 照度恰好等于门槛 → 达标，无缺口
    segs = merge_intervals([(_dt(8), _dt(18), 50)])
    gaps, compliant = coverage_gaps(segs, _dt(8), _dt(18), 50)
    assert gaps == []
    assert compliant == 10.0


def test_coverage_gaps_no_light_at_all():
    # 完全无灯：整个窗口是一个缺口
    gaps, compliant = coverage_gaps([], _dt(8), _dt(18), 50)
    assert gaps == [(_dt(8), _dt(18))]
    assert compliant == 0.0


def test_coverage_gaps_split_around_bright_period():
    # 10-12 @100 达标，前后各一段缺口
    segs = merge_intervals([(_dt(10), _dt(12), 100)])
    gaps, compliant = coverage_gaps(segs, _dt(8), _dt(18), 50)
    assert gaps == [(_dt(8), _dt(10)), (_dt(12), _dt(18))]
    assert compliant == 2.0


def test_coverage_gaps_clips_cross_day_segment_to_window():
    # 跨日片段 20:00 → 次日 10:00：当日窗口 08:00-12:00 无交集，次日窗口取 08:00-10:00
    segs = merge_intervals([(_dt(20), _dt(10, day=2), 80)])
    gaps, compliant = coverage_gaps(segs, _dt(8), _dt(12), 50)
    assert gaps == [(_dt(8), _dt(12))]
    assert compliant == 0.0
    gaps, compliant = coverage_gaps(segs, _dt(8, day=2), _dt(12, day=2), 50)
    assert gaps == [(_dt(10, day=2), _dt(12, day=2))]
    assert compliant == 2.0


# ------------------------------------------------------------- 曝光归因

def test_attribute_exposure_overlap_takes_max_and_shadows():
    # A: 08-18 @50；B: 10-12 @200 → 重叠段取 max，A 被遮蔽 2 小时
    total, attrs = attribute_exposure(
        [(1, _dt(8), _dt(18), 50), (2, _dt(10), _dt(12), 200)], _dt(8), _dt(18)
    )
    assert total == 50 * 2 + 200 * 2 + 50 * 6
    by_id = {a.interval_id: a for a in attrs}
    assert by_id[1].lux_hours == 50 * 2 + 50 * 6
    assert by_id[1].shadowed_hours == 2.0
    assert by_id[2].lux_hours == 200 * 2
    assert by_id[2].shadowed_hours == 0.0


def test_attribute_exposure_tied_brightest_split_evenly():
    # 两条 100 lux 并列最亮：该片的 200 lux·h 平分；50 lux 不获归因只记遮蔽
    total, attrs = attribute_exposure(
        [
            (1, _dt(10), _dt(12), 100),
            (2, _dt(10), _dt(12), 100),
            (3, _dt(10), _dt(12), 50),
        ],
        _dt(8),
        _dt(18),
    )
    assert total == 200.0
    by_id = {a.interval_id: a for a in attrs}
    assert by_id[1].lux_hours == 100.0
    assert by_id[2].lux_hours == 100.0
    assert by_id[3].lux_hours == 0.0
    assert by_id[3].shadowed_hours == 2.0


def test_attribute_exposure_clips_cross_boundary_intervals():
    # 跨界区间只计入窗口内部分；窗口外部分不产生归因与遮蔽
    total, attrs = attribute_exposure(
        [(1, _dt(20), _dt(10, day=2), 80), (2, _dt(21), _dt(23), 30)],
        _dt(22),
        _dt(23),
    )
    assert total == 80.0
    by_id = {a.interval_id: a for a in attrs}
    assert by_id[1].lux_hours == 80.0
    assert by_id[1].shadowed_hours == 0.0
    assert by_id[2].lux_hours == 0.0
    assert by_id[2].shadowed_hours == 1.0


def test_attribute_exposure_zero_lux_and_empty_window():
    # 0 lux 区间相交但不获归因也不被遮蔽；窗口内无区间时总量为 0、列表为空
    total, attrs = attribute_exposure([(1, _dt(8), _dt(10), 0)], _dt(8), _dt(18))
    assert total == 0.0
    assert [a.interval_id for a in attrs] == [1]
    assert attrs[0].lux_hours == 0.0
    assert attrs[0].shadowed_hours == 0.0

    total, attrs = attribute_exposure([], _dt(8), _dt(18))
    assert total == 0.0
    assert attrs == []


# ------------------------------------------------------------- 照明计划精简

def test_simplify_tied_equal_lux_inner_interval_removable():
    # A: 08-18 @50；B: 10-12 @50（等照度并列，被 A 完全覆盖）→ B 可撤、A 保留
    result = simplify_intervals(
        [(1, _dt(8), _dt(18), 50), (2, _dt(10), _dt(12), 50)]
    )
    assert result.removable_ids == (2,)
    assert result.retained_ids == (1,)
    # 原计划与精简计划的恒照度片段逐时刻一致（08-18 一段 50）
    assert result.original_segments == (Segment(_dt(8), _dt(18), 50),)
    assert result.simplified_segments == result.original_segments
    assert result.range_start == _dt(8)
    assert result.range_end == _dt(18)


def test_simplify_brighter_interval_must_be_retained():
    # A: 08-18 @50；B: 12-13 @300 → 两条都保留；撤任一条都会改变某时刻照度
    result = simplify_intervals(
        [
            (1, _dt(8), _dt(18), 50),
            (2, _dt(12), _dt(13), 300),
        ]
    )
    assert result.removable_ids == ()
    assert result.retained_ids == (1, 2)
    assert result.simplified_segments == result.original_segments


def test_simplify_dimmer_interval_dominated_by_brighter():
    # 两条范围相同：撤去较暗的 50 不影响逐时刻照度（取 max 恒为 80）
    result = simplify_intervals(
        [
            (1, _dt(8), _dt(10), 50),
            (2, _dt(8), _dt(10), 80),
        ]
    )
    assert result.removable_ids == (1,)
    assert result.retained_ids == (2,)
    assert result.simplified_segments == (Segment(_dt(8), _dt(10), 80),)


def test_simplify_id_descending_greedy_keeps_smallest_id_for_duplicates():
    # 完全相同的两条：ID 大者先试撤 → 2 可撤；再试 1 时集合里只剩它 → 保留
    result = simplify_intervals(
        [
            (1, _dt(8), _dt(10), 50),
            (2, _dt(8), _dt(10), 50),
        ]
    )
    assert result.removable_ids == (2,)
    assert result.retained_ids == (1,)


def test_simplify_chain_redundancy_decided_on_updated_set():
    # 三条等照度首尾相接（合并后 08-14 一段 50）：14 撤后 13 成为必需（盖住 12-14）。
    # ID 从大到小：撤 3 → 剩 1(08-12)+2(10-14) 仍为 08-14 @50 ✓；
    # 再撤 2 → 只剩 08-12，12-14 变暗 ✗ 保留 2；再撤 1 → 10-14 不含 08-10 ✗ 保留 1。
    result = simplify_intervals(
        [
            (1, _dt(8), _dt(12), 50),
            (2, _dt(10), _dt(14), 50),
            (3, _dt(12), _dt(14), 50),
        ]
    )
    assert result.removable_ids == (3,)
    assert result.retained_ids == (1, 2)


def test_simplify_zero_lux_intervals_always_removable():
    # 单独的 0 lux 区间：时间线恒为 0（无正照度片段），可撤；范围仍按其起止给出
    result = simplify_intervals([(1, _dt(8), _dt(10), 0)])
    assert result.removable_ids == (1,)
    assert result.retained_ids == ()
    assert result.original_segments == ()
    assert result.simplified_segments == ()
    assert result.range_start == _dt(8)
    assert result.range_end == _dt(10)

    # 与有灯区间重叠的 0 lux 同样可撤
    result = simplify_intervals(
        [
            (1, _dt(8), _dt(12), 50),
            (2, _dt(10), _dt(11), 0),
        ]
    )
    assert result.removable_ids == (2,)
    assert result.retained_ids == (1,)
    assert result.simplified_segments == (Segment(_dt(8), _dt(12), 50),)


def test_simplify_cross_day_and_cross_year_ranges():
    # 跨日：背景长区间覆盖短区间（等照度）→ 短区间可撤
    result = simplify_intervals(
        [
            (1, _dt(20), _dt(10, day=2), 80),
            (2, _dt(22), _dt(8, day=2), 80),
        ]
    )
    assert result.removable_ids == (2,)
    assert result.retained_ids == (1,)
    assert result.range_start == _dt(20)
    assert result.range_end == _dt(10, day=2)

    # 跨年：2026-12-31 → 2027-01-01 的两条等照度，短的可撤
    ny_eve = datetime(2026, 12, 31, 20, tzinfo=timezone.utc)
    ny_morn = datetime(2027, 1, 1, 4, tzinfo=timezone.utc)
    result = simplify_intervals(
        [
            (1, ny_eve, ny_morn, 100),
            (2, datetime(2026, 12, 31, 22, tzinfo=timezone.utc),
             datetime(2027, 1, 1, 2, tzinfo=timezone.utc), 100),
        ]
    )
    assert result.removable_ids == (2,)
    assert result.retained_ids == (1,)
    assert result.range_start == ny_eve
    assert result.range_end == ny_morn


def test_simplify_empty_intervals():
    result = simplify_intervals([])
    assert result.removable_ids == ()
    assert result.retained_ids == ()
    assert result.original_segments == ()
    assert result.simplified_segments == ()
    assert result.range_start is None
    assert result.range_end is None


def test_simplify_partial_overlap_requires_both():
    # A: 08-12 @50、B: 10-14 @50（部分重叠）：撤任一条都会让某段变暗
    result = simplify_intervals(
        [(1, _dt(8), _dt(12), 50), (2, _dt(10), _dt(14), 50)]
    )
    assert result.removable_ids == ()
    assert result.retained_ids == (1, 2)


def test_simplify_adjacent_different_lux_both_retained():
    # 首尾相接但照度不同：两条都必需
    result = simplify_intervals(
        [(1, _dt(8), _dt(10), 50), (2, _dt(10), _dt(12), 80)]
    )
    assert result.removable_ids == ()
    assert result.retained_ids == (1, 2)


def test_timelines_equal_is_pointwise_not_integral():
    # 两条时间线曝光积分相同（均 200 lux·h）但照度分布不同 → 不相等
    left = [Segment(_dt(8), _dt(12), 50)]
    right = [Segment(_dt(8), _dt(10), 100)]
    assert integrate(left) == integrate(right) == 200
    assert timelines_equal(left, right, _dt(8), _dt(12)) is False

    # 同一时间线逐片段一致 → 相等；范围外的差异不影响窗口内判定
    same = [Segment(_dt(8), _dt(12), 50)]
    assert timelines_equal(left, same, _dt(8), _dt(12)) is True
    assert timelines_equal(
        [Segment(_dt(8), _dt(10), 50)], same, _dt(8), _dt(10)
    ) is True


# ------------------------------------------------------------- 现场照度对账

def test_reconcile_basic_positive_and_negative_deviation():
    # 计划 08-12 @50；实测 08-10 @80（+30）、10-12 @30（-20）；允许偏差 10
    plan = merge_intervals([(_dt(8), _dt(12), 50)])
    readings = [(_dt(8), 80.0), (_dt(10), 30.0)]
    result = reconcile_illuminance(plan, readings, _dt(8), _dt(12), 10.0)
    assert result.positive_lux_hours == 60.0
    assert result.negative_lux_hours == 40.0
    assert result.over_tolerance_hours == 4.0
    assert result.anomalies == (
        DeviationAnomaly(_dt(8), _dt(10), DEVIATION_OVER, 2.0),
        DeviationAnomaly(_dt(10), _dt(12), DEVIATION_UNDER, 2.0),
    )


def test_reconcile_unscheduled_time_counts_as_zero_plan():
    # 窗口内完全无排灯：计划按 0 计算；实测 08-10 @20、10-12 @5，允许偏差 10
    readings = [(_dt(8), 20.0), (_dt(10), 5.0)]
    result = reconcile_illuminance([], readings, _dt(8), _dt(12), 10.0)
    assert result.positive_lux_hours == 50.0
    assert result.negative_lux_hours == 0.0
    assert result.over_tolerance_hours == 2.0  # 仅 08-10 超差
    assert result.anomalies == (DeviationAnomaly(_dt(8), _dt(10), DEVIATION_OVER, 2.0),)


def test_reconcile_exact_tolerance_is_not_anomaly():
    # 偏差恰等于阈值不算异常，但仍计入正负偏差积分
    plan = merge_intervals([(_dt(8), _dt(12), 50)])
    readings = [(_dt(8), 60.0), (_dt(10), 40.0)]
    result = reconcile_illuminance(plan, readings, _dt(8), _dt(12), 10.0)
    assert result.positive_lux_hours == 20.0
    assert result.negative_lux_hours == 20.0
    assert result.over_tolerance_hours == 0.0
    assert result.anomalies == ()


def test_reconcile_merges_adjacent_same_direction_anomalies():
    # 计划 08-14 @50；实测 08-10 @70、10-12 @90、12-14 @45（允许偏差 10）
    # 前两段同为正偏差且相邻 → 合并为 08-12；12-14 实测 45（-5）合规但计入负偏差
    plan = merge_intervals([(_dt(8), _dt(14), 50)])
    readings = [(_dt(8), 70.0), (_dt(10), 90.0), (_dt(12), 45.0)]
    result = reconcile_illuminance(plan, readings, _dt(8), _dt(14), 10.0)
    assert result.positive_lux_hours == 20 * 2 + 40 * 2
    assert result.negative_lux_hours == 5 * 2
    assert result.over_tolerance_hours == 4.0
    assert result.anomalies == (DeviationAnomaly(_dt(8), _dt(12), DEVIATION_OVER, 4.0),)


def test_reconcile_plan_change_point_splits_reading_step():
    # 计划 08-10 @50、10-12 @200；实测全程 @100（允许偏差 10）
    # 08-10: +50 超差；10-12: -100 超差 → 两个方向各一段
    plan = merge_intervals([(_dt(8), _dt(10), 50), (_dt(10), _dt(12), 200)])
    readings = [(_dt(8), 100.0)]
    result = reconcile_illuminance(plan, readings, _dt(8), _dt(12), 10.0)
    assert result.positive_lux_hours == 100.0
    assert result.negative_lux_hours == 200.0
    assert result.over_tolerance_hours == 4.0
    assert result.anomalies == (
        DeviationAnomaly(_dt(8), _dt(10), DEVIATION_OVER, 2.0),
        DeviationAnomaly(_dt(10), _dt(12), DEVIATION_UNDER, 2.0),
    )


def test_reconcile_clips_plan_segments_to_window():
    # 跨界计划区间 06-09 @50 裁剪到窗口 08-12；实测 08-12 @40（允许偏差 10）
    plan = merge_intervals([(_dt(6), _dt(9), 50)])
    readings = [(_dt(8), 40.0)]
    result = reconcile_illuminance(plan, readings, _dt(8), _dt(12), 10.0)
    # 08-09: -10 恰好到阈值不记异常；09-12: 未排灯计划按 0，实测 +40 超差
    assert result.positive_lux_hours == 120.0
    assert result.negative_lux_hours == 10.0
    assert result.over_tolerance_hours == 3.0
    assert result.anomalies == (DeviationAnomaly(_dt(9), _dt(12), DEVIATION_OVER, 3.0),)
