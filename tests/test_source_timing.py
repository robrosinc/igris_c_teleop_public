"""Source timing: which polled records are new, and when the sender clock starts over."""

from __future__ import annotations

from igris_c_teleop.uplink.source_timing import MAX_SOURCE_TIMESTAMP_NS, SourceTimeline


def test_each_record_is_fresh_once():
    # A 100 Hz loop polling a ~52 Hz sender sees most records twice.
    timeline = SourceTimeline()
    stamps = (1_000, 1_000, 20_300, 20_300, 20_300, 34_100)
    ticks = [timeline.update(stamp) for stamp in stamps]
    assert [tick.fresh for tick in ticks] == [True, False, True, False, False, True]
    assert [tick.source_ns for tick in ticks] == list(stamps)
    assert {tick.segment for tick in ticks} == {1}
    assert timeline.repeats == 3 and timeline.backward_steps == 0


def test_reconnection_starts_a_new_segment():
    timeline = SourceTimeline()
    assert timeline.update(500).segment == 1
    timeline.reset()  # skeleton lost
    # The same stamp after a reconnection is a new record of a new segment, not a repeat.
    tick = timeline.update(500)
    assert tick.fresh and tick.segment == 2 and timeline.repeats == 0


def test_backward_sender_clock_starts_a_new_segment():
    timeline = SourceTimeline()
    timeline.update(9_000_000_000)
    tick = timeline.update(1_000_000)  # headset rebooted inside the stale timeout
    assert tick.fresh and tick.segment == 2 and tick.source_ns == 1_000_000
    assert timeline.backward_steps == 1
    assert timeline.update(2_000_000).segment == 2


def test_unstamped_records_run_on_the_host_clock():
    timeline = SourceTimeline()
    for stamp in (0, 0, -5, MAX_SOURCE_TIMESTAMP_NS + 1):
        tick = timeline.update(stamp)
        assert tick.fresh and tick.source_ns is None and tick.segment == 0
    assert timeline.repeats == 0
    # Stamps appearing later open the first segment.
    assert timeline.update(MAX_SOURCE_TIMESTAMP_NS) == type(tick)(True, MAX_SOURCE_TIMESTAMP_NS, 1)


def test_disabled_timeline_never_skips_a_tick():
    timeline = SourceTimeline(enabled=False)
    ticks = [timeline.update(stamp) for stamp in (1_000, 1_000, 2_000)]
    assert all(tick.fresh and tick.source_ns is None for tick in ticks)
    assert timeline.segment == 0 and timeline.repeats == 0
