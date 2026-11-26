"""Source timing: one reference frame per headset record, timed on the headset clock.

The headset sends VRS1 records at its own frame rate (about 52 Hz with 13 to
28 ms gaps in a 2026-11 capture) while the producer polls the newest record at
``--target_fps`` (100 Hz). Retargeting on every poll publishes most poses
twice and divides each motion step by the poll period: the published
velocities alternate between zero and about twice the true value, and the
consumer's history shows a staircase. ``SourceTimeline`` classifies each polled
record instead:

* ``fresh``: the sender timestamp differs from the last record processed. Only
  fresh records are retargeted and published as a reference frame.
* ``source_ns``: that timestamp, the time base of the finite-difference
  velocities and of ``record_meta.source_timestamp_ns``.
* ``segment``: one stretch of a continuous sender clock. It advances on every
  (re)connection and when the timestamp steps backwards (headset reboot); the
  consumer starts a new reference history on a new ``source_session_id``.

A record without a usable timestamp (0 from a sender older than the field) or a
disabled timeline (``--no_source_timing``) is always fresh with ``source_ns``
None: every poll is processed on the host clock, as before 2026-11-26.
"""

from __future__ import annotations

from dataclasses import dataclass

# The consumer parses the stamp as a signed 64-bit integer.
MAX_SOURCE_TIMESTAMP_NS = 2**63 - 1


@dataclass(frozen=True)
class SourceTick:
    fresh: bool  # a record not processed before: retarget and publish it
    source_ns: int | None  # sender timestamp of the record, None on the host-clock path
    segment: int  # index of the current continuous sender clock (0 before the first stamped record)


class SourceTimeline:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.segment = 0
        self._last_ns: int | None = None
        self.repeats = 0  # polls that found no new record
        self.backward_steps = 0  # sender clock went backwards on a live connection

    def reset(self) -> None:
        """Skeleton lost: the next stamped record starts a new segment."""

        self._last_ns = None

    def update(self, source_timestamp_ns: int) -> SourceTick:
        """Classify the record polled this tick; ``source_timestamp_ns`` is ``VRInputState.source_timestamp_ns``."""

        stamp = int(source_timestamp_ns)
        if not self.enabled or not 0 < stamp <= MAX_SOURCE_TIMESTAMP_NS:
            self._last_ns = None
            return SourceTick(True, None, self.segment)
        if self._last_ns is None:
            self.segment += 1
        elif stamp == self._last_ns:
            self.repeats += 1
            return SourceTick(False, stamp, self.segment)
        elif stamp < self._last_ns:
            self.segment += 1
            self.backward_steps += 1
        self._last_ns = stamp
        return SourceTick(True, stamp, self.segment)


__all__ = ["MAX_SOURCE_TIMESTAMP_NS", "SourceTick", "SourceTimeline"]
