"""Controller events -> control-mode service requests (``IgrisC_Client``) on a worker thread.

Switches the real robot's control mode. ROBROS hardware-change approval and validation are
required before running against it.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any

from ..uplink.controller import EVENT_TO_CONTROL_MODE
from .channel import DdsChannel

logger = logging.getLogger("igris_c_teleop.dds")

# Bare service topic the SDK's IgrisC_Client publishes on (igris_c_sdk src/igris_c_client.cpp), matching
# igris_c_bridge dds_manager.cpp. Only informational here: the client owns the real name, and a wheel built
# from an older SDK (e.g. the 2026-09-15 build) still uses "rt/service/..." and will not reach the current
# bridge. Install the tag-pinned release wheel from pyproject.toml (`pixi install`).
CONTROL_MODE_REQUEST_TOPIC = "service/control_mode/request"


class ControlModeClient:
    """Controller events -> ``SendControlModeCommand`` on a worker thread.

    ``request`` never blocks: events queue up (bounded, newest dropped with a
    warning when full) and the worker performs the blocking service call and
    logs the response. Counters expose what happened for the periodic status
    line and for tests.
    """

    def __init__(
        self,
        channel: DdsChannel,
        *,
        timeout_ms: int = 3000,
        queue_size: int = 4,
        client=None,
        mapping: dict[str, tuple[str, str]] = EVENT_TO_CONTROL_MODE,
    ) -> None:
        sdk = channel.sdk
        self.timeout_ms = int(timeout_ms)
        self._commands: dict[str, tuple[Any, str]] = {}
        for event, (constant_name, preset_id) in mapping.items():
            value = getattr(sdk, constant_name, None)
            if value is None:
                raise SystemExit(f"igris_c_sdk has no {constant_name!r} (needed for event {event!r}); binding version mismatch.")
            self._commands[event] = (value, preset_id)
        self.request_topic = channel.resolve(CONTROL_MODE_REQUEST_TOPIC)
        if client is None:
            client = sdk.IgrisC_Client()
            client.Init()
        self._client = client
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._closed = False
        self.requested = 0
        self.dropped = 0
        self.succeeded = 0
        self.failed = 0
        self._worker = threading.Thread(target=self._run, name="igris_c_teleop-control-mode", daemon=True)
        self._worker.start()

    @property
    def events(self) -> tuple[str, ...]:
        return tuple(self._commands)

    def request(self, event: str) -> bool:
        """Queue one control-mode request; False when unknown, closed or the queue is full."""

        if event not in self._commands:
            logger.warning("Unknown control-mode event %r ignored.", event)
            return False
        if self._closed:
            return False
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self.dropped += 1
            logger.warning("Control-mode request %r dropped: %d requests still pending.", event, self._queue.qsize())
            return False
        self.requested += 1
        return True

    def _run(self) -> None:
        while True:
            event = self._queue.get()
            if event is None:
                return
            command_type, preset_id = self._commands[event]
            try:
                response = self._client.SendControlModeCommand(command_type, preset_id, False, self.timeout_ms)
            except Exception as exc:
                self.failed += 1
                logger.error("Control-mode request %r raised: %s", event, exc)
                continue
            if response.success():
                self.succeeded += 1
                logger.warning(
                    "Control-mode request %r accepted (%s%s): request_id=%s message=%r",
                    event,
                    command_type,
                    f" preset={preset_id}" if preset_id else "",
                    response.request_id(),
                    response.message(),
                )
            else:
                self.failed += 1
                logger.error(
                    "Control-mode request %r FAILED: error_code=%s message=%r request_id=%s",
                    event,
                    response.error_code(),
                    response.message(),
                    response.request_id(),
                )

    def pending(self) -> int:
        return self._queue.qsize()

    def close(self, timeout: float | None = None) -> None:
        """Stop accepting requests and wait for in-flight service calls."""

        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._worker.join(timeout if timeout is not None else self.timeout_ms / 1000.0 + 1.0)
        if self._worker.is_alive():
            logger.warning("Control-mode worker still running after close; a service call is hanging.")


__all__ = ["CONTROL_MODE_REQUEST_TOPIC", "ControlModeClient"]
