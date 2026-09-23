"""igris_c_sdk (CycloneDDS) outputs: ``HandCmd`` publisher, control-mode service requests,
and the opt-in ``RefState`` publisher (retargeted joints for the headset overlay).

HandCmd and the control-mode requests drive the real robot. ROBROS hardware-change approval
and validation are required before running against it. The RefState stream is consumed
only by the VR headset (display). The SDK binding is a Python 3.12 wheel
(``task wheel`` in ``thirdparty/igris_c_sdk``); it is imported lazily so the
producer can run ROS-only with ``--no_dds`` in an environment without it.

Ordering constraints (from the binding):
  * ``ChannelFactory.Instance().Init`` once per process, before any pub/sub.
  * ``IgrisC_Client.Init`` holds the GIL while creating readers; call it right
    after the channel init, before other subscribers exist, and never again.
  * ``SendControlModeCommand`` blocks up to ``timeout_ms`` but releases the
    GIL, so it runs on a worker thread and never stalls the 100 Hz loop.
  * Release the channel only after the worker has finished.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any

import numpy as np

from .controller import EVENT_TO_CONTROL_MODE
from .hands import HAND_COMMAND_DIM, MOTOR_IDS

logger = logging.getLogger("igris_c_gmr.dds")

DEFAULT_HAND_TOPIC = "handcmd"
# Bare topic of the retargeted-joint reference stream read by the vr_system headset
# (ARobotTeleopPawn.ReferenceTopic). Deliberately not "masterarmstate": that name belongs to
# the master-arm device, and the robot bridge must not mistake this display stream for one.
DEFAULT_REF_TOPIC = "gmr/refstate"
REF_JOINT_COUNT = 31  # MJCF hinge order == igris_c_sdk 31-motor order (robros_igris_c_v2)
# Bare service topic the SDK's IgrisC_Client publishes on (thirdparty/igris_c_sdk src/igris_c_client.cpp),
# matching igris_c_bridge dds_manager.cpp. Only informational here: the client owns the real name, and a
# wheel built from an older SDK (e.g. the 2026-09-15 build) still uses "rt/service/..." and will not reach
# the current bridge. Rebuild the wheel from the submodule pin.
CONTROL_MODE_REQUEST_TOPIC = "service/control_mode/request"
TOPIC_NAMINGS = ("native", "ros_compatible")
NO_DDS_HINT = "Run with --no_dds for a ROS-only session, or install the igris_c_sdk Python 3.12 wheel (task wheel in thirdparty/igris_c_sdk, then pixi run install-sdk)."


def import_sdk():
    try:
        import igris_c_sdk as sdk
    except ImportError as exc:
        raise SystemExit(f"igris_c_sdk Python binding not importable ({exc}). {NO_DDS_HINT}") from exc
    if not hasattr(sdk, "ChannelFactory"):
        raise SystemExit(
            f"'igris_c_sdk' resolved to {getattr(sdk, '__file__', '?')} which has no ChannelFactory: this is the rosidl "
            "message package from the igris_c ROS overlay shadowing the pybind wheel. Remove the igris_c install "
            "overlay from PYTHONPATH (run through `pixi run`; scripts/activate_ros.sh rewrites it) or run with --no_dds."
        )
    return sdk


def sdk_type(sdk, name: str):
    """Struct lookup accepting both the current (``HandCmd``) and older (``HandCmd_``) spellings."""

    for candidate in (name, f"{name}_"):
        found = getattr(sdk, candidate, None)
        if found is not None:
            return found
    raise SystemExit(f"igris_c_sdk exposes neither {name!r} nor {name + '_'!r}; binding version mismatch.")


class DdsChannel:
    """Process-wide ``ChannelFactory`` singleton wrapper (init once, release last)."""

    def __init__(self, sdk, channel, *, domain_id: int, namespace: str, topic_naming: str) -> None:
        self.sdk = sdk
        self._channel = channel
        self.domain_id = int(domain_id)
        self.namespace = str(namespace)
        self.topic_naming = topic_naming
        self._released = False

    @classmethod
    def create(cls, *, domain_id: int, namespace: str, topic_naming: str = "native", sdk=None) -> "DdsChannel":
        if topic_naming not in TOPIC_NAMINGS:
            raise ValueError(f"topic_naming must be one of {TOPIC_NAMINGS}, got {topic_naming!r}.")
        if not namespace:
            raise ValueError("DDS namespace must not be empty: a wrong namespace fails silently on the wire.")
        if sdk is None:
            sdk = import_sdk()
        channel = sdk.ChannelFactory.Instance()
        parse_naming = getattr(sdk, "parse_topic_naming", None)
        if parse_naming is not None:
            channel.Init(int(domain_id), str(namespace), "", parse_naming(topic_naming))
        else:
            # Older binding: Init(domain_id, ns[, config]); only native naming exists there.
            if topic_naming != "native":
                raise SystemExit("This igris_c_sdk binding predates topic naming; only 'native' is available.")
            channel.Init(int(domain_id), str(namespace))
        if hasattr(channel, "IsInitialized") and not channel.IsInitialized():
            raise SystemExit(f"igris_c_sdk ChannelFactory failed to initialize (domain {domain_id}, namespace {namespace!r}).")
        # The wire contract (type names, service topic prefix) is whatever the installed wheel was
        # built from, so it must match the robot bridge; log what we actually loaded.
        logger.info(
            "igris_c_sdk %s from %s (domain %d, namespace %r, naming %s)",
            getattr(sdk, "__version__", "?"),
            getattr(sdk, "__file__", "?"),
            int(domain_id),
            namespace,
            topic_naming,
        )
        return cls(sdk, channel, domain_id=domain_id, namespace=namespace, topic_naming=topic_naming)

    def resolve(self, topic: str) -> str:
        """Wire name of a bare topic under the configured namespace/naming."""

        resolver = getattr(self._channel, "resolve", None)
        if resolver is not None:
            return str(resolver(topic))
        bare = topic[1:] if topic.startswith("/") else topic
        if self.topic_naming == "ros_compatible":
            return f"rt/{self.namespace}/{bare}"
        return f"{self.namespace}/{bare}"

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            self._channel.Release()
        except Exception as exc:  # teardown must not mask the run's outcome
            logger.warning("igris_c_sdk ChannelFactory release failed: %s", exc)


class HandCmdDdsPublisher:
    """``handcmd`` publisher: 12 normalized motor targets in ``MOTOR_IDS`` order."""

    def __init__(self, channel: DdsChannel, topic: str = DEFAULT_HAND_TOPIC) -> None:
        sdk = channel.sdk
        self._motor_cmd_type = sdk_type(sdk, "MotorCmd")
        self._hand_cmd_type = sdk_type(sdk, "HandCmd")
        self.topic = str(topic)
        self.resolved_topic = channel.resolve(self.topic)
        self._publisher = sdk.HandCmdPublisher(self.topic, sdk.QosProfile.SensorData())
        if not self._publisher.init():
            raise SystemExit(f"igris_c_sdk HandCmdPublisher failed to initialize on {self.topic!r}.")
        self.published = 0
        self.rejected = 0

    def publish(self, command) -> None:
        values = np.asarray(command, dtype=np.float64).reshape(-1)
        if values.shape != (HAND_COMMAND_DIM,):
            raise ValueError(f"Hand command must have {HAND_COMMAND_DIM} values, got shape {values.shape}.")
        motors = []
        for motor_id, value in zip(MOTOR_IDS, values):
            motor = self._motor_cmd_type()
            motor.id(int(motor_id))
            motor.q(float(min(1.0, max(0.0, value))))
            motors.append(motor)
        message = self._hand_cmd_type()
        message.motor_cmd(motors)
        if self._publisher.write(message) is False:
            self.rejected += 1
            return
        self.published += 1
        if self.published == 1:
            logger.info("First HandCmd written on %s: ids=%s q=%s", self.resolved_topic, list(MOTOR_IDS), np.round(values, 3).tolist())

    def close(self) -> None:
        try:
            stop = getattr(self._publisher, "stop", None)
            if stop is not None:
                stop()
        except Exception as exc:
            logger.warning("igris_c_sdk HandCmdPublisher stop failed: %s", exc)


class RefStateDdsPublisher:
    """``gmr/refstate`` publisher: the 31 retargeted hinge joints as a ``MasterArmState`` message.

    Display-only side channel for the vr_system headset (hand-target overlay,
    vr_system/docs/Arm_Impedance_Overlay_Plan.md): the headset runs the arm FK on
    ``joint_state[i].q`` to draw where the operator's hand is on the robot and compares it
    with LowState. The type is reused so no IDL/SDK change is needed; only ``header.stamp``
    (producer wall clock, the same clock the video frames are stamped with) and
    ``joint_state[].q`` are filled, every other field stays at its default. The robot bridge
    does not subscribe to this topic.
    """

    def __init__(self, channel: DdsChannel, topic: str = DEFAULT_REF_TOPIC) -> None:
        sdk = channel.sdk
        self._state_type = sdk_type(sdk, "MasterArmState")
        self._joint_type = sdk_type(sdk, "JointState")
        self._header_type = sdk_type(sdk, "Header")
        self._time_type = sdk_type(sdk, "Time")
        self.topic = str(topic)
        self.resolved_topic = channel.resolve(self.topic)
        self._publisher = sdk.MasterArmStatePublisher(self.topic, sdk.QosProfile.SensorData())
        if not self._publisher.init():
            raise SystemExit(f"igris_c_sdk MasterArmStatePublisher failed to initialize on {self.topic!r}.")
        self.published = 0
        self.rejected = 0

    def publish(self, joint_positions, stamp: float) -> None:
        """Write one reference sample; ``stamp`` is ``time.time()`` seconds of the solve."""

        values = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
        if values.shape != (REF_JOINT_COUNT,):
            raise ValueError(f"Reference joints must have {REF_JOINT_COUNT} values, got shape {values.shape}.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Reference joints contain NaN/inf.")
        joints = []
        for value in values:
            joint = self._joint_type()
            joint.q(float(value))
            joints.append(joint)
        sec = int(stamp)
        nanosec = int(round((stamp - sec) * 1e9))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        when = self._time_type()
        when.sec(sec)
        when.nanosec(nanosec)
        header = self._header_type()
        header.stamp(when)
        message = self._state_type()
        message.header(header)
        message.joint_state(joints)
        if self._publisher.write(message) is False:
            self.rejected += 1
            return
        self.published += 1
        if self.published == 1:
            logger.info("First RefState written on %s (%d joints, stamp %d.%09d)", self.resolved_topic, REF_JOINT_COUNT, sec, nanosec)

    def close(self) -> None:
        try:
            stop = getattr(self._publisher, "stop", None)
            if stop is not None:
                stop()
        except Exception as exc:
            logger.warning("igris_c_sdk MasterArmStatePublisher stop failed: %s", exc)


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
        self._worker = threading.Thread(target=self._run, name="igris_c_gmr-control-mode", daemon=True)
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


__all__ = [
    "CONTROL_MODE_REQUEST_TOPIC",
    "DEFAULT_HAND_TOPIC",
    "DEFAULT_REF_TOPIC",
    "REF_JOINT_COUNT",
    "TOPIC_NAMINGS",
    "ControlModeClient",
    "DdsChannel",
    "HandCmdDdsPublisher",
    "RefStateDdsPublisher",
    "import_sdk",
    "sdk_type",
]
