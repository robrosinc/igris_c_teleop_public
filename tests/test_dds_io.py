"""igris_c_sdk outputs against a fake binding: channel init, HandCmd order, control-mode worker."""

from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from general_motion_retargeting.hand_retarget import igris_c_hand_command as hand_cmd

from igris_c_gmr import dds_io
from igris_c_gmr.controller import EVENT_TO_CONTROL_MODE


class _FakeMotorCmd:
    def __init__(self):
        self.motor_id = None
        self.value = None

    def id(self, value):
        self.motor_id = value

    def q(self, value):
        self.value = value


class _FakeHandCmd:
    def __init__(self):
        self.motors = None

    def motor_cmd(self, motors):
        self.motors = list(motors)


class _FakeChannel:
    def __init__(self):
        self.init_args = None
        self.releases = 0

    def Init(self, domain_id, ns="", config="", naming="native"):
        self.init_args = (domain_id, ns, config, naming)

    def IsInitialized(self):
        return self.init_args is not None

    def resolve(self, topic):
        _domain, ns, _config, naming = self.init_args
        bare = topic.lstrip("/")
        return f"rt/{ns}/{bare}" if naming == "ros_compatible" else f"{ns}/{bare}"

    def Release(self):
        self.releases += 1


class _FakePublisher:
    def __init__(self, topic, qos):
        self.topic = topic
        self.qos = qos
        self.written = []
        self.stops = 0

    def init(self):
        return True

    def write(self, message):
        self.written.append(message)
        return True

    def stop(self):
        self.stops += 1


class _FakeResponse:
    def __init__(self, ok: bool, code: int = 0, message: str = "", request_id: str = "req-1"):
        self._ok, self._code, self._message, self._id = ok, code, message, request_id

    def success(self):
        return self._ok

    def error_code(self):
        return self._code

    def message(self):
        return self._message

    def request_id(self):
        return self._id


class _FakeClient:
    """Records SendControlModeCommand calls; optionally blocks until released."""

    def __init__(self, ok: bool = True, block: threading.Event | None = None):
        self.calls = []
        self.ok = ok
        self.block = block
        self.inits = 0
        self.called = threading.Event()

    def Init(self):
        self.inits += 1

    def SendControlModeCommand(self, command_type, preset_id, is_cyclic, timeout_ms):
        self.calls.append((command_type, preset_id, is_cyclic, timeout_ms))
        self.called.set()
        if self.block is not None:
            self.block.wait(5.0)
        return _FakeResponse(self.ok, 0 if self.ok else 7, "" if self.ok else "rejected", f"req-{len(self.calls)}")


def _fake_sdk(legacy_names: bool = False):
    channel = _FakeChannel()
    publishers = []

    class _ChannelFactory:
        @staticmethod
        def Instance():
            return channel

    def _publisher(topic, qos):
        publisher = _FakePublisher(topic, qos)
        publishers.append(publisher)
        return publisher

    class _Qos:
        @staticmethod
        def SensorData():
            return "sensor-data"

    fields = dict(
        ChannelFactory=_ChannelFactory,
        HandCmdPublisher=_publisher,
        QosProfile=_Qos,
        parse_topic_naming=lambda value: value,
        IgrisC_Client=_FakeClient,
        CONTROL_MODE_CMD_LOW_LEVEL_JOINT_CONTROL=5,
        CONTROL_MODE_CMD_JOINT_POSITION_HOLD=2,
        CONTROL_MODE_CMD_MOTION_PRESET=0,
    )
    if legacy_names:
        fields.update(MotorCmd_=_FakeMotorCmd, HandCmd_=_FakeHandCmd)
    else:
        fields.update(MotorCmd=_FakeMotorCmd, HandCmd=_FakeHandCmd)
    return SimpleNamespace(**fields), channel, publishers


def _wait(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting for the control-mode worker"
        time.sleep(0.005)


# -- channel ---------------------------------------------------------------------------


def test_channel_init_resolve_release():
    sdk, channel, _ = _fake_sdk()
    dds = dds_io.DdsChannel.create(domain_id=3, namespace="igris_c_IG07", sdk=sdk)
    assert channel.init_args == (3, "igris_c_IG07", "", "native")
    assert dds.resolve("handcmd") == "igris_c_IG07/handcmd"
    assert dds.resolve(dds_io.CONTROL_MODE_REQUEST_TOPIC) == "igris_c_IG07/service/control_mode/request"
    dds.release()
    dds.release()
    assert channel.releases == 1

    ros_compat = dds_io.DdsChannel.create(domain_id=0, namespace="igris_c_IG07", topic_naming="ros_compatible", sdk=sdk)
    assert ros_compat.resolve("handcmd") == "rt/igris_c_IG07/handcmd"
    with pytest.raises(ValueError):
        dds_io.DdsChannel.create(domain_id=0, namespace="", sdk=sdk)
    with pytest.raises(ValueError):
        dds_io.DdsChannel.create(domain_id=0, namespace="x", topic_naming="ros", sdk=sdk)


def test_channel_requires_the_pybind_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "igris_c_sdk", None)  # import raises ImportError
    with pytest.raises(SystemExit, match="no_dds"):
        dds_io.DdsChannel.create(domain_id=0, namespace="igris_c_IG01")
    # The rosidl message package of the same name (ROS overlay) shadows the binding.
    monkeypatch.setitem(sys.modules, "igris_c_sdk", SimpleNamespace(__file__="/igris_c/install/igris_c_sdk/__init__.py"))
    with pytest.raises(SystemExit, match="rosidl"):
        dds_io.DdsChannel.create(domain_id=0, namespace="igris_c_IG01")


# -- hand publisher -----------------------------------------------------------------------


def test_hand_publisher_writes_motor_order():
    sdk, _channel, publishers = _fake_sdk()
    dds = dds_io.DdsChannel.create(domain_id=0, namespace="igris_c_IG07", sdk=sdk)
    hand = dds_io.HandCmdDdsPublisher(dds)
    assert hand.topic == "handcmd" and hand.resolved_topic == "igris_c_IG07/handcmd"
    assert len(publishers) == 1 and publishers[0].topic == "handcmd" and publishers[0].qos == "sensor-data"

    command = np.linspace(0.0, 1.1, 12)
    hand.publish(command)
    (message,) = publishers[0].written
    assert [motor.motor_id for motor in message.motors] == list(hand_cmd.MOTOR_IDS)
    assert [motor.value for motor in message.motors] == pytest.approx(np.clip(command, 0.0, 1.0).tolist())
    assert hand.published == 1 and hand.rejected == 0
    with pytest.raises(ValueError):
        hand.publish(np.zeros(6))
    hand.close()
    assert publishers[0].stops == 1


def test_hand_publisher_accepts_legacy_struct_names():
    sdk, _channel, publishers = _fake_sdk(legacy_names=True)
    dds = dds_io.DdsChannel.create(domain_id=0, namespace="igris_c_IG01", sdk=sdk)
    dds_io.HandCmdDdsPublisher(dds, "handcmd").publish(np.zeros(12))
    assert len(publishers[0].written) == 1
    broken = SimpleNamespace(ChannelFactory=sdk.ChannelFactory, HandCmdPublisher=sdk.HandCmdPublisher, QosProfile=sdk.QosProfile)
    with pytest.raises(SystemExit, match="MotorCmd"):
        dds_io.HandCmdDdsPublisher(dds_io.DdsChannel.create(domain_id=0, namespace="igris_c_IG01", sdk=broken))


# -- control mode -------------------------------------------------------------------------


def test_control_mode_client_maps_events_and_logs_results():
    sdk, _channel, _ = _fake_sdk()
    dds = dds_io.DdsChannel.create(domain_id=0, namespace="igris_c_IG07", sdk=sdk)
    client = _FakeClient(ok=True)
    control = dds_io.ControlModeClient(dds, timeout_ms=1234, client=client)
    assert control.events == tuple(EVENT_TO_CONTROL_MODE)
    assert control.request_topic == "igris_c_IG07/service/control_mode/request"

    assert control.request("lowcmd_start") and control.request("stop") and control.request("go_home")
    assert not control.request("jump")
    _wait(lambda: len(client.calls) == 3)
    assert client.calls == [(5, "", False, 1234), (2, "", False, 1234), (0, "HOME", False, 1234)]
    control.close()
    assert control.requested == 3 and control.succeeded == 3 and control.failed == 0 and control.dropped == 0
    assert not control.request("stop")  # closed


def test_control_mode_client_counts_failures():
    sdk, _channel, _ = _fake_sdk()
    dds = dds_io.DdsChannel.create(domain_id=0, namespace="igris_c_IG07", sdk=sdk)
    control = dds_io.ControlModeClient(dds, client=_FakeClient(ok=False))
    control.request("stop")
    _wait(lambda: control.failed == 1)
    control.close()
    assert control.succeeded == 0


def test_control_mode_client_bounded_queue_drops_when_service_hangs():
    sdk, _channel, _ = _fake_sdk()
    dds = dds_io.DdsChannel.create(domain_id=0, namespace="igris_c_IG07", sdk=sdk)
    release = threading.Event()
    client = _FakeClient(ok=True, block=release)
    control = dds_io.ControlModeClient(dds, timeout_ms=100, queue_size=2, client=client)
    assert control.request("stop")  # taken by the worker, now blocked in the service call
    client.called.wait(2.0)
    assert control.request("go_home") and control.request("go_home")  # fills the queue of 2
    assert not control.request("go_home")  # dropped
    assert control.dropped == 1 and control.pending() == 2
    release.set()
    _wait(lambda: len(client.calls) == 3)
    control.close()
    assert control.succeeded == 3


def test_control_mode_client_inits_sdk_client_once():
    sdk, _channel, _ = _fake_sdk()
    dds = dds_io.DdsChannel.create(domain_id=0, namespace="igris_c_IG07", sdk=sdk)
    control = dds_io.ControlModeClient(dds)
    assert control._client.inits == 1
    control.close()


def test_control_mode_client_rejects_binding_without_constants():
    sdk, _channel, _ = _fake_sdk()
    del sdk.CONTROL_MODE_CMD_MOTION_PRESET
    dds = dds_io.DdsChannel.create(domain_id=0, namespace="igris_c_IG07", sdk=sdk)
    with pytest.raises(SystemExit, match="CONTROL_MODE_CMD_MOTION_PRESET"):
        dds_io.ControlModeClient(dds, client=_FakeClient())
