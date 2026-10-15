"""igris_c_teleop.dds against a fake binding: channel init, HandCmd order, control-mode worker."""

from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from general_motion_retargeting.hand_retarget import igris_c_hand_command as hand_cmd

from igris_c_teleop import dds as dds_io
from igris_c_teleop.uplink.controller import EVENT_TO_CONTROL_MODE


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


class _FakeJointState:
    def __init__(self):
        self.value = None

    def q(self, value):
        self.value = value


class _FakeTime:
    def __init__(self):
        self.sec_value = None
        self.nanosec_value = None

    def sec(self, value):
        self.sec_value = value

    def nanosec(self, value):
        self.nanosec_value = value


class _FakeHeader:
    def __init__(self):
        self.stamp_value = None

    def stamp(self, value):
        self.stamp_value = value


class _FakeMasterArmState:
    def __init__(self):
        self.header_value = None
        self.joints = None

    def header(self, value):
        self.header_value = value

    def joint_state(self, joints):
        self.joints = list(joints)


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


class _FakeSubscriber:
    def __init__(self, topic, qos):
        self.topic = topic
        self.qos = qos
        self.callback = None
        self.stops = 0

    def init(self, callback):
        self.callback = callback
        return True

    def stop(self):
        self.stops += 1


class _FakeLowState:
    def __init__(self, q):
        self._joints = [SimpleNamespace(q=lambda v=v: v) for v in q]

    def joint_state(self):
        return self._joints


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

        @staticmethod
        def Default():
            return "default"

    subscribers = []

    def _subscriber(topic, qos):
        subscriber = _FakeSubscriber(topic, qos)
        subscribers.append(subscriber)
        return subscriber

    fields = dict(
        ChannelFactory=_ChannelFactory,
        HandCmdPublisher=_publisher,
        MasterArmStatePublisher=_publisher,
        LowStateSubscriber=_subscriber,
        RobotStateSubscriber=_subscriber,
        subscribers=subscribers,
        MasterArmState=_FakeMasterArmState,
        JointState=_FakeJointState,
        Header=_FakeHeader,
        Time=_FakeTime,
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


# -- reference publisher --------------------------------------------------------------------


def test_ref_state_publisher_writes_joints_and_stamp():
    sdk, _channel, publishers = _fake_sdk()
    dds = dds_io.DdsChannel.create(domain_id=0, namespace="igris_c_IG07", sdk=sdk)
    ref = dds_io.RefStateDdsPublisher(dds)
    assert ref.topic == dds_io.DEFAULT_REF_TOPIC == "gmr/refstate"
    assert ref.resolved_topic == "igris_c_IG07/gmr/refstate"
    assert len(publishers) == 1 and publishers[0].topic == "gmr/refstate" and publishers[0].qos == "sensor-data"

    joints = np.linspace(-1.0, 1.0, dds_io.REF_JOINT_COUNT)
    ref.publish(joints, stamp=1_700_000_000.25)
    (message,) = publishers[0].written
    assert [joint.value for joint in message.joints] == pytest.approx(joints.tolist())
    assert message.header_value.stamp_value.sec_value == 1_700_000_000
    assert message.header_value.stamp_value.nanosec_value == 250_000_000
    assert ref.published == 1 and ref.rejected == 0

    # Rounding at the second boundary must not produce nanosec == 1e9.
    ref.publish(joints, stamp=1_700_000_000.9999999999)
    stamp = publishers[0].written[-1].header_value.stamp_value
    assert (stamp.sec_value, stamp.nanosec_value) == (1_700_000_001, 0)

    with pytest.raises(ValueError):
        ref.publish(joints[:30], stamp=0.0)
    with pytest.raises(ValueError):
        ref.publish(np.r_[joints[:30], np.nan], stamp=0.0)
    ref.close()
    assert publishers[0].stops == 1


def test_ref_state_publisher_requires_master_arm_types():
    sdk, _channel, _ = _fake_sdk()
    dds = dds_io.DdsChannel.create(domain_id=0, namespace="igris_c_IG07", sdk=sdk)
    broken = SimpleNamespace(
        ChannelFactory=sdk.ChannelFactory, MasterArmStatePublisher=sdk.MasterArmStatePublisher, QosProfile=sdk.QosProfile
    )
    with pytest.raises(SystemExit, match="MasterArmState"):
        dds_io.RefStateDdsPublisher(dds_io.DdsChannel.create(domain_id=0, namespace="igris_c_IG07", sdk=broken))


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


# -- LowState / RobotState readers (convergence ramp inputs) ---------------------------------------


def test_lowstate_subscriber_keeps_the_newest_31_joints_with_age():
    sdk, _channel, _ = _fake_sdk()
    dds = dds_io.DdsChannel.create(domain_id=0, namespace="igris_c_IG07", sdk=sdk)
    reader = dds_io.LowStateSubscriber(dds)
    fake = sdk.subscribers[-1]
    assert fake.topic == "lowstate" and fake.qos == "sensor-data" and reader.resolved_topic == "igris_c_IG07/lowstate"
    assert reader.latest() is None
    fake.callback(_FakeLowState(np.arange(40, dtype=float)))  # extra joints beyond 31 are ignored
    sample = reader.latest()
    assert sample is not None
    q, age = sample
    assert q.shape == (31,) and np.allclose(q, np.arange(31)) and 0.0 <= age < 1.0 and reader.received == 1
    fake.callback(_FakeLowState(np.zeros(5)))  # short message: counted as an error, previous sample kept
    assert reader.errors == 1 and np.allclose(reader.latest()[0], np.arange(31))
    reader.close()
    assert fake.stops == 1


def test_robot_mode_subscriber_keeps_the_newest_mode():
    sdk, _channel, _ = _fake_sdk()
    dds = dds_io.DdsChannel.create(domain_id=0, namespace="igris_c_IG07", sdk=sdk)
    reader = dds_io.RobotModeSubscriber(dds, "robotstate")
    fake = sdk.subscribers[-1]
    assert fake.topic == "robotstate" and fake.qos == "default"
    assert reader.latest() is None
    fake.callback(SimpleNamespace(state=lambda: 3))
    mode, age = reader.latest()
    assert mode == 3 and 0.0 <= age < 1.0
    fake.callback(SimpleNamespace(state=lambda: "bad"))
    assert reader.errors == 1 and reader.latest()[0] == 3
    reader.close()
    assert fake.stops == 1
