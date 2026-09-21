"""ROS 2 frame publisher: QoS/domain contract with a fake rclpy, plus a real-rclpy integration test."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from igris_c_gmr import ros_publisher


class _FakeString:
    def __init__(self):
        self.data = ""


class _FakePublisher:
    def __init__(self, msg_type, topic, qos):
        self.msg_type, self.topic, self.qos = msg_type, topic, qos
        self.messages = []

    def publish(self, message):
        self.messages.append(message.data)

    def get_subscription_count(self):
        return 2


class _FakeNode:
    def __init__(self, name, **kwargs):
        self.name, self.kwargs = name, kwargs
        self.publishers = []
        self.destroyed = False

    def create_publisher(self, msg_type, topic, qos):
        publisher = _FakePublisher(msg_type, topic, qos)
        self.publishers.append(publisher)
        return publisher

    def destroy_publisher(self, publisher):
        assert publisher in self.publishers

    def destroy_node(self):
        self.destroyed = True


class _FakeQos:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _fake_ros():
    calls = {"init": None, "shutdown": 0, "nodes": []}

    def _init(args=None, domain_id=None, signal_handler_options=None):
        calls["init"] = (args, domain_id, signal_handler_options)

    def _create_node(name, **kwargs):
        node = _FakeNode(name, **kwargs)
        calls["nodes"].append(node)
        return node

    def _shutdown():
        calls["shutdown"] += 1

    ros = {
        "rclpy": SimpleNamespace(init=_init, create_node=_create_node, shutdown=_shutdown),
        "String": _FakeString,
        "QoSProfile": _FakeQos,
        "ReliabilityPolicy": SimpleNamespace(RELIABLE="reliable", BEST_EFFORT="best_effort"),
        "DurabilityPolicy": SimpleNamespace(VOLATILE="volatile"),
        "HistoryPolicy": SimpleNamespace(KEEP_LAST="keep_last"),
        "SignalHandlerOptions": SimpleNamespace(NO="no-handlers"),
    }
    return ros, calls


def test_publisher_contract_with_fake_rclpy():
    ros, calls = _fake_ros()
    publisher = ros_publisher.RetargetFrameRosPublisher(ros=ros)
    assert calls["init"] == (None, 91, "no-handlers")  # inference domain; producer owns the signals
    (node,) = calls["nodes"]
    assert node.name == "igris_c_gmr_producer"
    assert node.kwargs == {"enable_rosout": False, "start_parameter_services": False}
    (pub,) = node.publishers
    assert pub.topic == "/gmr/teleop/retarget_frame" and pub.msg_type is _FakeString
    assert pub.qos.kwargs == {"depth": 1, "history": "keep_last", "reliability": "reliable", "durability": "volatile"}

    publisher.publish('{"dof_pos":[[0.0]]}')
    publisher.publish("second")
    assert pub.messages == ['{"dof_pos":[[0.0]]}', "second"] and publisher.published == 2
    assert publisher.subscription_count() == 2
    publisher.close()
    publisher.close()
    assert node.destroyed and calls["shutdown"] == 1


def test_publisher_custom_domain_topic_and_name():
    ros, calls = _fake_ros()
    ros_publisher.RetargetFrameRosPublisher(domain_id=7, topic="/x/frame", node_name="probe", ros=ros)
    assert calls["init"][1] == 7
    assert calls["nodes"][0].name == "probe" and calls["nodes"][0].publishers[0].topic == "/x/frame"


def test_defaults_match_inference_config():
    assert ros_publisher.DEFAULT_ROS_DOMAIN_ID == 91
    assert ros_publisher.DEFAULT_ROS_TOPIC == "/gmr/teleop/retarget_frame"


@pytest.mark.ros
def test_real_rclpy_roundtrip():
    """One String travels producer -> subscriber on a throwaway domain (needs the env's rclpy)."""

    rclpy = pytest.importorskip("rclpy")
    from std_msgs.msg import String

    os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
    publisher = ros_publisher.RetargetFrameRosPublisher(domain_id=117, topic="/igris_c_gmr_test/frame", node_name="igris_c_gmr_test_pub")
    received: list[str] = []
    context = rclpy.Context()
    context.init(domain_id=117)
    node = rclpy.create_node("igris_c_gmr_test_sub", context=context)
    node.create_subscription(String, "/igris_c_gmr_test/frame", lambda msg: received.append(msg.data), 1)
    executor = rclpy.executors.SingleThreadedExecutor(context=context)
    executor.add_node(node)
    try:
        for _ in range(200):
            publisher.publish('{"dof_pos":[[1.0]]}')
            executor.spin_once(timeout_sec=0.02)
            if received:
                break
        assert received and received[0] == '{"dof_pos":[[1.0]]}'
        assert publisher.subscription_count() >= 1
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        context.shutdown()
        publisher.close()
