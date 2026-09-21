"""ROS 2 publisher for the motion-reference frame consumed by ``igris_c_inference_node``.

Replaces gmr's ``redis_to_ros2_bridge.py``: one ``std_msgs/String`` per frame
on ``/gmr/teleop/retarget_frame``. Publish-only, so no executor is spun;
discovery and delivery run on the rmw threads.

QoS: RELIABLE, KEEP_LAST depth 1, VOLATILE (what the bridge used). The
consumer subscribes best-effort depth 1; a reliable publisher matches both
best-effort and reliable subscribers, a best-effort one would not match a
reliable subscriber.

``rclpy`` is imported lazily so unit tests can inject a fake ROS layer.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("igris_c_gmr.ros")

DEFAULT_ROS_DOMAIN_ID = 91  # must match the igris_c_inference command.yaml source.params.domain_id
DEFAULT_ROS_TOPIC = "/gmr/teleop/retarget_frame"
DEFAULT_NODE_NAME = "igris_c_gmr_producer"


def _import_ros() -> dict[str, Any]:
    try:
        import rclpy
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from rclpy.signals import SignalHandlerOptions
        from std_msgs.msg import String
    except ImportError as exc:
        raise SystemExit(
            f"rclpy / std_msgs not importable ({exc}). Run inside the pixi environment "
            "(`pixi run ...`): rclpy is a RoboStack package of that env, not a host /opt/ros install."
        ) from exc
    return {
        "rclpy": rclpy,
        "String": String,
        "QoSProfile": QoSProfile,
        "ReliabilityPolicy": ReliabilityPolicy,
        "DurabilityPolicy": DurabilityPolicy,
        "HistoryPolicy": HistoryPolicy,
        "SignalHandlerOptions": SignalHandlerOptions,
    }


class RetargetFrameRosPublisher:
    """Owns the rclpy context, one node and one String publisher."""

    def __init__(
        self,
        *,
        domain_id: int = DEFAULT_ROS_DOMAIN_ID,
        topic: str = DEFAULT_ROS_TOPIC,
        node_name: str = DEFAULT_NODE_NAME,
        ros: dict[str, Any] | None = None,
    ) -> None:
        self._ros = ros if ros is not None else _import_ros()
        rclpy = self._ros["rclpy"]
        self.domain_id = int(domain_id)
        self.topic = str(topic)
        self.node_name = str(node_name)
        # The producer owns SIGINT/SIGTERM; rclpy must not install its own handlers.
        rclpy.init(args=None, domain_id=self.domain_id, signal_handler_options=self._ros["SignalHandlerOptions"].NO)
        self._node = rclpy.create_node(self.node_name, enable_rosout=False, start_parameter_services=False)
        qos = self._ros["QoSProfile"](
            depth=1,
            history=self._ros["HistoryPolicy"].KEEP_LAST,
            reliability=self._ros["ReliabilityPolicy"].RELIABLE,
            durability=self._ros["DurabilityPolicy"].VOLATILE,
        )
        self._publisher = self._node.create_publisher(self._ros["String"], self.topic, qos)
        self._string_type = self._ros["String"]
        self.published = 0
        self._closed = False

    def publish(self, payload: str) -> None:
        message = self._string_type()
        message.data = payload
        self._publisher.publish(message)
        self.published += 1

    def subscription_count(self) -> int:
        """Matched subscribers (the inference node's motion terms count as 2)."""

        try:
            return int(self._publisher.get_subscription_count())
        except Exception:
            return -1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        rclpy = self._ros["rclpy"]
        try:
            self._node.destroy_publisher(self._publisher)
            self._node.destroy_node()
        finally:
            try:
                rclpy.shutdown()
            except Exception as exc:  # already shut down elsewhere
                logger.warning("rclpy shutdown: %s", exc)


__all__ = ["DEFAULT_NODE_NAME", "DEFAULT_ROS_DOMAIN_ID", "DEFAULT_ROS_TOPIC", "RetargetFrameRosPublisher"]
