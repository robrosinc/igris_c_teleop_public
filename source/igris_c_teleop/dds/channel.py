"""igris_c_sdk (CycloneDDS) channel: lazy SDK import, struct lookup, the process-wide
``ChannelFactory`` wrapper and the topic-naming contract.

The SDK binding is a Python 3.12 wheel (``task wheel`` in ``thirdparty/igris_c_sdk``);
it is imported lazily so the producer can run ROS-only with ``--no_dds`` in an
environment without it. Publishers, readers and the control-mode client live in the
sibling modules of this package; the ordering constraints below bind all of them.

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

logger = logging.getLogger("igris_c_teleop.dds")

BODY_JOINT_COUNT = 31  # MJCF hinge order == igris_c_sdk 31-motor order (robros_igris_c_v2)
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


__all__ = ["BODY_JOINT_COUNT", "NO_DDS_HINT", "TOPIC_NAMINGS", "DdsChannel", "import_sdk", "sdk_type"]
