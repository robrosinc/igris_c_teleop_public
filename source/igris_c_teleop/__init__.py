"""igris_c_teleop: VRS1 headset teleoperation producer for the IGRIS-C policy runtime.

VRS1 (vr_system headset) -> GMR retargeting -> ROS 2 motion-reference frame for
``igris_c_inference_node``; hand command and controller-driven control-mode
requests go directly over igris_c_sdk DDS. No Redis, no Isaac, no hit-bench.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
