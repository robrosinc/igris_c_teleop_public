"""igris_c_sdk (CycloneDDS) side of the producer: channel, robot-command publishers, state
readers and the control-mode service client.

HandCmd and the control-mode requests drive the real robot (ROBROS hardware-change approval
and validation before running against it); RefState is display-only for the headset;
LowState/RobotState are read-only inputs of the convergence ramp. The SDK is imported
lazily (``channel.import_sdk``) so ``--no_dds`` sessions need no wheel.
"""

from .channel import BODY_JOINT_COUNT, NO_DDS_HINT, TOPIC_NAMINGS, DdsChannel, import_sdk, sdk_type
from .control_mode import CONTROL_MODE_REQUEST_TOPIC, ControlModeClient
from .hand_cmd import DEFAULT_HAND_TOPIC, HandCmdDdsPublisher
from .ref_state import DEFAULT_REF_TOPIC, REF_JOINT_COUNT, RefStateDdsPublisher
from .state_readers import DEFAULT_LOWSTATE_TOPIC, DEFAULT_ROBOTSTATE_TOPIC, LowStateSubscriber, RobotModeSubscriber

__all__ = [
    "BODY_JOINT_COUNT",
    "CONTROL_MODE_REQUEST_TOPIC",
    "DEFAULT_HAND_TOPIC",
    "DEFAULT_LOWSTATE_TOPIC",
    "DEFAULT_REF_TOPIC",
    "DEFAULT_ROBOTSTATE_TOPIC",
    "NO_DDS_HINT",
    "REF_JOINT_COUNT",
    "TOPIC_NAMINGS",
    "ControlModeClient",
    "DdsChannel",
    "HandCmdDdsPublisher",
    "LowStateSubscriber",
    "RefStateDdsPublisher",
    "RobotModeSubscriber",
    "import_sdk",
    "sdk_type",
]
