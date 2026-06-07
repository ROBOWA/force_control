"""Python bindings for Franka Emika robot control library (libfranka).

pylibfranka provides high-level Python access to Franka Emika robots,
enabling real-time control with torque, position, and velocity commands.

Example:
    >>> import pylibfranka
    >>> robot = pylibfranka.Robot("robot.franka.de")
    >>> state = robot.read_once()
    >>> print(state.q)  # Current joint positions

"""

from ._pylibfranka import (
    CartesianPose,
    CartesianVelocities,
    CommandException,
    ControlException,
    ControllerMode,
    Duration,
    Errors,
    Frame,
    FrankaException,
    Gripper,
    GripperState,
    InvalidOperationException,
    JointPositions,
    JointVelocities,
    Model,
    NetworkException,
    RealtimeConfig,
    RealtimeException,
    Robot,
    RobotMode,
    RobotState,
    Torques,
    kDefaultCutoffFrequency,
    kMaxCutoffFrequency,
)

__version__ = "0.11.0"

__all__ = [
    "CartesianPose",
    "CartesianVelocities",
    "CommandException",
    "ControlException",
    "ControllerMode",
    "Duration",
    "Errors",
    "Frame",
    "FrankaException",
    "Gripper",
    "GripperState",
    "InvalidOperationException",
    "JointPositions",
    "JointVelocities",
    "Model",
    "NetworkException",
    "RealtimeConfig",
    "RealtimeException",
    "Robot",
    "RobotMode",
    "RobotState",
    "Torques",
    "kDefaultCutoffFrequency",
    "kMaxCutoffFrequency",
]
