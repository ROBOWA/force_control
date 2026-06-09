"""Shared data types passed between backends and the controller core.

No MuJoCo, pylibfranka, or ROS imports allowed here.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RobotStateLite:
    """Minimal robot state consumed by ControllerCore."""
    t: float
    q: np.ndarray       # shape (7,) joint positions [rad]
    dq: np.ndarray      # shape (7,) joint velocities [rad/s]
    O_T_EE: np.ndarray  # shape (4,4) base→EE homogeneous transform
    J: np.ndarray       # shape (6,7) geometric Jacobian
    coriolis: np.ndarray  # shape (7,) coriolis + gravity torques [Nm]


@dataclass(frozen=True)
class WrenchSample:
    """One FT sensor sample (real or simulated)."""
    t: float
    wrench: np.ndarray  # shape (6,) [Fx,Fy,Fz,Tx,Ty,Tz] in control frame [N, Nm]
    seq: int
    valid: bool = True  # False when no data has arrived yet (hardware) or sensor absent


@dataclass(frozen=True)
class ControlCommand:
    """One control reference emitted by a state machine each tick.

    A command is either joint-space or Cartesian, selected by `mode`:

        "joint"     — track q_des / dq_des with a joint PD controller.
        "cartesian" — track (x_des, dx_des, R_des, w_des) with a Cartesian
                      impedance controller (tau = J.T @ wrench).
        "failed"    — terminal error; the backend should hold safely / stop.

    Only the fields relevant to `mode` are populated; the rest stay None.
    """
    mode: str
    # joint-space targets (mode == "joint")
    q_des:  np.ndarray | None = None
    dq_des: np.ndarray | None = None
    # Cartesian targets (mode == "cartesian"), all world-frame
    x_des:  np.ndarray | None = None   # (3,) tip position [m]
    dx_des: np.ndarray | None = None   # (3,) tip linear velocity [m/s]
    R_des:  np.ndarray | None = None   # (3,3) tip orientation
    w_des:  np.ndarray | None = None   # (3,) tip angular velocity [rad/s]


@dataclass(frozen=True)
class TargetSample:
    """One target sample produced by the outer (400 Hz) loop."""
    t: float
    x_d: np.ndarray    # shape (3,) desired EE position [m]
    dx_d: np.ndarray   # shape (3,) desired EE linear velocity [m/s]
    R_d: np.ndarray    # shape (3,3) desired EE orientation
    w_d: np.ndarray    # shape (3,) desired EE angular velocity [rad/s]
    F_ff: np.ndarray   # shape (6,) feedforward wrench [N, Nm]
    K: np.ndarray      # shape (6,) Cartesian stiffness diagonal
    D: np.ndarray      # shape (6,) Cartesian damping diagonal
    seq: int
