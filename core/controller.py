"""ControllerCore: pure-Python Cartesian impedance + force controller.

No MuJoCo, pylibfranka, or ROS imports.
"""

from __future__ import annotations

import numpy as np
from .types import RobotStateLite, TargetSample, WrenchSample
from .interpolation import interpolate_target
from .safety import saturate_torque_rate, check_target_staleness, check_ft_staleness


class ControllerCore:
    """Stateful 1 kHz Cartesian impedance controller.

    Call step() once per control tick. Thread-safe reads of cached state
    are the caller's responsibility.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: dict with keys:
                max_torque_rate   (7,) [Nm/ms]
                null_stiffness    float [Nm/rad]
                null_q_d          (7,) desired null-space joint config
                force_pi_kp       float
                force_pi_ki       float
                force_pi_max      float  anti-windup clamp [N]
                target_stale_ms   float
                ft_stale_ms       float
        """
        self._cfg = config
        self._tau_prev = np.zeros(7)
        self._force_pi_integral = 0.0
        self._initialized = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, tau_init: np.ndarray | None = None) -> None:
        """Reset integrators and previous-torque buffer."""
        self._tau_prev = np.zeros(7) if tau_init is None else tau_init.copy()
        self._force_pi_integral = 0.0
        self._initialized = True

    def step(
        self,
        state: RobotStateLite,
        target: TargetSample,
        wrench: WrenchSample,
        dt: float,
        contact: bool = False,
    ) -> np.ndarray:
        """Compute joint torque command for one 1 kHz tick.

        Args:
            state:   current robot state
            target:  latest target sample (may be from 400 Hz outer loop)
            wrench:  latest FT sample
            dt:      control timestep [s], typically 1e-3
            contact: True when contact state machine is in contact

        Returns:
            tau_cmd: shape (7,) [Nm]
        """
        # TODO: implement full controller
        # 1. Extrapolate/interpolate target to current tick
        # 2. Compute Cartesian pose error
        # 3. Compute impedance wrench
        # 4. Optionally add force PI / admittance
        # 5. Map to joint torques via J.T
        # 6. Add coriolis
        # 7. Add null-space torque
        # 8. Torque-rate saturation
        # 9. Cache tau, return

        target_age = state.t - target.t
        ft_age = state.t - wrench.t

        target_ok = check_target_staleness(target_age, self._cfg.get("target_stale_ms", 50.0))
        ft_ok = check_ft_staleness(ft_age, self._cfg.get("ft_stale_ms", 20.0))

        tgt = interpolate_target(target, state.t)

        tau = self._impedance(state, tgt)
        tau += self._nullspace(state)
        tau += state.coriolis

        if contact and ft_ok:
            tau += self._force_pi(wrench, tgt, dt)

        tau = saturate_torque_rate(tau, self._tau_prev, self._cfg.get("max_torque_rate", 1.0))
        self._tau_prev = tau.copy()
        return tau

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _impedance(self, state: RobotStateLite, tgt: TargetSample) -> np.ndarray:
        """Cartesian impedance: F_task = K*e + D*(v_d - v), tau = J.T @ F_task."""
        # TODO: implement
        # position error
        # orientation error (rotation matrix → axis-angle or log map)
        # velocity error
        # F_task = K * e6 + D * (v_d - v) + F_ff
        # return J.T @ F_task
        return np.zeros(7)

    def _nullspace(self, state: RobotStateLite) -> np.ndarray:
        """Null-space joint centering torque."""
        # TODO: implement
        # tau_null = (I - J.T @ J_pinv.T) @ (-k_ns * (q - q_d) - d_ns * dq)
        return np.zeros(7)

    def _force_pi(self, wrench: WrenchSample, tgt: TargetSample, dt: float) -> np.ndarray:
        """Force PI along the contact normal (z-axis of tool frame)."""
        # TODO: implement
        # error = F_desired - wrench.wrench[2]
        # integral with anti-windup
        # correction mapped through Jacobian
        return np.zeros(7)


# ---------------------------------------------------------------------------
# Orientation error (pure numpy — keeps this module MuJoCo-free)
# ---------------------------------------------------------------------------

def orientation_error(R_des: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Small-angle world-frame orientation error between two rotations.

    Returns 0.5 * vee(R_des R^T - (R_des R^T)^T), the standard sin(theta)*axis
    approximation expressed in the world frame — consistent with a world-frame
    rotational Jacobian (mj_jacSite) and the convention used by solve_ik_dls.
    """
    Re = R_des @ R.T
    return 0.5 * np.array([
        Re[2, 1] - Re[1, 2],
        Re[0, 2] - Re[2, 0],
        Re[1, 0] - Re[0, 1],
    ])


# ---------------------------------------------------------------------------
# Milestone 2: Cartesian impedance controller (tip task wrench -> joint torque)
# ---------------------------------------------------------------------------

class CartesianImpedanceController:
    """Stateless diagonal Cartesian impedance controller.

    Computes a task wrench from pose/velocity error and maps it to joint
    torques through the transpose Jacobian::

        F   = Kp_pos * (x_des - x)  + Kd_pos * (dx_des - v)
        T   = Kp_ori * e_ori        + Kd_ori * (w_des - w)
        tau = J.T @ [F, T]

    It adds NO gravity/coriolis term — the backend supplies arm gravity (sim:
    G_zero / libfranka on hardware) and payload-gravity compensation
    separately, exactly as in the joint-PD path.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: controller dict with 6-vectors K and D ordered
                    [x, y, z, rx, ry, rz]; split into translational (first 3)
                    and rotational (last 3) diagonal gains.
        """
        K = np.asarray(config["K"], dtype=float)
        D = np.asarray(config["D"], dtype=float)
        if K.shape != (6,) or D.shape != (6,):
            raise ValueError("controller.K and controller.D must each be length 6")
        self.Kp_pos, self.Kp_ori = K[:3], K[3:]
        self.Kd_pos, self.Kd_ori = D[:3], D[3:]

    def compute(
        self,
        x:      np.ndarray,   # (3,) current tip position [m]
        R:      np.ndarray,   # (3,3) current tip orientation
        v:      np.ndarray,   # (3,) current tip linear velocity [m/s]
        w:      np.ndarray,   # (3,) current tip angular velocity [rad/s]
        J:      np.ndarray,   # (6,7) world-frame tip Jacobian
        x_des:  np.ndarray,   # (3,) desired tip position [m]
        dx_des: np.ndarray,   # (3,) desired tip linear velocity [m/s]
        R_des:  np.ndarray,   # (3,3) desired tip orientation
        w_des:  np.ndarray,   # (3,) desired tip angular velocity [rad/s]
    ) -> np.ndarray:
        """Return joint torque command, shape (7,) [Nm]."""
        e_pos = x_des - x
        e_ori = orientation_error(R_des, R)
        F = self.Kp_pos * e_pos + self.Kd_pos * (dx_des - v)
        T = self.Kp_ori * e_ori + self.Kd_ori * (w_des - w)
        return J.T @ np.concatenate([F, T])


# ---------------------------------------------------------------------------
# Milestone 1: joint-space PD controller
# ---------------------------------------------------------------------------

class JointPDController:
    """Stateless joint-space PD controller for Milestone 1.

    Computes PD tracking torques given desired and actual joint state.
    Does not know about MuJoCo, pylibfranka, IK, or gravity compensation —
    those are handled by the backend or state machine.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: dict with keys:
                kp  list[float] shape (7,) — position gains [Nm/rad]
                kd  list[float] shape (7,) — velocity gains [Nm·s/rad]
        """
        self.kp = np.asarray(config["kp"], dtype=float)
        self.kd = np.asarray(config["kd"], dtype=float)

    def compute(
        self,
        q:      np.ndarray,
        dq:     np.ndarray,
        q_des:  np.ndarray,
        dq_des: np.ndarray,
    ) -> np.ndarray:
        """Return PD torque command, shape (7,) [Nm].

        Args:
            q:      current joint positions [rad]
            dq:     current joint velocities [rad/s]
            q_des:  desired joint positions [rad]
            dq_des: desired joint velocities [rad/s]
        """
        q      = np.asarray(q,      dtype=float)
        dq     = np.asarray(dq,     dtype=float)
        q_des  = np.asarray(q_des,  dtype=float)
        dq_des = np.asarray(dq_des, dtype=float)
        return self.kp * (q_des - q) + self.kd * (dq_des - dq)
