"""ControllerCore: pure-Python Cartesian impedance + force controller.

No MuJoCo, pylibfranka, or ROS imports.
"""

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
