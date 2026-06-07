"""Safety utilities: torque-rate saturation, staleness checks.

No MuJoCo, pylibfranka, or ROS imports.
"""

import numpy as np


def saturate_torque_rate(
    tau_des: np.ndarray,
    tau_last: np.ndarray,
    max_delta: float,
) -> np.ndarray:
    """Clamp per-joint torque change to ±max_delta [Nm] per control tick.

    Franka is sensitive to torque discontinuities; this prevents large jumps
    that would trigger the robot's internal safety stop.

    Args:
        tau_des:   desired torque command, shape (7,)
        tau_last:  torque sent in the previous tick, shape (7,)
        max_delta: maximum allowed change per tick [Nm]

    Returns:
        Saturated torque, shape (7,)
    """
    delta = np.clip(tau_des - tau_last, -max_delta, max_delta)
    return tau_last + delta


def check_target_staleness(age_s: float, threshold_ms: float) -> bool:
    """Return True if the target is fresh enough to trust.

    Args:
        age_s:        target age in seconds (now - target.t)
        threshold_ms: staleness threshold in milliseconds

    Returns:
        True if age_s * 1000 < threshold_ms
    """
    return (age_s * 1000.0) < threshold_ms


def check_ft_staleness(age_s: float, threshold_ms: float) -> bool:
    """Return True if the FT sample is fresh enough to use for force feedback.

    Args:
        age_s:        FT sample age in seconds
        threshold_ms: staleness threshold in milliseconds

    Returns:
        True if age_s * 1000 < threshold_ms
    """
    return (age_s * 1000.0) < threshold_ms


def clip_torque(tau: np.ndarray, max_tau: np.ndarray) -> np.ndarray:
    """Absolute torque clipping per joint.

    Args:
        tau:     desired torque, shape (7,)
        max_tau: per-joint maximum absolute torque, shape (7,)

    Returns:
        Clipped torque, shape (7,)
    """
    return np.clip(tau, -max_tau, max_tau)
