"""Target interpolation/extrapolation: 400 Hz outer loop → 1 kHz inner loop.

No MuJoCo, pylibfranka, or ROS imports.
"""

from __future__ import annotations

import numpy as np
from .types import TargetSample


# Maximum extrapolation window [s]. Beyond this, velocity term is frozen.
MAX_EXTRAP_AGE = 0.005  # 5 ms = one 200 Hz period as a conservative cap


def interpolate_target(target: TargetSample, t_now: float) -> TargetSample:
    """Bounded linear extrapolation of a 400 Hz target to the current 1 kHz tick.

    Given the latest outer-loop sample (t_k, x_k, dx_k), estimate the
    current position as:
        x_d = x_k + dx_k * clamp(t_now - t_k, 0, MAX_EXTRAP_AGE)

    This avoids overshoot when the next 400 Hz sample is late.

    Args:
        target:  latest TargetSample from the 400 Hz outer loop
        t_now:   current wall-clock time [s]

    Returns:
        A new TargetSample with updated x_d (all other fields passed through).
    """
    # TODO: implement orientation extrapolation via rotation-vector/slerp
    age = np.clip(t_now - target.t, 0.0, MAX_EXTRAP_AGE)
    x_d = target.x_d + target.dx_d * age
    # Orientation: hold constant for now (angular extrapolation is optional)
    R_d = target.R_d

    return TargetSample(
        t=t_now,
        x_d=x_d,
        dx_d=target.dx_d,
        R_d=R_d,
        w_d=target.w_d,
        F_ff=target.F_ff,
        K=target.K,
        D=target.D,
        seq=target.seq,
    )


def make_interpolated_buffer(samples: list[TargetSample], t_query: float) -> TargetSample:
    """True interpolation between two bracketing samples.

    Requires a 5–10 ms intentional delay so that both bracketing samples
    have arrived. Use only if you can tolerate the added latency.

    Args:
        samples: sorted list of recent TargetSamples (oldest first)
        t_query: time to interpolate to

    Returns:
        Interpolated TargetSample, or the nearest sample if bracket not found.
    """
    # TODO: implement
    # find i such that samples[i].t <= t_query < samples[i+1].t
    # alpha = (t_query - samples[i].t) / (samples[i+1].t - samples[i].t)
    # lerp position, slerp orientation
    raise NotImplementedError
