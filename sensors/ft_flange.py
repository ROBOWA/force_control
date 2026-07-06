"""Flange FT source: the Panda's own external-wrench estimate, read in-loop.

No ROS, no shared memory, no background thread.  libfranka reports an
external-wrench estimate on every RobotState delivered to the 1 kHz torque
callback, so — unlike FTRosSource / FTShmSource — there is nothing to
subscribe to and no cache to keep warm: the value is simply pulled off the
`robot_state` object each tick.

Because the estimate is derived from the joint-torque sensors (not a dedicated
load cell) it is NOISY and carries a pose-dependent model-error bias.  Always
pair this source with core.ft_processor_flange.FlangeFTProcessor (low-pass +
optional despike) and tare the payload baseline before reading contact force.

Interface (compatible with FTMuJoCo / FTRosSource, plus update())::

    ft = FTFlangeSource(frame="base")
    ft.start()                        # no-op (no thread)
    sample = ft.update(robot_state)   # call at the TOP of the torque callback
    sample = ft.get_latest()          # returns the last update()'d sample
    ft.stop()                         # no-op

`update(robot_state)` must be called once per control tick; `get_latest()`
mirrors the other sources for any downstream code that expects the pull API.
"""

from __future__ import annotations
import numpy as np

from core.types import WrenchSample

# Sentinel returned before the first update() (should not occur in practice —
# the wrench is available on the very first RobotState).
_ZERO_SAMPLE = WrenchSample(t=0.0, wrench=np.zeros(6), seq=0, valid=False)


class FTFlangeSource:
    """Extracts the Panda flange external wrench from a libfranka RobotState.

    frame:
        "base"      -> O_F_ext_hat_K: external wrench in the robot BASE frame
                       (z up).  Already world-aligned, so NO MuJoCo world-frame
                       rotation is needed downstream — wrench[2] is the vertical
                       (contact) force in the same frame the state machine and
                       tip kinematics use.  This is the recommended setting.
        "stiffness" -> K_F_ext_hat_K: external wrench in the wrist/stiffness (K)
                       frame, which rotates with the arm.  Only meaningful if a
                       world-frame rotator is applied afterwards; NOT used by the
                       base-frame state machine, so avoid unless you know why.

    Sign convention is NOT applied here (that belongs to FlangeFTProcessor.sign);
    this source returns libfranka's raw estimate verbatim.  Per libfranka,
    F_ext is the wrench the environment exerts ON the robot, so a hanging tool
    reads base-frame Fz < 0 and a surface pushing up during a downward approach
    reads base-frame Fz > 0.
    """

    _FIELD = {"base": "O_F_ext_hat_K", "stiffness": "K_F_ext_hat_K"}

    def __init__(self, frame: str = "base"):
        frame = str(frame).lower()
        if frame not in self._FIELD:
            raise ValueError(
                f"frame must be 'base' or 'stiffness', got {frame!r}"
            )
        self._frame = frame
        self._field = self._FIELD[frame]
        self._seq = 0
        self._latest: WrenchSample = _ZERO_SAMPLE

    # ------------------------------------------------------------------
    # Common FT source interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """No-op: the wrench arrives with every RobotState; no thread needed."""

    def stop(self) -> None:
        """No-op."""

    def get_latest(self) -> WrenchSample:
        """Return the last sample cached by update(). Non-blocking.

        Returns a zero sample with valid=False only if update() has never been
        called (i.e. before the first control tick).
        """
        return self._latest

    # ------------------------------------------------------------------
    # Flange-specific: pull the wrench off the live RobotState
    # ------------------------------------------------------------------

    def update(self, robot_state, t: float | None = None) -> WrenchSample:
        """Extract the flange wrench from `robot_state`; cache and return it.

        Call once at the top of the 1 kHz torque callback.  Every call yields a
        fresh, valid sample with an incrementing seq, so the state-machine TARE
        (which dedups on seq) averages exactly one reading per control tick.

        Args:
            robot_state: the pylibfranka RobotState passed to the callback.
            t:           control-loop time [s] to stamp the sample with; the
                         seq (not t) is what TARE dedups on, so t is for logging.
        """
        w = np.asarray(getattr(robot_state, self._field), dtype=np.float64)
        self._seq += 1
        self._latest = WrenchSample(
            t=float(t) if t is not None else 0.0,
            wrench=w.copy(),
            seq=self._seq,
            valid=True,
        )
        return self._latest
