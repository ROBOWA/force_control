"""Contact state machine: APPROACH → CONTACT → SETTLE → SLIDE → LIFT → DONE.

No MuJoCo, pylibfranka, or ROS imports.
"""

from __future__ import annotations
from enum import Enum, auto
import numpy as np
from .types import RobotStateLite, WrenchSample, TargetSample


class Phase(Enum):
    APPROACH = auto()
    CONTACT  = auto()
    SETTLE   = auto()
    SLIDE    = auto()
    LIFT     = auto()
    DONE     = auto()


class StateMachine:
    """Drives high-level phase transitions and generates TargetSamples.

    The outer (400 Hz) loop calls update() each tick. The output
    TargetSample is written to a thread-safe buffer read by the 1 kHz
    callback via ControllerCore.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: dict with keys:
                contact_fz_threshold    float [N]  Fz magnitude to detect contact
                settle_duration         float [s]
                slide_speed             float [m/s] tangential speed during SLIDE
                slide_distance          float [m]   total sliding distance
                lift_height             float [m]
                lift_speed              float [m/s]
                approach_speed          float [m/s]
                target_z_contact        float [m]   z position of surface in base frame
                F_normal_desired        float [N]   desired normal force during SLIDE
        """
        self._cfg = config
        self.phase = Phase.APPROACH
        self._phase_t0: float = 0.0
        self._slide_distance_covered: float = 0.0
        self._contact_pos: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def in_contact(self) -> bool:
        return self.phase in (Phase.CONTACT, Phase.SETTLE, Phase.SLIDE)

    def update(
        self,
        t: float,
        state: RobotStateLite,
        wrench: WrenchSample,
        current_target: TargetSample,
    ) -> TargetSample:
        """Advance the state machine and return the next TargetSample.

        Called at ~400 Hz by the outer loop thread.
        """
        fz = wrench.wrench[2]  # normal force (sign convention: positive = compression)

        if self.phase == Phase.APPROACH:
            return self._handle_approach(t, state, wrench, current_target, fz)
        elif self.phase == Phase.CONTACT:
            return self._handle_contact(t, state, wrench, current_target)
        elif self.phase == Phase.SETTLE:
            return self._handle_settle(t, state, wrench, current_target)
        elif self.phase == Phase.SLIDE:
            return self._handle_slide(t, state, wrench, current_target)
        elif self.phase == Phase.LIFT:
            return self._handle_lift(t, state, wrench, current_target)
        else:  # DONE
            return current_target

    def reset(self, t: float) -> None:
        self.phase = Phase.APPROACH
        self._phase_t0 = t
        self._slide_distance_covered = 0.0
        self._contact_pos = None

    # ------------------------------------------------------------------
    # Phase handlers (each returns a TargetSample)
    # ------------------------------------------------------------------

    def _handle_approach(
        self,
        t: float,
        state: RobotStateLite,
        wrench: WrenchSample,
        current_target: TargetSample,
        fz: float,
    ) -> TargetSample:
        """Descend at constant speed until Fz threshold is exceeded."""
        # TODO: implement
        # move x_d downward at approach_speed * dt
        # if abs(fz) > contact_fz_threshold → transition to CONTACT
        return current_target

    def _handle_contact(
        self,
        t: float,
        state: RobotStateLite,
        wrench: WrenchSample,
        current_target: TargetSample,
    ) -> TargetSample:
        """Record contact position, hold, wait one tick before SETTLE."""
        # TODO: implement
        if self._contact_pos is None:
            self._contact_pos = state.O_T_EE[:3, 3].copy()
            self._phase_t0 = t
        self._transition(Phase.SETTLE, t)
        return current_target

    def _handle_settle(
        self,
        t: float,
        state: RobotStateLite,
        wrench: WrenchSample,
        current_target: TargetSample,
    ) -> TargetSample:
        """Hold contact position for settle_duration, then transition to SLIDE."""
        # TODO: implement
        # if (t - self._phase_t0) > settle_duration → transition to SLIDE
        return current_target

    def _handle_slide(
        self,
        t: float,
        state: RobotStateLite,
        wrench: WrenchSample,
        current_target: TargetSample,
    ) -> TargetSample:
        """Slide tangentially at slide_speed while force PI regulates normal force."""
        # TODO: implement
        # advance x_d in tangential direction
        # accumulate slide_distance_covered
        # if covered >= slide_distance → transition to LIFT
        return current_target

    def _handle_lift(
        self,
        t: float,
        state: RobotStateLite,
        wrench: WrenchSample,
        current_target: TargetSample,
    ) -> TargetSample:
        """Lift EE vertically to lift_height above contact, then DONE."""
        # TODO: implement
        return current_target

    def _transition(self, new_phase: Phase, t: float) -> None:
        self.phase = new_phase
        self._phase_t0 = t
