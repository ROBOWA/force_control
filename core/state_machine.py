"""State machines for force-control.

Contains two state machines:

    JointMoveStateMachine   — Milestone 1: joint-space IK + min-jerk move
    StateMachine            — future: Cartesian contact/force/slide phases

No MuJoCo viewer, pylibfranka, or ROS imports.
MuJoCo model/data are used only inside JointMoveStateMachine for IK FK calls.
"""

from __future__ import annotations
from enum import Enum, auto
import threading
import numpy as np
import mujoco
from .types import RobotStateLite, WrenchSample, TargetSample
from .ik import min_jerk, solve_ik_dls, euler_xyz_to_R


# ---------------------------------------------------------------------------
# Milestone 1: joint-space move state machine
# ---------------------------------------------------------------------------

class JointMoveState(Enum):
    HOLD             = auto()   # holds whatever q_hold is currently set to
    SOLVING_IK       = auto()   # synchronous; never seen by update()
    WAIT_USER_CONFIRM = auto()
    MOVE_JOINT       = auto()
    FAILED           = auto()


class JointMoveStateMachine:
    """Drives the joint-move sequence for Milestone 1.

    Typical usage::

        sm = JointMoveStateMachine(config)
        sm.start(model, data)          # synchronous IK + launches Enter-waiter thread
        # then in the sim loop:
        q_des, dq_des = sm.update(t, q)
        if q_des is None:
            break                      # FAILED state — stop sim loop
    """

    def __init__(self, config: dict):
        """
        Args:
            config: full sim config dict; reads config["ik"] for IK parameters.
        """
        self._cfg_ik = config["ik"]
        self.state = JointMoveState.HOLD

        self.q_hold:  np.ndarray | None = None
        self.q_goal:  np.ndarray | None = None
        self.q_start: np.ndarray | None = None

        self.move_time:       float = float(self._cfg_ik.get("move_time", 5.0))
        self.move_start_time: float = 0.0

        self._enter_pressed  = threading.Event()
        self._confirm_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, mj_model, mj_data) -> None:
        """Called once after MuJoCo model/data are loaded and reset to home.

        Captures q_hold, solves IK synchronously using a scratch MjData so
        the live simulation state is never corrupted, then transitions to
        WAIT_USER_CONFIRM (or FAILED) and launches the Enter-waiter thread.
        """
        self.q_hold = mj_data.qpos[:7].copy()
        self.state  = JointMoveState.SOLVING_IK
        self._solve_ik(mj_model, mj_data)

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def update(
        self,
        t: float,
        q_current: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return (q_des, dq_des) for the current sim tick.

        Returns (None, None) when the state machine has FAILED — the backend
        should break the simulation loop immediately.
        """
        if self.state == JointMoveState.HOLD:
            return self.q_hold.copy(), np.zeros(7)

        if self.state == JointMoveState.WAIT_USER_CONFIRM:
            if self._enter_pressed.is_set():
                # Capture actual robot q at the moment the user triggers motion.
                self.q_start         = q_current.copy()
                self.move_start_time = t
                self.state           = JointMoveState.MOVE_JOINT
                print("Starting movement...")
            # Hold for this tick whether or not we just transitioned.
            return self.q_hold.copy(), np.zeros(7)

        if self.state == JointMoveState.MOVE_JOINT:
            t_elapsed = t - self.move_start_time
            if t_elapsed <= self.move_time:
                s, ds  = min_jerk(t_elapsed, self.move_time)
                q_des  = self.q_start + s  * (self.q_goal - self.q_start)
                dq_des = ds * (self.q_goal - self.q_start)
                return q_des, dq_des
            else:
                self.q_hold = self.q_goal.copy()
                self.state  = JointMoveState.HOLD
                print("Reached HOLD")
                return self.q_hold.copy(), np.zeros(7)

        if self.state == JointMoveState.FAILED:
            return self.q_hold.copy(), np.zeros(7)

        # SOLVING_IK is synchronous — update() should never see this.
        return self.q_hold.copy(), np.zeros(7)

    # ------------------------------------------------------------------
    # IK (synchronous, called from start())
    # ------------------------------------------------------------------

    def _solve_ik(self, mj_model, mj_data) -> None:
        """Solve IK using a scratch MjData; never writes to the live mj_data."""
        site_name  = self._cfg_ik["site_name"]
        target_pos = np.array(self._cfg_ik["target_pos"])

        # Scratch copy so IK FK calls do not corrupt the live sim state.
        ik_data = mujoco.MjData(mj_model)
        ik_data.qpos[:] = mj_data.qpos.copy()
        ik_data.qvel[:] = 0.0
        mujoco.mj_forward(mj_model, ik_data)

        site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            print(f"ERROR: site '{site_name}' not found in model. State → FAILED.")
            self.state = JointMoveState.FAILED
            return

        print(f"Using site:            {site_name}")
        print(f"Initial site position: {ik_data.site_xpos[site_id].round(4)}")
        print(f"Target site position:  {target_pos}")

        target_R = self._resolve_target_R(ik_data, site_id)
        print(f"Target orientation R:\n{target_R.round(4)}")
        q_init = ik_data.qpos[:7].copy()

        q_goal, n_iter, err, converged = solve_ik_dls(
            mj_model, ik_data, site_id, q_init, target_pos, target_R
        )

        print(f"IK converged: {converged}  (iter={n_iter}, err={err:.5f})")
        print(f"q_hold: {self.q_hold.round(4)}")

        if not converged:
            print(f"IK failed (err={err:.5f}). State → FAILED.")
            self.state = JointMoveState.FAILED
            return

        self.q_goal = q_goal.copy()
        print(f"q_goal: {self.q_goal.round(4)}")

        self.state = JointMoveState.WAIT_USER_CONFIRM
        self._confirm_thread = threading.Thread(
            target=self._wait_for_enter, daemon=True
        )
        self._confirm_thread.start()

    def _resolve_target_R(self, ik_data, site_id: int) -> np.ndarray:
        """Return the 3×3 target orientation matrix from config.

        Always starts from the current site orientation, then applies
        target_euler_xyz as an additional intrinsic XYZ rotation in the
        site's own body frame:

            target_R = R_current @ R_delta

        If target_euler_xyz is [0, 0, 0] (or omitted), orientation is unchanged.
        """
        R_current = ik_data.site_xmat[site_id].copy().reshape(3, 3)
        euler = self._cfg_ik.get("target_euler_xyz", [0.0, 0.0, 0.0])
        R_delta = euler_xyz_to_R(euler)
        return R_current @ R_delta

    def _wait_for_enter(self) -> None:
        input("\nIK solved. Press Enter to start moving...")
        self._enter_pressed.set()


# ---------------------------------------------------------------------------
# Future: Cartesian contact / force / slide state machine
# ---------------------------------------------------------------------------


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
