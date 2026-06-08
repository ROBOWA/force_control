"""Joint-space move state machine for Milestone 1.

States:
    HOLD_CURRENT      → holding initial q while IK is being solved
    SOLVING_IK        → IK runs synchronously inside start(); never seen by update()
    WAIT_USER_CONFIRM → IK done, waiting for user to press Enter
    MOVE_JOINT        → executing min-jerk trajectory to q_goal
    HOLD_GOAL         → holding at q_goal
    FAILED            → IK failed; backend should stop the loop

No MuJoCo viewer, pylibfranka, or ROS imports.
MuJoCo model/data are used only for IK forward kinematics.
"""

from __future__ import annotations
from enum import Enum, auto
import threading
import numpy as np
import mujoco

from .ik import min_jerk, solve_ik_dls


class JointMoveState(Enum):
    HOLD_CURRENT      = auto()
    SOLVING_IK        = auto()
    WAIT_USER_CONFIRM = auto()
    MOVE_JOINT        = auto()
    HOLD_GOAL         = auto()
    FAILED            = auto()


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
            config: full sim config dict; reads config["ik"] and uses it for IK params.
        """
        self._cfg_ik = config["ik"]
        self.state = JointMoveState.HOLD_CURRENT

        self.q_hold:  np.ndarray | None = None
        self.q_goal:  np.ndarray | None = None
        self.q_start: np.ndarray | None = None

        self.move_time:       float = float(self._cfg_ik.get("move_time", 5.0))
        self.move_start_time: float = 0.0

        self._enter_pressed   = threading.Event()
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

        Args:
            t:         current simulation time [s]
            q_current: current joint positions, shape (7,)
        """
        if self.state == JointMoveState.HOLD_CURRENT:
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
            if t_elapsed < self.move_time:
                s, ds  = min_jerk(t_elapsed, self.move_time)
                q_des  = self.q_start + s  * (self.q_goal - self.q_start)
                dq_des = ds * (self.q_goal - self.q_start)
                return q_des, dq_des
            else:
                self.state = JointMoveState.HOLD_GOAL
                print("Reached HOLD_GOAL")
                return self.q_goal.copy(), np.zeros(7)

        if self.state == JointMoveState.HOLD_GOAL:
            return self.q_goal.copy(), np.zeros(7)

        if self.state == JointMoveState.FAILED:
            return None, None

        # SOLVING_IK is synchronous — update() should never see it.
        return self.q_hold.copy(), np.zeros(7)

    # ------------------------------------------------------------------
    # IK (synchronous, called from start())
    # ------------------------------------------------------------------

    def _solve_ik(self, mj_model, mj_data) -> None:
        """Solve IK using a scratch MjData; never writes to the live mj_data."""
        site_name  = self._cfg_ik["site_name"]
        target_pos = np.array(self._cfg_ik["target_pos"])
        keep_ori   = bool(self._cfg_ik.get("keep_current_orientation", True))

        # Scratch copy so IK forward-kinematics calls do not touch the sim state.
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

        target_R = (
            ik_data.site_xmat[site_id].copy().reshape(3, 3)
            if keep_ori else np.eye(3)
        )
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

    def _wait_for_enter(self) -> None:
        input("\nIK solved. Press Enter to start moving...")
        self._enter_pressed.set()
