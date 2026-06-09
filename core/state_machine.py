"""State machines for force-control.

Contains two state machines:

    JointMoveStateMachine   — joint-space IK + min-jerk move, FT tare, then a
                              Cartesian downward approach and z-force hold
                              (Milestones 1–2)
    StateMachine            — future: Cartesian contact/force/slide phases

update() returns a ControlCommand each tick: mode="joint" for the IK-move
phases, mode="cartesian" (world-frame tip targets) for the approach / force
hold, mode="failed" on error.

No MuJoCo viewer, pylibfranka, or ROS imports.
MuJoCo model/data are used only inside JointMoveStateMachine for IK FK calls;
the backend supplies the live tip pose for the Cartesian phases.
"""

from __future__ import annotations
from enum import Enum, auto
import threading
import numpy as np
import mujoco
from .types import RobotStateLite, WrenchSample, TargetSample, ControlCommand
from .ik import min_jerk, solve_ik_dls, euler_xyz_to_R


# ---------------------------------------------------------------------------
# Milestone 1: joint-space move state machine
# ---------------------------------------------------------------------------

class JointMoveState(Enum):
    HOLD             = auto()   # holds whatever q_hold is currently set to
    SOLVING_IK       = auto()   # synchronous; never seen by update()
    WAIT_USER_CONFIRM = auto()
    MOVE_JOINT       = auto()
    TARE             = auto()   # at the IK goal: average FT for tare_duration, then tare
    WAIT_APPROACH_CONFIRM = auto()  # at the goal: hold, wait for Enter to start the approach
    APPROACH         = auto()   # Cartesian: descend along world -Z until contact
    FORCE_HOLD       = auto()   # Cartesian: regulate Fz by admittance on z_ref
    WAIT_SLIDE_CONFIRM = auto() # Cartesian: keep regulating Fz, wait for Enter to start the slide
    SLIDE            = auto()   # Cartesian: Fz force-hold + smooth xy slide trajectory
    FAILED           = auto()


class JointMoveStateMachine:
    """Drives the joint-move sequence for Milestone 1.

    Typical usage::

        sm = JointMoveStateMachine(config)
        sm.start(model, data)          # synchronous IK + launches Enter-waiter thread
        # then in the control loop:
        cmd = sm.update(t, q, dq_current=dq, wrench=processed_wrench,
                        ft_sample=raw_ft, x_current=x_tip, R_current=R_tip)
        if cmd.mode == "failed":
            break                      # FAILED state — stop the loop
        # dispatch cmd.mode in ("joint", "cartesian") to the right controller
    """

    def __init__(self, config: dict, ft_processor=None):
        """
        Args:
            config: full config dict; reads config["ik"] and config["ft"].
            ft_processor: optional FTProcessor. When provided and ft.tare_on_hold
                is true, the machine spends ft.tare_duration seconds holding the
                IK goal pose, averages the FT reading there, then tares the
                processor — zeroing the payload's gravity wrench at that
                orientation so any subsequent wrench is pure contact force.
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

        # ---- Payload FT auto-tare at the IK goal (entered after MOVE_JOINT) --
        ft_cfg = config.get("ft", {})
        self._ft_processor  = ft_processor
        self._tare_enabled  = (
            bool(ft_cfg.get("tare_on_hold", True)) and ft_processor is not None
        )
        self._tare_duration = float(ft_cfg.get("tare_duration", 2.0))
        self._tare_t0       = 0.0
        self._tare_sum      = np.zeros(6)   # running sum of raw wrench samples
        self._tare_count    = 0
        self._tare_last_seq = None          # dedup: only accumulate fresh samples

        # ---- Cartesian approach + force-hold (Milestone 2) ------------------
        sm_cfg = config.get("state_machine", {})
        # When false, the machine stops at HOLD after the tare (Milestone-1 behavior).
        self._approach_enabled = bool(sm_cfg.get("approach_enabled", True))

        # APPROACH: open-loop descent along world -Z until contact.
        self._approach_speed   = float(sm_cfg.get("approach_speed", 0.002))     # [m/s]
        self._contact_fz       = float(sm_cfg.get("contact_fz_threshold", 2.0)) # [N]
        self._contact_confirm  = int(sm_cfg.get("contact_confirm_samples", 5))

        # FORCE_HOLD: admittance outer loop, Fz -> z_ref velocity.
        self._F_des            = float(sm_cfg.get("F_normal_desired", 2.0))     # [N]
        self._force_kp         = float(sm_cfg.get("force_kp", 5e-4))            # [m/s per N]
        self._force_kd         = float(sm_cfg.get("force_kd", 0.0))            # [m per N]
        self._max_hold_speed   = float(sm_cfg.get("max_force_hold_speed", 0.01))# [m/s]
        self._max_depth        = float(sm_cfg.get("max_depth", 0.05))          # [m]
        self._retract_allow    = float(sm_cfg.get("retract_allowance", 0.01))  # [m]

        # ---- Grouped force-z config (Milestone 3) ---------------------------
        # FORCE_HOLD and SLIDE each get their own z-force parameter set, consumed
        # by the shared _update_force_z() helper.  Each falls back to the legacy
        # state_machine.* scalars above so existing configs behave identically.
        fh = config.get("force_hold", {})
        self._force_hold_cfg = {
            "desired_fz":        float(fh.get("desired_fz",        self._F_des)),
            "kp_f":              float(fh.get("kp_f",              self._force_kp)),
            "kd_f":              float(fh.get("kd_f",              self._force_kd)),
            "max_z_speed":       float(fh.get("max_z_speed",       self._max_hold_speed)),
            "max_depth":         float(fh.get("max_depth",         self._max_depth)),
            "retract_allowance": float(fh.get("retract_allowance", self._retract_allow)),
        }

        slide = config.get("slide", {})
        # When false, the machine stays in FORCE_HOLD forever (no slide offered).
        self._slide_enabled = bool(slide.get("enabled", sm_cfg.get("slide_enabled", True)))

        sf = slide.get("force", {})           # SLIDE uses different z-force gains
        self._slide_force_cfg = {
            "desired_fz":        float(sf.get("desired_fz",        self._force_hold_cfg["desired_fz"])),
            "kp_f":              float(sf.get("kp_f",              self._force_hold_cfg["kp_f"])),
            "kd_f":              float(sf.get("kd_f",              self._force_hold_cfg["kd_f"])),
            "max_z_speed":       float(sf.get("max_z_speed",       self._force_hold_cfg["max_z_speed"])),
            "max_depth":         float(sf.get("max_depth",         self._force_hold_cfg["max_depth"])),
            "retract_allowance": float(sf.get("retract_allowance", self._force_hold_cfg["retract_allowance"])),
        }

        self._slide_motion_cfg = slide.get("motion", {})
        self._slide_speed     = float(self._slide_motion_cfg.get("speed", 0.001))     # [m/s]
        self._slide_ramp_time = float(self._slide_motion_cfg.get("ramp_time", 1.0))   # [s]
        self._slide_distance  = float(self._slide_motion_cfg.get("distance", 0.015))  # [m] safety cap

        tang = slide.get("tangential", {})
        self._slide_ki_vel        = float(tang.get("ki_vel_s", 20.0))       # integral gain on v-error
        self._slide_vel_int_limit = float(tang.get("vel_int_limit_s", 0.003))
        self._slide_dx_int_gain   = float(tang.get("dx_int_gain_s", 1.0))

        # Slide runtime state (set in _begin_slide; tangential integral is
        # SLIDE-only and reset on entry — kept separate from force-hold vars).
        self._p_slide_start:  np.ndarray | None = None
        self._R_slide_anchor: np.ndarray | None = None
        self._slide_t0    = 0.0
        self._slide_s_ref = 0.0
        self._slide_v_int = 0.0
        self._u_slide = np.array([1.0, 0.0, 0.0])
        self._u_cross = np.array([0.0, 1.0, 0.0])
        self._slide_waiter_launched = False

        # Cartesian reference state (captured at approach start).
        self.x_anchor:  float = 0.0
        self.y_anchor:  float = 0.0
        self.z_ref:     float = 0.0
        self.z_contact: float = 0.0
        self.R_anchor:  np.ndarray | None = None
        self._contact_count = 0
        self._force_e_prev  = 0.0
        self._z_dot         = 0.0

        # dt tracking for the force loop / z_ref integration.
        self._t_prev: float | None = None

        # Second Enter-waiter (before the approach).
        self._approach_pressed  = threading.Event()
        self._approach_thread: threading.Thread | None = None

        # Third Enter-waiter (before the slide).
        self._slide_pressed = threading.Event()
        self._slide_thread: threading.Thread | None = None

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
        dq_current: np.ndarray | None = None,
        wrench: np.ndarray | None = None,
        ft_sample: WrenchSample | None = None,
        x_current: np.ndarray | None = None,
        R_current: np.ndarray | None = None,
        v_current: np.ndarray | None = None,
        **_,
    ) -> ControlCommand:
        """Return the ControlCommand for the current control tick.

        Joint phases (HOLD, WAIT_USER_CONFIRM, MOVE_JOINT, TARE,
        WAIT_APPROACH_CONFIRM) return mode="joint" with q_des/dq_des.  The
        Cartesian phases (APPROACH, FORCE_HOLD) return mode="cartesian" with
        world-frame tip targets.  FAILED returns mode="failed".

        Inputs:
            wrench:     processed world-frame wrench (tared); Fz = wrench[2],
                        with the convention +Fz = compression/contact force.
            ft_sample:  raw WrenchSample, consumed only in TARE for averaging.
            x_current:  current tip world position (3,) — used to anchor the
                        Cartesian phases at approach start.
            R_current:  current tip world orientation (3,3) — held through the
                        approach / force hold.
        """
        dt = (t - self._t_prev) if self._t_prev is not None else 1e-3
        if dt <= 0.0:
            dt = 1e-3
        self._t_prev = t

        if self.state == JointMoveState.HOLD:
            return self._joint_cmd(self.q_hold.copy(), np.zeros(7))

        if self.state == JointMoveState.WAIT_USER_CONFIRM:
            if self._enter_pressed.is_set():
                # Capture actual robot q at the moment the user triggers motion.
                self.q_start         = q_current.copy()
                self.move_start_time = t
                self.state           = JointMoveState.MOVE_JOINT
                print("Starting movement...")
            # Hold for this tick whether or not we just transitioned.
            return self._joint_cmd(self.q_hold.copy(), np.zeros(7))

        if self.state == JointMoveState.MOVE_JOINT:
            t_elapsed = t - self.move_start_time
            if t_elapsed <= self.move_time:
                s, ds  = min_jerk(t_elapsed, self.move_time)
                q_des  = self.q_start + s  * (self.q_goal - self.q_start)
                dq_des = ds * (self.q_goal - self.q_start)
                return self._joint_cmd(q_des, dq_des)
            self.q_hold = self.q_goal.copy()
            if self._tare_enabled:
                self._begin_tare(t)
            else:
                self._after_goal_reached()
            return self._joint_cmd(self.q_hold.copy(), np.zeros(7))

        if self.state == JointMoveState.TARE:
            # Hold the goal pose and average fresh, valid FT samples. When the
            # window elapses, tare the processor so the payload's gravity wrench
            # at this orientation becomes the new zero.
            if ft_sample is not None and getattr(ft_sample, "valid", False):
                if ft_sample.seq != self._tare_last_seq:
                    self._tare_sum += np.asarray(ft_sample.wrench, dtype=float)
                    self._tare_count += 1
                    self._tare_last_seq = ft_sample.seq
            if (t - self._tare_t0) >= self._tare_duration:
                self._finish_tare()
            return self._joint_cmd(self.q_hold.copy(), np.zeros(7))

        if self.state == JointMoveState.WAIT_APPROACH_CONFIRM:
            if self._approach_pressed.is_set():
                self._begin_approach(x_current, R_current)
                if self.state == JointMoveState.FAILED:
                    return self._failed_cmd()
                # First Cartesian tick: hold the just-captured anchor pose.
                return self._cart_hold_cmd()
            # Keep holding the goal joint pose while waiting for Enter.
            return self._joint_cmd(self.q_hold.copy(), np.zeros(7))

        if self.state == JointMoveState.APPROACH:
            return self._update_approach(dt, wrench)

        if self.state == JointMoveState.FORCE_HOLD:
            return self._update_force_hold(dt, wrench)

        if self.state == JointMoveState.WAIT_SLIDE_CONFIRM:
            return self._update_wait_slide(t, dt, wrench, x_current, R_current)

        if self.state == JointMoveState.SLIDE:
            return self._update_slide(t, dt, wrench, v_current)

        if self.state == JointMoveState.FAILED:
            return self._failed_cmd()

        # SOLVING_IK is synchronous — update() should never see this.
        return self._joint_cmd(self.q_hold.copy(), np.zeros(7))

    # ------------------------------------------------------------------
    # Command builders
    # ------------------------------------------------------------------

    @staticmethod
    def _joint_cmd(q_des: np.ndarray, dq_des: np.ndarray) -> ControlCommand:
        return ControlCommand(mode="joint", q_des=q_des, dq_des=dq_des)

    def _cart_cmd(self, x_des: np.ndarray, dx_des: np.ndarray) -> ControlCommand:
        """Cartesian command holding the anchor orientation (w_des = 0)."""
        return ControlCommand(
            mode="cartesian",
            x_des=x_des,
            dx_des=dx_des,
            R_des=self.R_anchor.copy(),
            w_des=np.zeros(3),
        )

    def _cart_hold_cmd(self) -> ControlCommand:
        return self._cart_cmd(self._x_des(), np.zeros(3))

    @staticmethod
    def _failed_cmd() -> ControlCommand:
        return ControlCommand(mode="failed")

    def _x_des(self) -> np.ndarray:
        """Current Cartesian position reference: anchored xy, live z_ref."""
        return np.array([self.x_anchor, self.y_anchor, self.z_ref])

    # ------------------------------------------------------------------
    # Cartesian approach + force hold (Milestone 2)
    # ------------------------------------------------------------------

    def _after_goal_reached(self) -> None:
        """Branch taken once the goal pose is reached (post-move / post-tare)."""
        if self._approach_enabled:
            self._begin_wait_approach()
        else:
            self.state = JointMoveState.HOLD
            print("Reached HOLD")

    def _begin_wait_approach(self) -> None:
        """Enter WAIT_APPROACH_CONFIRM and launch the approach Enter-waiter."""
        self.state = JointMoveState.WAIT_APPROACH_CONFIRM
        self._approach_pressed.clear()
        self._approach_thread = threading.Thread(
            target=self._wait_for_enter_approach, daemon=True
        )
        self._approach_thread.start()
        print("At goal pose (FT tared). Press Enter to begin the downward approach...")

    def _wait_for_enter_approach(self) -> None:
        input("\nReady. Press Enter to descend along world -Z...")
        self._approach_pressed.set()

    def _begin_approach(self, x_current, R_current) -> None:
        """Capture the Cartesian anchor pose and start the descent."""
        if x_current is None or R_current is None:
            print("ERROR: no tip pose supplied to start the approach. State → FAILED.")
            self.state = JointMoveState.FAILED
            return
        self.x_anchor = float(x_current[0])
        self.y_anchor = float(x_current[1])
        self.z_ref    = float(x_current[2])
        self.R_anchor = np.asarray(R_current, dtype=float).copy()
        self._contact_count = 0
        self._z_dot = 0.0
        self.state = JointMoveState.APPROACH
        print(f"Approach anchor: x={self.x_anchor:+.4f} y={self.y_anchor:+.4f} "
              f"z={self.z_ref:+.4f}; descending at {self._approach_speed * 1e3:.1f} mm/s "
              f"(contact when Fz ≥ {self._contact_fz:.1f} N x{self._contact_confirm}).")

    def _update_approach(self, dt: float, wrench) -> ControlCommand:
        """Descend along world -Z at constant speed; detect contact on Fz."""
        Fz = float(wrench[2]) if wrench is not None else 0.0

        # Contact detection (debounced) at the current depth, before stepping.
        if Fz >= self._contact_fz:
            self._contact_count += 1
        else:
            self._contact_count = 0

        if self._contact_count >= self._contact_confirm:
            self.z_contact = self.z_ref
            self._force_e_prev = Fz - self._F_des
            self._z_dot = 0.0
            self.state = JointMoveState.FORCE_HOLD
            print(f"Contact (Fz={Fz:+.2f} N, z_contact={self.z_contact:+.4f} m). "
                  f"→ force hold, F_des={self._F_des:.1f} N.")
            # Hold this depth for the transition tick; force loop starts next.
            return self._cart_cmd(self._x_des(), np.zeros(3))

        # No contact yet: open-loop step down.
        self.z_ref -= self._approach_speed * dt
        dx_des = np.array([0.0, 0.0, -self._approach_speed])
        return self._cart_cmd(self._x_des(), dx_des)

    def _update_force_z(self, dt: float, wrench, cfg: dict) -> tuple[float, float]:
        """Shared admittance z-force regulator. Returns (z_ref, z_dot).

        Drives z_ref so Fz tracks cfg['desired_fz'].  Sign convention unchanged:
        +Fz = compression; Fz < desired → e < 0 → z_dot < 0 (descend to push
        harder); Fz > desired → z_dot > 0 (retract).  Used by both FORCE_HOLD
        (force_hold_cfg) and SLIDE (slide_force_cfg).  _force_e_prev is shared so
        the z loop is continuous across the force-hold → slide hand-off.
        """
        Fz = float(wrench[2]) if wrench is not None else 0.0
        e = Fz - cfg["desired_fz"]

        z_dot = cfg["kp_f"] * e + cfg["kd_f"] * (e - self._force_e_prev) / dt
        z_dot = float(np.clip(z_dot, -cfg["max_z_speed"], cfg["max_z_speed"]))

        self.z_ref += z_dot * dt
        # Safety clamp relative to the depth at first contact.
        z_lo = self.z_contact - cfg["max_depth"]
        z_hi = self.z_contact + cfg["retract_allowance"]
        self.z_ref = float(np.clip(self.z_ref, z_lo, z_hi))

        self._force_e_prev = e
        self._z_dot = z_dot
        return self.z_ref, self._z_dot

    def _update_force_hold(self, dt: float, wrench) -> ControlCommand:
        """Regulate Fz at the contact point; once active, offer the slide."""
        _, z_dot = self._update_force_z(dt, wrench, self._force_hold_cfg)

        # Once force hold is active, launch the slide Enter-waiter and move to
        # WAIT_SLIDE_CONFIRM (which keeps regulating z with the same cfg).
        if self._slide_enabled and not self._slide_waiter_launched:
            self._begin_wait_slide()

        dx_des = np.array([0.0, 0.0, z_dot])
        return self._cart_cmd(self._x_des(), dx_des)

    # ------------------------------------------------------------------
    # User-triggered slide (Milestone 3)
    # ------------------------------------------------------------------

    def _begin_wait_slide(self) -> None:
        """Launch the slide Enter-waiter and enter WAIT_SLIDE_CONFIRM."""
        self._slide_waiter_launched = True
        self._slide_pressed.clear()
        self._slide_thread = threading.Thread(
            target=self._wait_for_enter_slide, daemon=True
        )
        self._slide_thread.start()
        self.state = JointMoveState.WAIT_SLIDE_CONFIRM
        print("Force hold active. Press Enter to begin the slide "
              "(z force-hold continues while waiting)...")

    def _wait_for_enter_slide(self) -> None:
        input("\nForce hold active. Press Enter to start sliding...")
        self._slide_pressed.set()

    def _update_wait_slide(self, t, dt, wrench, x_current, R_current) -> ControlCommand:
        """Keep regulating Fz at the contact point while awaiting the slide Enter."""
        _, z_dot = self._update_force_z(dt, wrench, self._force_hold_cfg)
        if self._slide_pressed.is_set():
            self._begin_slide(t, x_current, R_current)
            if self.state == JointMoveState.FAILED:
                return self._failed_cmd()
        # Hold xy at the contact anchor; z keeps force-tracking.
        return self._cart_cmd(self._x_des(), np.array([0.0, 0.0, z_dot]))

    def _begin_slide(self, t, x_current, R_current) -> None:
        """Record the slide start pose / direction; reset the tangential integral."""
        if x_current is None or R_current is None:
            print("ERROR: no tip pose supplied to start the slide. State → FAILED.")
            self.state = JointMoveState.FAILED
            return
        self._p_slide_start  = np.asarray(x_current, dtype=float).copy()
        self._R_slide_anchor = np.asarray(R_current, dtype=float).copy()
        self._slide_t0    = t
        self._slide_s_ref = 0.0
        self._slide_v_int = 0.0          # SLIDE-only integral, reset on entry

        direction_xy = np.asarray(
            self._slide_motion_cfg.get("direction_xy", [1.0, 0.0]), dtype=float
        )
        u = np.array([direction_xy[0], direction_xy[1], 0.0])
        self._u_slide = u / np.linalg.norm(u)
        self._u_cross = np.array([-self._u_slide[1], self._u_slide[0], 0.0])

        self.state = JointMoveState.SLIDE
        print(f"Slide start p={self._p_slide_start.round(4)} "
              f"dir={self._u_slide[:2].round(3)} "
              f"speed={self._slide_speed * 1e3:.2f} mm/s "
              f"distance={self._slide_distance * 1e3:.1f} mm.")

    def _update_slide(self, t, dt, wrench, v_current) -> ControlCommand:
        """z force-hold (slide gains) + smooth constant-speed xy slide."""
        # ---- z: force tracking with SLIDE force params ----------------
        # Updates self.z_ref and self._z_dot in place (used below for x/dx_des).
        self._update_force_z(dt, wrench, self._slide_force_cfg)

        # ---- xy: min-jerk ramp to constant slide speed ----------------
        s = float(np.clip((t - self._slide_t0) / self._slide_ramp_time, 0.0, 1.0))
        r = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
        v_slide_ref = self._slide_speed * r

        # Stop advancing the reference once the configured distance is covered.
        if self._slide_s_ref >= self._slide_distance:
            v_slide_ref = 0.0
        self._slide_s_ref = min(
            self._slide_s_ref + v_slide_ref * dt, self._slide_distance
        )
        p_xy_des = self._p_slide_start[:2] + self._u_slide[:2] * self._slide_s_ref

        # ---- tangential velocity-error integral (friction lag comp) ---
        if v_current is not None:
            v_slide_actual = float(np.dot(np.asarray(v_current, dtype=float), self._u_slide))
        else:
            v_slide_actual = 0.0
        e_vs = v_slide_ref - v_slide_actual
        self._slide_v_int += self._slide_ki_vel * e_vs * dt
        self._slide_v_int = float(np.clip(
            self._slide_v_int, -self._slide_vel_int_limit, self._slide_vel_int_limit
        ))
        v_slide_cmd = v_slide_ref + self._slide_dx_int_gain * self._slide_v_int

        x_des = np.array([p_xy_des[0], p_xy_des[1], self.z_ref])
        dx_des = np.array([
            self._u_slide[0] * v_slide_cmd,
            self._u_slide[1] * v_slide_cmd,
            self._z_dot,
        ])
        return ControlCommand(
            mode="cartesian",
            x_des=x_des,
            dx_des=dx_des,
            R_des=self._R_slide_anchor.copy(),
            w_des=np.zeros(3),
        )

    # ------------------------------------------------------------------
    # Payload FT auto-tare (entered once after MOVE_JOINT completes)
    # ------------------------------------------------------------------

    def _begin_tare(self, t: float) -> None:
        """Enter TARE: reset the accumulator and start the averaging window."""
        self._tare_t0       = t
        self._tare_sum[:]   = 0.0
        self._tare_count    = 0
        self._tare_last_seq = None
        self.state = JointMoveState.TARE
        print(f"Reached goal — averaging FT for {self._tare_duration:.1f}s "
              "to tare the payload baseline (keep the tool free of contact)...")

    def _finish_tare(self) -> None:
        """Average the recorded samples, tare the processor, then advance."""
        if self._tare_count > 0:
            mean_raw = self._tare_sum / self._tare_count
            self._ft_processor.tare(mean_raw)
            print(f"Payload baseline tared over {self._tare_count} FT samples:")
            print(f"  bias (raw wrench) = {np.round(mean_raw, 3).tolist()}")
            print("  Subsequent wrench now reads contact force only.")
        else:
            print("⚠️  No valid FT samples during the tare window — "
                  "baseline NOT tared (check the FT source).")
        self._after_goal_reached()

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

        target_R = self._resolve_target_R(mj_model, ik_data, site_id)
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

    def _resolve_target_R(self, mj_model, ik_data, site_id: int) -> np.ndarray:
        """Return the 3×3 target orientation matrix from config.

        target_euler_xyz is applied as an additional intrinsic XYZ rotation on
        top of a reference orientation:

            target_R = R_ref @ R_delta

        By default R_ref is the controlled site's orientation at the model's
        home keyframe (ik.orientation_reference: "keyframe").  This makes the
        configured euler delta produce the SAME target orientation on hardware —
        where IK is seeded from the live robot pose — as in sim, which already
        starts at the keyframe.  Set ik.orientation_reference: "current" to apply
        the delta on top of the live start orientation instead (legacy behavior).

        If target_euler_xyz is [0, 0, 0] (or omitted), target_R == R_ref.
        """
        R_ref = self._reference_orientation(mj_model, ik_data, site_id)
        euler = self._cfg_ik.get("target_euler_xyz", [0.0, 0.0, 0.0])
        R_delta = euler_xyz_to_R(euler)
        return R_ref @ R_delta

    def _reference_orientation(self, mj_model, ik_data, site_id: int) -> np.ndarray:
        """Site orientation that target_euler_xyz is applied relative to.

        "keyframe" (default): evaluate the site orientation at the model's home
        keyframe via a scratch MjData, so the reference is independent of the
        live IK seed (the robot's actual start pose on hardware).
        "current": use the live site orientation (orientation as seeded).
        Falls back to the live orientation if the model defines no keyframe.
        """
        mode = str(self._cfg_ik.get("orientation_reference", "keyframe")).lower()

        if mode == "keyframe" and mj_model.nkey > 0:
            key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
            if key_id < 0:
                key_id = 0
            scratch = mujoco.MjData(mj_model)
            mujoco.mj_resetDataKeyframe(mj_model, scratch, key_id)
            mujoco.mj_forward(mj_model, scratch)
            key_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_KEY, key_id)
            print(f"Orientation reference: keyframe '{key_name}'")
            return scratch.site_xmat[site_id].copy().reshape(3, 3)

        if mode == "keyframe":
            print("Orientation reference: current pose (model defines no keyframe)")
        else:
            print("Orientation reference: current pose")
        return ik_data.site_xmat[site_id].copy().reshape(3, 3)

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
