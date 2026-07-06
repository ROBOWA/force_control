"""Franka real-robot backend — FLANGE FT variant (Ubuntu Panda PC only).

Identical control pipeline to backends.franka_backend.FrankaBackend, but the
contact wrench comes from the Panda's OWN external-wrench estimate read straight
off the RobotState in the 1 kHz callback (sensors.ft_flange.FTFlangeSource),
instead of an external FT sensor bridged over ROS / shared memory.

Consequences vs the Bota/ROS backend:
    * No FT background thread, no ROS, no /dev/shm — one fewer moving part.
    * No FTWorldRotator: with frame="base" the source returns O_F_ext_hat_K,
      already in the robot base (world-aligned) frame, so wrench[2] is the
      vertical force in the same frame the state machine / tip kinematics use.
    * The estimate is noisy (torque-derived), so the FT processor is the
      Butterworth-based FlangeFTProcessor rather than the light EWA FTProcessor.

Everything downstream — state machine, joint PD, Cartesian impedance, payload
gravity compensation, safety, logging — is the SAME core code, reused unchanged.

Threading model:
    Thread A (Enter-waiter):  inside JointMoveStateMachine.start() — key press
    Thread B (pylibfranka 1 kHz): torque callback — reads flange F_ext off the
                                  RobotState, runs the state machine, computes
                                  joint PD / Cartesian impedance + payload
                                  gravity, returns Torques.
(There is no FT thread — the wrench arrives with every RobotState.)

The 1 kHz callback MUST return within ~1 ms. It must not block on any lock,
socket, or ROS primitive.
"""

from __future__ import annotations
import time
import numpy as np

from core.state_machine import JointMoveStateMachine, JointMoveState
from core.controller import JointPDController, CartesianImpedanceController
from core.ft_processor_flange import FlangeFTProcessor
from core.kinematics import SiteKinematics
from core.data_logger import TrajectoryLogger
from core.payload_gravity import PayloadGravityCompensator
from core.safety import saturate_torque_rate
from sensors.ft_flange import FTFlangeSource


class FrankaBackendFlange:
    """Connects to a real Panda and runs the 1 kHz loop on the flange F_ext.

    Structurally analogous to FrankaBackend; only the FT wiring differs.

    Usage::

        backend = FrankaBackendFlange(robot_ip, cfg)
        backend.connect()
        backend.start_ft_source()          # builds the (thread-less) flange source
        backend.initialize_core_pipeline()
        backend.run()                      # blocks inside robot.control_torques()
    """

    def __init__(self, robot_ip: str, config: dict):
        self._robot_ip = robot_ip
        self._cfg = config

        # Hardware objects — created in connect() / run()
        self._robot = None
        self._Torques = None       # pylibfranka Torques class, stored in run()

        # Core pipeline — created in start_ft_source() / initialize_core_pipeline()
        self._ft_source:        FTFlangeSource            | None = None
        self._ft_processor:     FlangeFTProcessor         | None = None
        self._state_machine:    JointMoveStateMachine     | None = None
        self._joint_controller: JointPDController         | None = None
        self._cart_controller:  CartesianImpedanceController | None = None
        self._tip_kin:          SiteKinematics            | None = None
        self._payload_comp:     PayloadGravityCompensator | None = None
        self._logger:           TrajectoryLogger          | None = None

        # Pre-read robot state (captured in connect())
        self._q_initial: np.ndarray = np.zeros(7)

        # 1 kHz callback state — preallocated, never reallocated in callback
        self._tau_prev        = np.zeros(7)
        self._t_control       = 0.0
        self._max_torque      = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
        self._max_torque_rate = 1.0

        # Lightweight timing stats (accumulated, printed 1 Hz)
        self._stat_sum_us  = 0.0
        self._stat_max_us  = 0.0
        self._stat_ticks   = 0
        self._last_print_t = 0.0

        # Debug FT heartbeat (1 Hz) — gated by config ft.debug_print
        self._ft_debug = bool(self._cfg.get("ft", {}).get("debug_print", False))
        self._last_ft_print_t = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to robot, set conservative collision behavior, read initial q.

        pylibfranka is imported here so non-Linux machines never touch it.
        """
        from pylibfranka import Robot  # noqa: PLC0415

        print(f"Connecting to Franka at {self._robot_ip} …")
        self._robot = Robot(self._robot_ip)
        print("Connected.")

        # Conservative joint-torque and Cartesian-force collision thresholds.
        self._robot.set_collision_behavior(
            [40.0, 40.0, 38.0, 38.0, 30.0, 25.0, 20.0],  # lower torque [Nm] (7,)
            [40.0, 40.0, 38.0, 38.0, 30.0, 25.0, 20.0],  # upper torque [Nm] (7,)
            [40.0, 40.0, 40.0, 50.0, 50.0, 50.0],          # lower force [N/Nm] (6,)
            [40.0, 40.0, 40.0, 50.0, 50.0, 50.0],          # upper force [N/Nm] (6,)
        )

        state = self._robot.read_once()
        self._q_initial = np.asarray(state.q, dtype=np.float64)
        print(f"Initial q: {self._q_initial.round(4)}")

    def start_ft_source(self) -> None:
        """Create the (thread-less) flange FT source.

        No background thread and nothing to connect: the wrench is pulled off
        each RobotState inside the callback via FTFlangeSource.update().  Kept as
        a separate lifecycle step to mirror FrankaBackend's call sequence.
        """
        ft_cfg = self._cfg.get("ft", {})
        frame = str(ft_cfg.get("frame", "base")).lower()
        self._ft_source = FTFlangeSource(frame=frame)
        self._ft_source.start()  # no-op; symmetry with the other backends
        print(f"Flange FT source ready (frame='{frame}', "
              f"field='{self._ft_source._FIELD[frame]}').")

    def initialize_core_pipeline(self) -> None:
        """Load IK model, create and start the core pipeline.

        Must be called after connect() (needs self._q_initial).
        Solves IK synchronously; launches the Enter-waiter background thread.
        Never called from inside the 1 kHz callback.
        """
        import mujoco  # noqa: PLC0415

        ft_cfg = self._cfg.get("ft", {})

        # ---- IK model -------------------------------------------------
        ik_cfg = self._cfg.get("ik", {})
        ik_xml = ik_cfg.get(
            "mjcf",
            self._cfg.get("payload_gravity", {}).get("model_full"),
        )
        if ik_xml is None:
            raise ValueError(
                "No IK model path: set ik.mjcf or payload_gravity.model_full in config"
            )

        ik_model = mujoco.MjModel.from_xml_path(ik_xml)
        ik_data  = mujoco.MjData(ik_model)
        ik_data.qpos[:7] = self._q_initial
        ik_data.qvel[:]  = 0.0
        mujoco.mj_forward(ik_model, ik_data)
        print(f"IK model loaded: {ik_xml}")

        # ---- FT processor (Butterworth low-pass on the noisy flange F_ext) ----
        # No world-frame rotator: the base-frame O_F_ext_hat_K wrench is already
        # world-aligned, so the tared baseline stays valid as the wrist moves
        # (gravity's vertical force component is orientation-independent).
        filt = ft_cfg.get("filter", {})
        self._ft_processor = FlangeFTProcessor(
            sign=ft_cfg.get("sign", 1.0),   # scalar or per-axis [Fx,Fy,Fz,Tx,Ty,Tz]
            filter_type=str(filt.get("type", "butterworth")),
            cutoff_hz=float(filt.get("cutoff_hz", 20.0)),
            sample_rate_hz=float(filt.get("sample_rate_hz", 1000.0)),
            ewa_alpha=float(filt.get("alpha", 0.1)),
            despike_window=int(filt.get("despike_window", 1)),
        )
        print(f"Flange FT processor: filter={filt.get('type', 'butterworth')}, "
              f"cutoff={filt.get('cutoff_hz', 20.0)} Hz, "
              f"despike_window={filt.get('despike_window', 1)}, "
              f"sign={ft_cfg.get('sign', 1.0)}.")

        # ---- State machine (IK + Enter-waiter thread) -----------------
        # Pass the FT processor so the machine auto-tares the payload baseline
        # once it reaches the IK goal (TARE state).
        self._state_machine = JointMoveStateMachine(
            self._cfg, ft_processor=self._ft_processor
        )
        self._state_machine.start(ik_model, ik_data)   # synchronous IK solve

        # ---- Joint PD controller --------------------------------------
        self._joint_controller = JointPDController(self._cfg["joint_pd"])

        # ---- Cartesian impedance controller + tip kinematics ----------
        self._cart_controller = CartesianImpedanceController(self._cfg["controller"])
        self._tip_kin = SiteKinematics(ik_model, ik_cfg["site_name"])
        print(f"Cartesian controller ready (tip site '{ik_cfg['site_name']}').")

        # ---- Payload gravity compensation -----------------------------
        pg_cfg = self._cfg.get("payload_gravity", {})
        if pg_cfg.get("enabled", False):
            self._payload_comp = PayloadGravityCompensator(self._cfg)
            self._benchmark_payload_comp()
        else:
            print("Payload gravity compensation disabled.")

        # ---- Safety limits from config --------------------------------
        safety_cfg = self._cfg.get("safety", {})
        self._max_torque = np.array(
            safety_cfg.get("max_torque", [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]),
            dtype=np.float64,
        )
        self._max_torque_rate = float(safety_cfg.get("max_torque_rate", 1.0))

        # ---- Trajectory logging (tip x, v, Fz, phase per tick) --------
        log_cfg = self._cfg.get("logging", {})
        if log_cfg.get("enabled", False):
            self._logger = TrajectoryLogger(
                output_dir=log_cfg.get("output_dir", "data/logs"),
                prefix=log_cfg.get("prefix", "real_flange"),
                capacity=int(log_cfg.get("capacity", 1_500_000)),
                phase_names={s.value: s.name for s in JointMoveState},
            )
            print(f"Trajectory logging enabled -> {log_cfg.get('output_dir', 'data/logs')}")

        print("Core pipeline ready. Entering torque control…")

    def run(self) -> None:
        """Enter robot.control_torques() — blocks until finished or Ctrl+C."""
        from pylibfranka import Torques  # noqa: PLC0415

        # Store Torques class so the callback can use it without importing.
        self._Torques = Torques

        self._tau_prev[:]  = 0.0
        self._t_control    = 0.0
        self._last_print_t = time.perf_counter()
        self._last_ft_print_t = time.perf_counter()

        print("Torque control active — press Ctrl+C to stop.")
        try:
            self._robot.control_torques(self._torque_callback)
        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            # Flush the trajectory log once the RT loop has stopped.
            if self._logger is not None:
                self._logger.save()

    def stop(self) -> None:
        """No FT thread to stop; kept for interface parity with FrankaBackend."""
        if self._ft_source is not None:
            self._ft_source.stop()

    # ------------------------------------------------------------------
    # 1 kHz torque callback
    # ------------------------------------------------------------------

    def _torque_callback(self, robot_state, duration):
        """Called by pylibfranka at ~1000 Hz.

        MUST return a Torques object within ~1 ms.
        Must NOT block, print every tick, solve IK, call input(), or allocate
        large objects.
        """
        t0 = time.perf_counter()

        q  = np.asarray(robot_state.q,  dtype=np.float64)
        dq = np.asarray(robot_state.dq, dtype=np.float64)

        dt = duration.to_sec() if hasattr(duration, "to_sec") else 1e-3
        self._t_control += dt

        # ---- FT: pull the flange wrench straight off the RobotState -----
        # Base-frame (O_F_ext_hat_K) => already world-aligned, so no rotation.
        raw_ft = self._ft_source.update(robot_state, t=self._t_control)
        processed_wrench = self._ft_processor.process(raw_ft.wrench)

        # ---- Debug: FT heartbeat every 1 s, gated by config -----------
        # Prints the PROCESSED wrench (sign + bias + low-pass): before the TARE
        # it tracks the filtered raw reading, and once the payload baseline is
        # tared at the IK goal it shows the contact wrench only.
        if self._ft_debug:
            now_dbg = time.perf_counter()
            if now_dbg - self._last_ft_print_t >= 1.0:
                self._last_ft_print_t = now_dbg
                w = processed_wrench
                print(
                    f"FT(flange)  "
                    f"F=[{w[0]:+7.2f} {w[1]:+7.2f} {w[2]:+7.2f}]  "
                    f"T=[{w[3]:+7.2f} {w[4]:+7.2f} {w[5]:+7.2f}]"
                )

        # ---- Tip kinematics (world pose / Jacobian / velocity) -------
        x_tip, R_tip, J_tip, v_tip, w_tip = self._tip_kin.compute(q, dq)

        # ---- State machine -------------------------------------------
        cmd = self._state_machine.update(
            t=self._t_control,
            q_current=q,
            dq_current=dq,
            wrench=processed_wrench,
            ft_sample=raw_ft,
            x_current=x_tip,
            R_current=R_tip,
            v_current=v_tip,
        )

        # ---- Trajectory log: tip x, v, Fz (world), phase --------------
        if self._logger is not None:
            self._logger.log(
                self._t_control, x_tip, v_tip,
                processed_wrench[2], self._state_machine.state.value,
            )

        # ---- Task torque: joint PD or Cartesian impedance ------------
        if cmd.mode == "cartesian":
            tau_task = self._cart_controller.compute(
                x_tip, R_tip, v_tip, w_tip, J_tip,
                cmd.x_des, cmd.dx_des, cmd.R_des, cmd.w_des,
            )
        else:
            # "joint" tracks the command; "failed" holds the current pose.
            if cmd.mode == "failed":
                q_des, dq_des = q.copy(), np.zeros(7)
            else:
                q_des, dq_des = cmd.q_des, cmd.dq_des
            tau_task = self._joint_controller.compute(q, dq, q_des, dq_des)

        # ---- Payload gravity (delta only) ----------------------------
        # Franka firmware already compensates arm gravity internally.
        # We only add the payload delta: G_full(q) - G_zero(q).
        if self._payload_comp is not None:
            tau_payload = self._payload_comp.compute(q)
        else:
            tau_payload = np.zeros(7)

        tau_cmd = tau_task + tau_payload

        # ---- Safety: clip then rate-limit ----------------------------
        tau_cmd = np.clip(tau_cmd, -self._max_torque, self._max_torque)
        tau_cmd = saturate_torque_rate(tau_cmd, self._tau_prev, self._max_torque_rate)
        self._tau_prev = tau_cmd.copy()

        # ---- Timing stats (accumulated; printed once per second) -----
        self._update_stats((time.perf_counter() - t0) * 1e6)

        return self._Torques(tau_cmd.tolist())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _benchmark_payload_comp(self) -> None:
        """Print payload torque at home pose and benchmark compute time."""
        q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

        tau_check = self._payload_comp.compute(q_home)
        print("Payload gravity torque at q_home [Nm]:")
        for i, t in enumerate(tau_check):
            print(f"  joint{i + 1}: {t:+.3f}")

        n = 1000
        t0 = time.perf_counter()
        for _ in range(n):
            self._payload_comp.compute(q_home)
        bench_us = (time.perf_counter() - t0) * 1e6 / n
        print(f"Payload comp compute: {bench_us:.1f} us / tick")

    def _update_stats(self, dt_us: float) -> None:
        """Accumulate timing; print once per second — never every tick."""
        self._stat_sum_us += dt_us
        if dt_us > self._stat_max_us:
            self._stat_max_us = dt_us
        self._stat_ticks += 1

        now = time.perf_counter()
        elapsed = now - self._last_print_t
        if elapsed >= 1.0:
            avg  = self._stat_sum_us / self._stat_ticks
            rate = self._stat_ticks / elapsed
            print(
                f"compute: avg {avg:.0f} us, "
                f"max {self._stat_max_us:.0f} us, "
                f"ticks/s {rate:.0f}"
            )
            self._stat_sum_us  = 0.0
            self._stat_max_us  = 0.0
            self._stat_ticks   = 0
            self._last_print_t = now
