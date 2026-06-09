"""Franka real-robot backend — Ubuntu Panda PC only.

Milestone 1: joint-space IK + min-jerk move, identical logic to MuJoCo sim.

All pylibfranka and rospy imports are local so that this file can be imported
on Mac without errors (compile/syntax check only).

Threading model (Milestone 1):
    Thread A (ft_ros_spin): FTRosSource background thread — receives FT at ~400 Hz
    Thread B (Enter-waiter):  inside JointMoveStateMachine.start() — waits for key press
    Thread C (pylibfranka 1 kHz): torque callback — reads FT cache, runs state machine,
                                  computes joint PD + payload gravity, returns Torques

The 1 kHz callback MUST return within ~1 ms. It must not block on any lock,
socket, or ROS primitive.
"""

from __future__ import annotations
import time
import numpy as np

from core.state_machine import JointMoveStateMachine
from core.controller import JointPDController
from core.ft_processor import FTProcessor
from core.payload_gravity import PayloadGravityCompensator
from core.safety import saturate_torque_rate
from sensors.ft_ros import FTRosSource
from sensors.ft_shm import FTShmSource


class FrankaBackend:
    """Connects to a real Panda and runs the 1 kHz torque-control loop.

    Structurally analogous to MuJoCoBackend.  Reuses all core modules; this
    class only provides the hardware wiring.

    Usage (new runner)::

        backend = FrankaBackend(robot_ip, cfg)
        backend.connect()
        backend.start_ft_source()
        backend.initialize_core_pipeline()
        backend.run()                  # blocks inside robot.control_torques()

    Usage (legacy run_controller.py)::

        backend = FrankaBackend(robot_ip, cfg)
        backend.connect()
        backend.start_ft_thread()      # alias for start_ft_source()
        backend.start_outer_loop()     # alias for initialize_core_pipeline()
        backend.run()
    """

    def __init__(self, robot_ip: str, config: dict):
        self._robot_ip = robot_ip
        self._cfg = config

        # Hardware objects — created in connect() / start_ft_source()
        self._robot = None
        self._Torques = None       # pylibfranka Torques class, stored in run()

        # Core pipeline — created in initialize_core_pipeline()
        self._ft_source:        FTRosSource | FTShmSource | None = None
        self._ft_processor:     FTProcessor               | None = None
        self._state_machine:    JointMoveStateMachine     | None = None
        self._joint_controller: JointPDController         | None = None
        self._payload_comp:     PayloadGravityCompensator | None = None

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

        # Debug FT heartbeat (10 Hz) — gated by config ft.debug_print
        self._ft_debug = bool(self._cfg.get("ft", {}).get("debug_print", False))
        self._last_ft_print_t = 0.0
        # Two independent counters: by writer seq (authoritative msg count, one per
        # bridged/received message) and by distinct timestamp (true unique-sample
        # rate). If "by msg" >> "by stamp", the sensor republishes each sample.
        self._ft_seq_count    = 0
        self._ft_prev_seq     = None
        self._ft_t_count      = 0
        self._ft_prev_t       = None
        self._ft_seq_at_print = 0
        self._ft_t_at_print   = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to robot, set conservative collision behavior, read initial q.

        pylibfranka is imported here so Mac never touches it.
        """
        from pylibfranka import Robot  # noqa: PLC0415

        print(f"Connecting to Franka at {self._robot_ip} …")
        self._robot = Robot(self._robot_ip)
        print("Connected.")

        # Conservative joint-torque and Cartesian-force collision thresholds.
        # Tune upward once the system is validated.
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
        """Create and start FTRosSource in its background thread.

        Must be called before robot.control_torques() so the FT stream is
        already running during HOLD / IK / WAIT_USER_CONFIRM / MOVE_JOINT.
        """
        ft_cfg = self._cfg.get("ft", {})
        source = str(ft_cfg.get("source", "ros")).lower()
        if source == "shm":
            # Separate-process bridge (scripts/ft_node.py) feeds /dev/shm at full
            # rate; we just read it here — no ROS, no GIL contention with the loop.
            self._ft_source = FTShmSource(
                shm_path=ft_cfg.get("shm_path", "/dev/shm/ft_wrench"),
            )
        else:
            self._ft_source = FTRosSource(
                topic=ft_cfg.get("topic", "/ft_sensor/wrench"),
                queue_size=int(ft_cfg.get("queue_size", 0)),
            )
        self._ft_source.start()
        print(f"FT source started ({source}).")

    def start_ft_thread(self) -> None:
        """Alias for start_ft_source() — backward compat with run_controller.py."""
        self.start_ft_source()

    def initialize_core_pipeline(self) -> None:
        """Load IK model, create and start the core pipeline.

        Must be called after connect() (needs self._q_initial).
        Solves IK synchronously; launches Enter-waiter background thread.
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

        # ---- FT processor ---------------------------------------------
        self._ft_processor = FTProcessor(
            sign=float(ft_cfg.get("sign", 1.0)),
            lowpass_alpha=float(ft_cfg.get("lowpass_alpha", 1.0)),
        )

        # ---- State machine (IK + Enter-waiter thread) -----------------
        self._state_machine = JointMoveStateMachine(self._cfg)
        self._state_machine.start(ik_model, ik_data)   # synchronous IK solve

        # ---- Joint PD controller --------------------------------------
        self._joint_controller = JointPDController(self._cfg["joint_pd"])

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

        print("Core pipeline ready. Entering torque control…")

    def start_outer_loop(self) -> None:
        """Alias for initialize_core_pipeline() — backward compat with run_controller.py."""
        self.initialize_core_pipeline()

    def run(self) -> None:
        """Enter robot.control_torques() — blocks until finished or Ctrl+C."""
        from pylibfranka import Torques  # noqa: PLC0415

        # Store Torques class so the callback can use it without importing.
        self._Torques = Torques

        self._tau_prev[:]  = 0.0
        self._t_control    = 0.0
        self._last_print_t = time.perf_counter()
        self._last_ft_print_t = time.perf_counter()
        self._ft_seq_count = 0
        self._ft_prev_seq = None
        self._ft_t_count = 0
        self._ft_prev_t = None
        self._ft_seq_at_print = 0
        self._ft_t_at_print = 0

        print("Torque control active — press Ctrl+C to stop.")
        try:
            self._robot.control_torques(self._torque_callback)
        except KeyboardInterrupt:
            print("\nInterrupted.")

    def stop(self) -> None:
        """Stop the FT background thread (safe to call from any thread)."""
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

        # ---- FT: read latest cached sample every tick ----------------
        raw_ft           = self._ft_source.get_latest()
        processed_wrench = self._ft_processor.process(raw_ft.wrench)

        # ---- Debug: FT heartbeat every 0.1 s (10 Hz), gated by config -
        # Reports TWO rates so we can tell "throttled" from "republished":
        #   by msg   = distinct writer seq  -> messages actually reaching the loop
        #   by stamp = distinct timestamps  -> unique sensor samples
        # by msg >> by stamp means the sensor republishes each sample N times.
        # Bring-up aid only — blocking print on the 1 kHz RT thread;
        # set ft.debug_print: false for timing-sensitive runs.
        if self._ft_debug:
            w = raw_ft.wrench
            if raw_ft.valid:
                if raw_ft.seq != self._ft_prev_seq:
                    self._ft_seq_count += 1
                    self._ft_prev_seq = raw_ft.seq
                if raw_ft.t != self._ft_prev_t:
                    self._ft_t_count += 1
                    self._ft_prev_t = raw_ft.t

            now_dbg = time.perf_counter()
            if now_dbg - self._last_ft_print_t >= 0.1:
                dt = now_dbg - self._last_ft_print_t
                hz_seq = (self._ft_seq_count - self._ft_seq_at_print) / dt if dt > 0 else 0.0
                hz_t = (self._ft_t_count - self._ft_t_at_print) / dt if dt > 0 else 0.0
                self._last_ft_print_t = now_dbg
                self._ft_seq_at_print = self._ft_seq_count
                self._ft_t_at_print = self._ft_t_count
                print(
                    f"FT {'ok' if raw_ft.valid else 'NO-DATA'} "
                    f"rawseq={raw_ft.seq} "          # current b[2] from the writer
                    f"recv={self._ft_seq_count} "
                    f"(~{hz_seq:.0f} Hz by msg, ~{hz_t:.0f} Hz by stamp)  "
                    f"F=[{w[0]:+7.2f} {w[1]:+7.2f} {w[2]:+7.2f}]  "
                    f"T=[{w[3]:+7.2f} {w[4]:+7.2f} {w[5]:+7.2f}]"
                )

        # ---- State machine -------------------------------------------
        q_des, dq_des = self._state_machine.update(
            t=self._t_control,
            q_current=q,
            dq_current=dq,
            wrench=processed_wrench,
            ft_sample=raw_ft,
        )

        # FAILED state: hold current pose safely
        if q_des is None:
            q_des  = q.copy()
            dq_des = np.zeros(7)

        # ---- Joint PD torque -----------------------------------------
        tau_joint_pd = self._joint_controller.compute(q, dq, q_des, dq_des)

        # ---- Payload gravity (delta only) ----------------------------
        # Franka firmware already compensates arm gravity internally.
        # We only add the payload delta: G_full(q) - G_zero(q).
        if self._payload_comp is not None:
            tau_payload = self._payload_comp.compute(q)
        else:
            tau_payload = np.zeros(7)

        tau_cmd = tau_joint_pd + tau_payload

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
