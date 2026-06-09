"""MuJoCo simulation backend — Mac development mode.

Milestone 1: thin backend that wires MuJoCo model/data to the state machine
and joint PD controller.  No IK, no trajectory math, no PD formula here.
"""

from __future__ import annotations
import numpy as np
import mujoco
import mujoco.viewer

from core.state_machine import JointMoveStateMachine, JointMoveState
from core.controller import JointPDController, CartesianImpedanceController
from core.ft_processor import FTProcessor
from core.kinematics import SiteKinematics
from core.data_logger import TrajectoryLogger
from core.payload_gravity import PayloadGravityCompensator
from sensors.ft_mujoco import FTMuJoCo


# panda.xml actuator definition (joints 1–7):
#   <general gainprm="1" biastype="none"/>
#   applied force = 1 × ctrl  (no bias subtracted)
#   → to command torque τ: ctrl = τ / ACTUATOR_GAIN
# forcerange=[-87, 87] Nm clips the output inside MuJoCo.
# This constant lives here because it is a property of the XML, not the controller.
ACTUATOR_GAIN = 1.0


class MuJoCoBackend:
    """Thin MuJoCo backend: loads model, wires state machine and controller.

    Usage::

        backend = MuJoCoBackend("franka_emika_panda/panda_impedance.xml", cfg)
        backend.load()
        backend.run()   # blocks until viewer is closed or FAILED
    """

    def __init__(self, xml_path: str, config: dict):
        self._xml_path = xml_path
        self._cfg = config
        self._model:         mujoco.MjModel            | None = None
        self._data:          mujoco.MjData              | None = None
        self._state_machine: JointMoveStateMachine      | None = None
        self._controller:    JointPDController          | None = None
        self._cart_controller: CartesianImpedanceController | None = None
        self._tip_kin:       SiteKinematics             | None = None
        self._ft_source:     FTMuJoCo                   | None = None
        self._ft_processor:  FTProcessor                | None = None
        self._payload_comp:  PayloadGravityCompensator  | None = None
        self._logger:        TrajectoryLogger           | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load MJCF, reset to home, create FT source, state machine, controller."""
        self._model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._data  = mujoco.MjData(self._model)

        mujoco.mj_resetDataKeyframe(self._model, self._data, 0)
        mujoco.mj_forward(self._model, self._data)

        # The home keyframe sets ctrl to joint-position-like values (matching qpos),
        # but the first 7 actuators are torque-scaled (force = 1 × ctrl).
        # Zero them so no unexpected torque is applied before the loop starts.
        self._data.ctrl[:7] = 0.0

        print(f"Loaded MJCF: {self._xml_path}")
        self._print_ft_sensor_frame()

        # ---- FT source: start immediately so data is available from tick 0 ----
        ft_cfg = self._cfg.get("ft", {})
        self._ft_source = FTMuJoCo(
            model=self._model,
            data=self._data,
            force_sensor_name=ft_cfg.get("force_sensor_name", "ft_force"),
            torque_sensor_name=ft_cfg.get("torque_sensor_name", "ft_torque"),
            site_name=ft_cfg.get("site_name", "ft_sensor_site"),
            output_frame=ft_cfg.get("output_frame", "site"),
        )
        self._ft_source.start()
        print("FT source started.")

        self._ft_processor = FTProcessor(
            sign=float(ft_cfg.get("sign", 1.0)),
            lowpass_alpha=float(ft_cfg.get("lowpass_alpha", 1.0)),
        )

        # ---- Controller + state machine ----------------------------------------
        self._controller    = JointPDController(self._cfg["joint_pd"])

        # Cartesian impedance controller + tip kinematics for the post-tare
        # APPROACH / FORCE_HOLD phases.  Controlled tip is the IK site.
        ik_cfg = self._cfg.get("ik", {})
        self._cart_controller = CartesianImpedanceController(self._cfg["controller"])
        self._tip_kin = SiteKinematics(self._model, ik_cfg["site_name"])
        # Pass the FT processor so the machine can auto-tare the payload baseline
        # once it reaches the IK goal (TARE state).
        self._state_machine = JointMoveStateMachine(
            self._cfg, ft_processor=self._ft_processor
        )

        pg_cfg = self._cfg.get("payload_gravity", {})
        if pg_cfg.get("enabled", False):
            self._payload_comp = PayloadGravityCompensator(self._cfg)
            print("Payload gravity compensator loaded.")

        # ---- Trajectory logging (tip x, v, Fz, phase per tick) ----------
        log_cfg = self._cfg.get("logging", {})
        if log_cfg.get("enabled", False):
            self._logger = TrajectoryLogger(
                output_dir=log_cfg.get("output_dir", "data/logs"),
                prefix=log_cfg.get("prefix", "sim"),
                capacity=int(log_cfg.get("capacity", 1_500_000)),
                phase_names={s.value: s.name for s in JointMoveState},
            )
            print(f"Trajectory logging enabled -> {log_cfg.get('output_dir', 'data/logs')}")

        # start() solves IK synchronously and launches the Enter-waiter thread.
        self._state_machine.start(self._model, self._data)

    def _print_ft_sensor_frame(self) -> None:
        """Print ft_sensor_site world position and orientation at home pose."""
        site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, "ft_sensor_site"
        )
        if site_id < 0:
            return
        pos = self._data.site_xpos[site_id]
        R   = self._data.site_xmat[site_id].reshape(3, 3)
        print(f"ft_sensor_site  pos : {pos.round(5)}")
        print(f"ft_sensor_site  R   :\n{R.round(4)}")

    def run(self) -> None:
        """Open passive viewer and run the control loop (blocking)."""
        with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
            # Show coordinate axes for all sites, making the IK target site visible.
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

            while viewer.is_running():
                q  = self._data.qpos[:7].copy()
                dq = self._data.qvel[:7].copy()

                # ---- FT data: read every tick, regardless of phase --------------
                raw_ft = self._ft_source.get_latest()
                processed_wrench = self._ft_processor.process(raw_ft.wrench)

                # ---- Tip kinematics (world pose / Jacobian / velocity) ----------
                x_tip, R_tip, J_tip, v_tip, w_tip = self._tip_kin.compute(q, dq)

                # ---- State machine ----------------------------------------------
                cmd = self._state_machine.update(
                    t=self._data.time,
                    q_current=q,
                    dq_current=dq,
                    wrench=processed_wrench,
                    ft_sample=raw_ft,
                    x_current=x_tip,
                    R_current=R_tip,
                )

                # ---- Trajectory log: tip x, v, Fz (world), phase ----------------
                if self._logger is not None:
                    self._logger.log(
                        self._data.time, x_tip, v_tip,
                        processed_wrench[2], self._state_machine.state.value,
                    )

                if cmd.mode == "failed":
                    # State machine reached FAILED — stop cleanly.
                    break

                # ---- Task torque: joint PD or Cartesian impedance ---------------
                if cmd.mode == "cartesian":
                    tau_task = self._cart_controller.compute(
                        x_tip, R_tip, v_tip, w_tip, J_tip,
                        cmd.x_des, cmd.dx_des, cmd.R_des, cmd.w_des,
                    )
                else:
                    tau_task = self._controller.compute(q, dq, cmd.q_des, cmd.dq_des)

                # ---- Gravity / bias compensation --------------------------------
                #
                # With payload compensator enabled:
                #   tau_internal_robot_comp = G_zero(q)  — arm-only gravity,
                #       mirrors what libfranka handles internally on real hardware.
                #   tau_payload_comp = G_full(q) - G_zero(q)  — payload delta only.
                #   Net: tau_total = G_full(q) + tau_joint_pd, which equals the
                #       full-payload sim's own qfrc_bias + PD.  No double-counting.
                #
                # Without payload compensator (fallback):
                #   Use live sim's qfrc_bias directly as the full-arm comp,
                #   payload term stays zero.
                #
                if self._payload_comp is not None:
                    tau_internal_robot_comp = self._payload_comp.gravity_zero(q)
                    tau_payload_comp        = self._payload_comp.compute(q)
                else:
                    tau_internal_robot_comp = self._data.qfrc_bias[:7].copy()
                    tau_payload_comp        = np.zeros(7)

                tau_total = tau_internal_robot_comp + tau_task + tau_payload_comp

                # MuJoCo-specific actuator scaling (see ACTUATOR_GAIN above).
                self._data.ctrl[:7] = tau_total / ACTUATOR_GAIN

                mujoco.mj_step(self._model, self._data)
                viewer.sync()

        # Flush the trajectory log once the viewer/loop has stopped.
        if self._logger is not None:
            self._logger.save()
