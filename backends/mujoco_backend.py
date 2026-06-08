"""MuJoCo simulation backend — Mac development mode.

Milestone 1: thin backend that wires MuJoCo model/data to the state machine
and joint PD controller.  No IK, no trajectory math, no PD formula here.
"""

from __future__ import annotations
import numpy as np
import mujoco
import mujoco.viewer

from core.state_machine import JointMoveStateMachine
from core.controller import JointPDController
from core.payload_gravity import PayloadGravityCompensator


# panda.xml actuator definition (joints 1–7):
#   <general gainprm="500" biastype="none"/>
#   applied force = 500 × ctrl  (no bias subtracted)
#   → to command torque τ: ctrl = τ / ACTUATOR_GAIN
# forcerange=[-87, 87] Nm clips the output inside MuJoCo.
# This constant lives here because it is a property of the XML, not the controller.
ACTUATOR_GAIN = 1.0


class MuJoCoBackend:
    """Thin MuJoCo backend: loads model, wires state machine and controller.

    Usage::

        backend = MuJoCoBackend("franka_emika_panda/scene.xml", cfg)
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
        self._payload_comp:  PayloadGravityCompensator  | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load MJCF, reset to home, create state machine and controller."""
        self._model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._data  = mujoco.MjData(self._model)

        mujoco.mj_resetDataKeyframe(self._model, self._data, 0)
        mujoco.mj_forward(self._model, self._data)

        # The home keyframe sets ctrl to joint-position-like values (matching qpos),
        # but the first 7 actuators are torque-scaled (force = 500 × ctrl).
        # Zero them so no unexpected torque is applied before the loop starts.
        self._data.ctrl[:7] = 0.0

        print(f"Loaded MJCF: {self._xml_path}")
        self._print_ft_sensor_frame()

        self._controller    = JointPDController(self._cfg["joint_pd"])
        self._state_machine = JointMoveStateMachine(self._cfg)

        pg_cfg = self._cfg.get("payload_gravity", {})
        if pg_cfg.get("enabled", False):
            self._payload_comp = PayloadGravityCompensator(self._cfg)
            print("Payload gravity compensator loaded.")

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

                q_des, dq_des = self._state_machine.update(
                    t=self._data.time,
                    q_current=q,
                )

                if q_des is None:
                    # State machine reached FAILED — stop cleanly.
                    break

                tau_joint_pd = self._controller.compute(q, dq, q_des, dq_des)

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

                tau_total = tau_internal_robot_comp + tau_joint_pd + tau_payload_comp

                # MuJoCo-specific actuator scaling (see ACTUATOR_GAIN above).
                self._data.ctrl[:7] = tau_total / ACTUATOR_GAIN

                mujoco.mj_step(self._model, self._data)
                viewer.sync()
