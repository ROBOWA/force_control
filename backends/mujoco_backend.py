"""MuJoCo simulation backend — Mac development mode.

Milestone 1: thin backend that wires MuJoCo model/data to the state machine
and joint PD controller.  No IK, no trajectory math, no PD formula here.
"""

from __future__ import annotations
import numpy as np
import mujoco
import mujoco.viewer

from core.joint_move_state_machine import JointMoveStateMachine
from core.controller import JointPDController


# panda.xml actuator definition (joints 1–7):
#   <general gainprm="500" biastype="none"/>
#   applied force = 500 × ctrl  (no bias subtracted)
#   → to command torque τ: ctrl = τ / ACTUATOR_GAIN
# forcerange=[-87, 87] Nm clips the output inside MuJoCo.
# This constant lives here because it is a property of the XML, not the controller.
ACTUATOR_GAIN = 500.0


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
        self._model:         mujoco.MjModel       | None = None
        self._data:          mujoco.MjData         | None = None
        self._state_machine: JointMoveStateMachine | None = None
        self._controller:    JointPDController     | None = None

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

        self._controller    = JointPDController(self._cfg["joint_pd"])
        self._state_machine = JointMoveStateMachine(self._cfg)

        # start() solves IK synchronously and launches the Enter-waiter thread.
        self._state_machine.start(self._model, self._data)

    def run(self) -> None:
        """Open passive viewer and run the control loop (blocking)."""
        with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
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
                # In MuJoCo simulation:
                #   qfrc_bias (gravity + Coriolis) is fed forward here to emulate
                #   Franka's internal main-arm compensation.  This is NOT our
                #   controller doing full-arm gravity compensation — it is a
                #   sim-only shim so that the PD gains feel the same as on hardware.
                #
                # In real Franka execution:
                #   Main Panda arm gravity compensation is handled internally by
                #   libfranka/the control stack.  Our extra compensation should
                #   later only cancel payload gravity (FT sensor + tool + stick)
                #   that libfranka does not know about.
                #
                tau_internal_robot_comp = self._data.qfrc_bias[:7].copy()

                # TODO: add payload-only gravity torques (FT sensor/tool/stick)
                #       once the payload model is known.
                tau_payload_comp = np.zeros(7)

                tau_total = tau_internal_robot_comp + tau_joint_pd + tau_payload_comp

                # MuJoCo-specific actuator scaling (see ACTUATOR_GAIN above).
                self._data.ctrl[:7] = tau_total / ACTUATOR_GAIN

                mujoco.mj_step(self._model, self._data)
                viewer.sync()
