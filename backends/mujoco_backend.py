"""MuJoCo simulation backend — Mac development mode.

Runs the full control loop inside the MuJoCo viewer at the sim timestep.
No pylibfranka or ROS imports.
"""

from __future__ import annotations
import numpy as np
import mujoco
import mujoco.viewer

from force_control.core.types import RobotStateLite, TargetSample, WrenchSample
from force_control.core.controller import ControllerCore
from force_control.core.state_machine import StateMachine
from force_control.sensors.ft_mujoco import FTMuJoCo


# Indices of the 7 Panda actuators in mj_data.ctrl
_PANDA_CTRL_IDX = slice(0, 7)

# MuJoCo body / site names (must match the MJCF)
_EE_SITE_NAME = "attachment_site"


class MuJoCoBackend:
    """Wraps MuJoCo model/data and drives ControllerCore at the sim rate.

    Typical usage::

        backend = MuJoCoBackend("franka_emika_panda/scene.xml", cfg)
        backend.run()
    """

    def __init__(self, xml_path: str, config: dict):
        """
        Args:
            xml_path: path to the MJCF scene file
            config:   merged sim config dict (from configs/sim.yaml)
        """
        self._xml_path = xml_path
        self._cfg = config
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._controller: ControllerCore | None = None
        self._state_machine: StateMachine | None = None
        self._ft_source: FTMuJoCo | None = None
        self._current_target: TargetSample | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the MJCF and initialise MuJoCo model/data."""
        # TODO: implement
        # self._model = mujoco.MjModel.from_xml_path(self._xml_path)
        # self._data  = mujoco.MjData(self._model)
        # mujoco.mj_resetDataKeyframe(self._model, self._data, 0)
        # mujoco.mj_forward(self._model, self._data)
        raise NotImplementedError

    def run(self) -> None:
        """Run the passive viewer loop (blocking)."""
        # TODO: implement
        # with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
        #     while viewer.is_running():
        #         state  = self._read_state()
        #         wrench = self._ft_source.get_latest()
        #         target = self._state_machine.update(...)
        #         tau    = self._controller.step(state, target, wrench, dt)
        #         self._apply_tau(tau)
        #         mujoco.mj_step(self._model, self._data)
        #         viewer.sync()
        raise NotImplementedError

    # ------------------------------------------------------------------
    # State reading
    # ------------------------------------------------------------------

    def _read_state(self) -> RobotStateLite:
        """Convert current mj_data into RobotStateLite."""
        # TODO: implement
        # q        = self._data.qpos[:7].copy()
        # dq       = self._data.qvel[:7].copy()
        # O_T_EE   = self._ee_transform()
        # J        = self._jacobian()
        # coriolis = self._data.qfrc_bias[:7].copy()  # gravity+coriolis in sim
        # t        = self._data.time
        raise NotImplementedError

    def _ee_transform(self) -> np.ndarray:
        """Return 4×4 base→EE homogeneous transform from MuJoCo site."""
        # TODO: implement via mujoco.mj_forward + site_xpos/site_xmat
        raise NotImplementedError

    def _jacobian(self) -> np.ndarray:
        """Return 6×7 geometric Jacobian for the EE site."""
        # TODO: implement via mujoco.mj_jacSite
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Command output
    # ------------------------------------------------------------------

    def _apply_tau(self, tau: np.ndarray) -> None:
        """Write joint torques to mj_data.ctrl."""
        self._data.ctrl[_PANDA_CTRL_IDX] = tau

    # ------------------------------------------------------------------
    # Target initialisation
    # ------------------------------------------------------------------

    def _make_initial_target(self) -> TargetSample:
        """Build the initial TargetSample from the home keyframe pose."""
        # TODO: implement — read current EE pose, zero velocity/force targets
        raise NotImplementedError
