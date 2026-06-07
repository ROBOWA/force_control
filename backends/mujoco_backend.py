"""MuJoCo simulation backend — Mac development mode.

Milestone 1: joint-space IK + min-jerk trajectory + joint PD torque control.
No Cartesian impedance, no FT sensor, no contact, no ROS.
"""

from __future__ import annotations
from enum import Enum, auto
import numpy as np
import mujoco
import mujoco.viewer

from core.ik import min_jerk, solve_ik_dls


# panda.xml actuators: force = ACTUATOR_GAIN * ctrl
# To command torque τ: set ctrl = τ / ACTUATOR_GAIN
ACTUATOR_GAIN = 500.0


class Phase(Enum):
    SOLVE_IK   = auto()
    MOVE_JOINT = auto()
    HOLD_JOINT = auto()


class MuJoCoBackend:
    """Minimal MuJoCo backend: IK → min-jerk move → hold.

    Usage::

        backend = MuJoCoBackend("franka_emika_panda/scene.xml", cfg)
        backend.load()
        backend.run()   # blocks until viewer is closed
    """

    def __init__(self, xml_path: str, config: dict):
        self._xml_path = xml_path
        self._cfg = config
        self._model: mujoco.MjModel | None = None
        self._data:  mujoco.MjData  | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load MJCF, reset to home keyframe, print diagnostics."""
        self._model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._data  = mujoco.MjData(self._model)
        mujoco.mj_resetDataKeyframe(self._model, self._data, 0)
        mujoco.mj_forward(self._model, self._data)

        cfg_ik    = self._cfg["ik"]
        site_name = cfg_ik["site_name"]
        target_pos = np.array(cfg_ik["target_pos"])

        site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, site_name
        )
        if site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in {self._xml_path}")

        print(f"Loaded MJCF: {self._xml_path}")
        print(f"Using site:  {site_name}")
        print(f"Initial site position: {self._data.site_xpos[site_id].round(4)}")
        print(f"Target site position:  {target_pos}")

    def run(self) -> None:
        """Open passive viewer and run the control loop (blocking)."""
        model = self._model
        data  = self._data
        cfg_ik = self._cfg["ik"]
        cfg_pd = self._cfg["joint_pd"]

        site_name  = cfg_ik["site_name"]
        target_pos = np.array(cfg_ik["target_pos"])
        move_time  = float(cfg_ik.get("move_time", 5.0))
        keep_ori   = bool(cfg_ik.get("keep_current_orientation", True))
        kp = np.array(cfg_pd["kp"], dtype=float)
        kd = np.array(cfg_pd["kd"], dtype=float)

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

        # ---- Phase SOLVE_IK ------------------------------------------
        q_init   = data.qpos[:7].copy()
        target_R = data.site_xmat[site_id].copy().reshape(3, 3) if keep_ori else np.eye(3)

        q_ik, n_iter, err, converged = solve_ik_dls(
            model, data, site_id, q_init, target_pos, target_R
        )
        print(f"IK converged: {converged}  (iter={n_iter}, err={err:.5f})")
        print(f"q_start: {q_init.round(4)}")
        print(f"q_goal:  {q_ik.round(4)}")

        # IK perturbs qpos during FK calls — restore to keyframe pose
        data.qpos[:7] = q_init
        data.qvel[:7] = 0.0
        mujoco.mj_forward(model, data)

        # ---- Phase MOVE_JOINT / HOLD_JOINT ---------------------------
        q_start = q_init.copy()
        q_goal  = q_ik.copy()
        phase   = Phase.MOVE_JOINT
        move_start_time = data.time
        hold_printed = False

        print(f"Moving for {move_time:.1f} s ...")

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                t_elapsed = data.time - move_start_time

                # ---- Desired trajectory ----------------------------
                if phase == Phase.MOVE_JOINT:
                    if t_elapsed < move_time:
                        s, ds = min_jerk(t_elapsed, move_time)
                        q_des  = q_start + s  * (q_goal - q_start)
                        dq_des = ds * (q_goal - q_start)
                    else:
                        phase  = Phase.HOLD_JOINT
                        q_des  = q_goal
                        dq_des = np.zeros(7)

                if phase == Phase.HOLD_JOINT:
                    if not hold_printed:
                        print("Reached HOLD_JOINT")
                        hold_printed = True
                    q_des  = q_goal
                    dq_des = np.zeros(7)

                # ---- Joint-space PD + gravity compensation ----------
                q  = data.qpos[:7].copy()
                dq = data.qvel[:7].copy()
                tau = kp * (q_des - q) + kd * (dq_des - dq) + data.qfrc_bias[:7]

                # Map torque to ctrl (see ACTUATOR_GAIN comment at top)
                data.ctrl[:7] = tau / ACTUATOR_GAIN

                mujoco.mj_step(model, data)
                viewer.sync()
