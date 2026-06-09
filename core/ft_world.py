"""Rotate a sensor-frame wrench into the world frame using MuJoCo FK.

The FT sensor reports force/torque in its own (sensor-site) frame, which
rotates with the wrist as the robot moves.  For a payload tare captured at one
orientation to stay valid while the robot moves to another, the wrench must be
expressed in a fixed frame.  This helper computes the sensor-site -> world
rotation R_WS(q) from the current joint configuration via MuJoCo forward
kinematics and rotates the force and torque 3-vectors independently::

    f_world = R_WS @ f_sensor
    t_world = R_WS @ t_sensor

Only orientation is applied — no moment-arm / translation term ("only
orientation matters").

This mirrors what FTMuJoCo does in sim with output_frame="world" (there the
live site_xmat is read straight from the running model); on real hardware
there is no live MuJoCo state, so we drive a scratch MjData with the measured q
and read its site_xmat.
"""

from __future__ import annotations
import dataclasses
import numpy as np
import mujoco

from core.types import WrenchSample


class FTWorldRotator:
    """Rotates sensor-frame wrenches into world frame from the current q.

    A private scratch MjData is driven by q each call; the live control/IK
    state is never touched.  Uses mj_kinematics (pose-only FK — no dynamics,
    no collision), so the per-tick cost is a few microseconds.
    """

    def __init__(self, model: mujoco.MjModel, site_name: str = "ft_sensor_site"):
        self._model = model
        self._data = mujoco.MjData(model)
        # Seed non-arm DOFs (e.g. fingers) from the home keyframe so a fresh,
        # all-zero qpos can't yield a degenerate frame; we overwrite [:7] each call.
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, self._data, 0)

        self._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self._site_id < 0:
            raise ValueError(
                f"Site '{site_name}' not found in model for FT world-frame rotation"
            )

    def R_world_sensor(self, q: np.ndarray) -> np.ndarray:
        """Return the 3x3 sensor-site -> world rotation at configuration q (7,)."""
        self._data.qpos[:7] = q
        mujoco.mj_kinematics(self._model, self._data)
        return self._data.site_xmat[self._site_id].reshape(3, 3).copy()

    def to_world(self, q: np.ndarray, sample: WrenchSample) -> WrenchSample:
        """Return a copy of `sample` with its wrench rotated into world frame.

        Invalid samples (no data yet) are returned unchanged.
        """
        if not sample.valid:
            return sample
        R = self.R_world_sensor(q)
        w = np.asarray(sample.wrench, dtype=float)
        world = np.empty(6)
        world[:3] = R @ w[:3]
        world[3:] = R @ w[3:]
        return dataclasses.replace(sample, wrench=world)
