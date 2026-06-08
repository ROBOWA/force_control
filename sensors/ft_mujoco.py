"""MuJoCo FT source: reads contact/sensor forces from mj_data.

No pylibfranka or ROS imports.
"""

from __future__ import annotations
import numpy as np
import mujoco

from core.types import WrenchSample


class FTMuJoCo:
    """Reads FT data from MuJoCo contact forces or a force/torque sensor site.

    Two strategies are supported:
        "sensor"  — read from a named MuJoCo force/torque sensor element
        "contact" — sum contact forces on a named body (less accurate)
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        strategy: str = "sensor",
        sensor_name: str = "ft_sensor",
        contact_body: str = "hand",
    ):
        self._model = model
        self._data = data
        self._strategy = strategy
        self._sensor_name = sensor_name
        self._contact_body = contact_body
        self._seq = 0

    def get_latest(self) -> WrenchSample:
        """Return the current FT reading as a WrenchSample."""
        if self._strategy == "sensor":
            wrench = self._read_sensor()
        else:
            wrench = self._read_contact()

        self._seq += 1
        return WrenchSample(
            t=self._data.time,
            wrench=wrench,
            seq=self._seq,
        )

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _read_sensor(self) -> np.ndarray:
        """Read a 6-DOF force/torque sensor defined in the MJCF."""
        # TODO: implement
        # sensor_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR,
        #                                self._sensor_name)
        # adr = self._model.sensor_adr[sensor_id]
        # return self._data.sensordata[adr : adr + 6].copy()
        return np.zeros(6)

    def _read_contact(self) -> np.ndarray:
        """Sum contact forces on the named body in world frame."""
        # TODO: implement
        # body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,
        #                              self._contact_body)
        # wrench = np.zeros(6)
        # for i in range(self._data.ncon):
        #     contact = self._data.contact[i]
        #     if contact.geom1_body == body_id or contact.geom2_body == body_id:
        #         ...  # accumulate
        return np.zeros(6)
