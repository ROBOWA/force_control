"""MuJoCo FT source: reads from named force/torque sensor elements in the MJCF.

No pylibfranka or ROS imports.
"""

from __future__ import annotations
import numpy as np
import mujoco

from core.types import WrenchSample


class FTMuJoCo:
    """Reads FT data from a pair of named MuJoCo force/torque sensors.

    Sensor and site IDs are resolved once at construction; get_latest() is
    O(1) with no lookups — only a sensordata slice and an optional 3×3 matmul.

    The XML must define::

        <sensor>
            <force  name="ft_force"  site="ft_sensor_site"/>
            <torque name="ft_torque" site="ft_sensor_site"/>
        </sensor>

    Interface (same as FTRosSource)::

        ft = FTMuJoCo(model, data, ...)
        ft.start()                    # no-op for sim
        sample = ft.get_latest()      # WrenchSample, non-blocking
        ft.stop()                     # no-op for sim
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        force_sensor_name: str = "ft_force",
        torque_sensor_name: str = "ft_torque",
        site_name: str = "ft_sensor_site",
        output_frame: str = "site",   # "site" | "world"
    ):
        """
        Args:
            model:               MuJoCo model (read-only after init).
            data:                Live MjData — sensordata is read every tick.
            force_sensor_name:   Name of the <force> sensor element.
            torque_sensor_name:  Name of the <torque> sensor element.
            site_name:           Measurement site (used for world-frame rotation).
            output_frame:        "site" returns sensor-frame wrench;
                                 "world" rotates via site_xmat into world frame.
        """
        self._data = data
        self._output_frame = output_frame
        self._seq = 0

        fsid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, force_sensor_name)
        tsid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, torque_sensor_name)
        self._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

        if fsid < 0:
            raise ValueError(f"Force sensor '{force_sensor_name}' not found in model")
        if tsid < 0:
            raise ValueError(f"Torque sensor '{torque_sensor_name}' not found in model")
        if output_frame == "world" and self._site_id < 0:
            raise ValueError(
                f"Site '{site_name}' not found — required for output_frame='world'"
            )

        self._force_adr = int(model.sensor_adr[fsid])
        self._torque_adr = int(model.sensor_adr[tsid])

    # ------------------------------------------------------------------
    # Common FT source interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """No-op: MuJoCo sensordata is always live after mj_step."""

    def stop(self) -> None:
        """No-op."""

    def get_latest(self) -> WrenchSample:
        """Read current sensordata and return a WrenchSample. Non-blocking."""
        f = self._data.sensordata[self._force_adr  : self._force_adr  + 3].copy()
        t = self._data.sensordata[self._torque_adr : self._torque_adr + 3].copy()

        if self._output_frame == "world":
            R_W_S = self._data.site_xmat[self._site_id].reshape(3, 3)
            f = R_W_S @ f
            t = R_W_S @ t

        self._seq += 1
        return WrenchSample(
            t=float(self._data.time),
            wrench=np.r_[f, t].copy(),
            seq=self._seq,
            valid=True,
        )
