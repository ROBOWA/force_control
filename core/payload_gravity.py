"""Payload-only gravity compensation via two MJCF models.

    tau_payload(q) = G_full(q) - G_zero(q)

G_full: gravity torques from the full model (robot + ft_sensor + stick).
G_zero: gravity torques from the zero model (robot arm only, no payload).

On real Franka, libfranka handles the main-arm gravity internally.
The Franka controller only needs to cancel the extra payload gravity that
libfranka does not know about.  This class provides that delta.

In MuJoCo simulation:
    tau_total = G_zero(q) + tau_joint_pd + [G_full(q) - G_zero(q)]
              = G_full(q) + tau_joint_pd
which is correct: the full-payload model's own qfrc_bias equals G_full(q),
so the PD loop sees the same effective stiffness as on the real robot.
"""

from __future__ import annotations
import numpy as np
import mujoco


class PayloadGravityCompensator:
    """Compute payload-only gravity torques using two static MJCF models.

    Both models must have exactly 7 arm joints (nv >= 7).
    Scratch MjData instances are preallocated; no heap allocation at runtime.

    Usage::

        comp = PayloadGravityCompensator(config)
        tau_payload = comp.compute(q)          # G_full - G_zero, clipped
        tau_robot   = comp.gravity_zero(q)     # G_zero only (for sim fallback)
    """

    def __init__(self, config: dict):
        cfg = config["payload_gravity"]

        self._model_full = mujoco.MjModel.from_xml_path(cfg["model_full"])
        self._model_zero = mujoco.MjModel.from_xml_path(cfg["model_zero"])
        self._data_full  = mujoco.MjData(self._model_full)
        self._data_zero  = mujoco.MjData(self._model_zero)

        # Preallocate RNE output buffers — reused every call, no per-step alloc.
        self._rne_buf_full = np.zeros(self._model_full.nv)
        self._rne_buf_zero = np.zeros(self._model_zero.nv)

        clip = cfg.get("torque_clip", None)
        self._clip = float(clip) if clip is not None else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def gravity_full(self, q: np.ndarray) -> np.ndarray:
        """G_full(q): gravity torques of arm + payload, shape (7,)."""
        return self._run_rne(
            self._model_full, self._data_full, self._rne_buf_full, q
        )

    def gravity_zero(self, q: np.ndarray) -> np.ndarray:
        """G_zero(q): gravity torques of arm only (no payload), shape (7,)."""
        return self._run_rne(
            self._model_zero, self._data_zero, self._rne_buf_zero, q
        )

    def compute(self, q: np.ndarray) -> np.ndarray:
        """tau_payload = G_full(q) - G_zero(q), optionally clipped."""
        tau = self.gravity_full(q) - self.gravity_zero(q)
        if self._clip is not None:
            tau = np.clip(tau, -self._clip, self._clip)
        return tau

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _run_rne(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        rne_buf: np.ndarray,
        q: np.ndarray,
    ) -> np.ndarray:
        """Lean RNE path: kinematics → comPos → rne (flg_acc=0, qvel=0)."""
        data.qpos[:7] = q
        data.qvel[:]  = 0.0
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_rne(model, data, 0, rne_buf)
        return rne_buf[:7].copy()
