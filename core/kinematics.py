"""Tip forward kinematics + Jacobian from a measured joint configuration.

Drives a private scratch MjData with the measured q (and dq) and reads the
controlled site's world-frame pose, spatial Jacobian, and velocity.  This is
the same approach the FT world-frame rotation uses (FK from the measured q),
so sim and real share one kinematics path; only the FT source differs.

The minimal MuJoCo pipeline for a site Jacobian is mj_kinematics + mj_comPos
(comPos populates the cdof terms mj_jacSite needs); no dynamics or collision.
"""

from __future__ import annotations
import numpy as np
import mujoco


class SiteKinematics:
    """Pose / Jacobian / velocity of one named site, evaluated at (q, dq)."""

    def __init__(self, model: mujoco.MjModel, site_name: str):
        self._model = model
        self._data = mujoco.MjData(model)
        # Seed non-arm DOFs from the home keyframe so a fresh zero qpos can't
        # produce a degenerate frame; the first 7 are overwritten each call.
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, self._data, 0)

        self._sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self._sid < 0:
            raise ValueError(f"Site '{site_name}' not found in model for kinematics")

        # Preallocated Jacobian buffers (3 x nv) — reused, no per-tick alloc.
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def compute(
        self, q: np.ndarray, dq: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (x, R, J, v, w) for the site at configuration (q, dq).

            x : (3,)   world position [m]
            R : (3,3)  world orientation
            J : (6,7)  world spatial Jacobian [linear; angular]
            v : (3,)   world linear velocity [m/s]  = Jp @ dq
            w : (3,)   world angular velocity [rad/s] = Jr @ dq
        """
        d = self._data
        d.qpos[:7] = q
        mujoco.mj_kinematics(self._model, d)
        mujoco.mj_comPos(self._model, d)

        x = d.site_xpos[self._sid].copy()
        R = d.site_xmat[self._sid].reshape(3, 3).copy()

        mujoco.mj_jacSite(self._model, d, self._jacp, self._jacr, self._sid)
        Jp = self._jacp[:, :7]
        Jr = self._jacr[:, :7]
        J = np.vstack([Jp, Jr])

        dq = np.asarray(dq, dtype=float)
        v = Jp @ dq
        w = Jr @ dq
        return x, R, J, v, w
