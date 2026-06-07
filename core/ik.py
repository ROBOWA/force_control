"""IK and trajectory helpers.

These functions are the only place in core/ that imports mujoco.
No pylibfranka or ROS imports.
"""

import numpy as np
import mujoco


Q_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_HI = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])


def min_jerk(t: float, T: float) -> tuple[float, float]:
    """Scalar min-jerk interpolation s(t) and its time derivative ds(t).

    Args:
        t: current time [s]
        T: total move duration [s]

    Returns:
        (s, ds) where s ∈ [0,1] is the normalised position along the path
        and ds [1/s] is its rate of change.
    """
    s_t = float(np.clip(t / T, 0.0, 1.0))
    s  = 10.0 * s_t**3 - 15.0 * s_t**4 + 6.0 * s_t**5
    ds = (30.0 * s_t**2 - 60.0 * s_t**3 + 30.0 * s_t**4) / max(T, 1e-9)
    return s, ds


def solve_ik_dls(
    mj_model,
    mj_data,
    site_id: int,
    q_init: np.ndarray,
    target_pos: np.ndarray,
    target_R: np.ndarray,
    max_iter: int = 200,
    tol: float = 5e-4,
    lambda_: float = 0.05,
    step: float = 1.0,
) -> tuple[np.ndarray, int, float, bool]:
    """Damped-Least-Squares IK for the 7-DoF Panda arm.

    Modifies mj_data.qpos[:7] internally for forward-kinematics calls.
    Caller must restore qpos after this call if the original pose is needed.

    Args:
        mj_model:   MjModel
        mj_data:    MjData (qpos will be temporarily overwritten)
        site_id:    index of the target site
        q_init:     initial joint configuration, shape (7,)
        target_pos: desired site position in world frame, shape (3,)
        target_R:   desired site orientation in world frame, shape (3,3)
        max_iter:   maximum iterations
        tol:        convergence tolerance on the 6-D error norm
        lambda_:    DLS damping factor
        step:       step size multiplier

    Returns:
        (q, n_iter, err_norm, converged)
    """
    q = np.asarray(q_init, dtype=float).copy()
    Jp = np.zeros((3, mj_model.nv))
    Jr = np.zeros((3, mj_model.nv))
    e6 = np.zeros(6)
    err_norm = float("inf")

    for i in range(max_iter):
        mj_data.qpos[:7] = q
        mj_data.qvel[:7] = 0.0
        mujoco.mj_forward(mj_model, mj_data)

        cur_pos = mj_data.site_xpos[site_id].copy()
        cur_R   = mj_data.site_xmat[site_id].copy().reshape(3, 3)

        pos_err = target_pos - cur_pos
        R_err   = target_R @ cur_R.T
        ori_err = 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ])

        e6[:3] = pos_err
        e6[3:] = ori_err
        err_norm = float(np.linalg.norm(e6))

        if err_norm < tol:
            return q, i, err_norm, True

        mujoco.mj_jacSite(mj_model, mj_data, Jp, Jr, site_id)
        J  = np.vstack([Jp[:, :7], Jr[:, :7]])
        dq = J.T @ np.linalg.solve(J @ J.T + (lambda_ ** 2) * np.eye(6), e6)
        q  = np.clip(q + step * dq, Q_LO, Q_HI)

    return q, max_iter, err_norm, False
