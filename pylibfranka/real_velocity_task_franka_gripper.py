
#!/usr/bin/env python3
"""
Debug IK move for real Franka robot.

What this script does:
1. Connect to robot server and read current state
2. Check whether handwritten FK matches robot-reported O_T_EE
3. Build a target pose
4. Solve IK using handwritten FK/Jacobian
5. Move in joint position mode
6. Read back actual pose and report final error

IMPORTANT:
- This script is only trustworthy if [Model check] passes.
- If [Model check] fails, your FK / EE frame / tool transform does not match
  the robot's actual O_T_EE definition.
"""

import argparse
import time
import numpy as np

from franka_server import (
    FrankaClient,
    ControlMode,
    pose_to_matrix,
    get_position,
)

# =========================
# User parameters
# =========================
TARGET_X = 0.40
TARGET_Y = 0.00
TARGET_Z = 0.02

TILT_DEG = 15.0
TILT_FRAME = "local"   # "local" or "base"

CONTROL_RATE = 100.0
APPROACH_DURATION = 5.0

IK_POS_TOL = 2e-3                 # 2 mm
IK_ROT_TOL = np.deg2rad(3.0)      # about 3 deg

MODEL_POS_TOL = 5e-3              # 5 mm
MODEL_ROT_TOL = np.deg2rad(2.0)   # 2 deg

# =========================
# Franka Panda kinematics
# =========================
_DH_A = [0, 0, 0, 0.0825, -0.0825, 0, 0.088]
_DH_D = [0.333, 0, 0.316, 0, 0.384, 0, 0]
_DH_ALPHA = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2]
_EE_D = 0.107

_JOINT_LIMITS = np.array([
    [-2.8973,  2.8973],
    [-1.7628,  1.7628],
    [-2.8973,  2.8973],
    [-3.0718, -0.0698],
    [-2.8973,  2.8973],
    [-0.0175,  3.7525],
    [-2.8973,  2.8973],
], dtype=float)

_JOINT_CENTERS = (_JOINT_LIMITS[:, 0] + _JOINT_LIMITS[:, 1]) / 2.0


def dh_mat(a, d, alpha, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,       ca,      d],
        [0,         0,        0,      1],
    ], dtype=float)


def franka_fk(q: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for i in range(7):
        T = T @ dh_mat(_DH_A[i], _DH_D[i], _DH_ALPHA[i], q[i])

    T_ee = np.eye(4)
    T_ee[2, 3] = _EE_D
    return T @ T_ee


def franka_jacobian(q: np.ndarray) -> np.ndarray:
    """
    6x7 geometric Jacobian for the same handwritten FK chain.
    """
    frames = [np.eye(4)]
    for i in range(7):
        frames.append(frames[-1] @ dh_mat(_DH_A[i], _DH_D[i], _DH_ALPHA[i], q[i]))

    T_ee = np.eye(4)
    T_ee[2, 3] = _EE_D
    p_e = (frames[7] @ T_ee)[:3, 3]

    J = np.zeros((6, 7))
    for i in range(7):
        z_i = frames[i][:3, 2]
        p_i = frames[i][:3, 3]
        J[:3, i] = np.cross(z_i, p_e - p_i)
        J[3:, i] = z_i
    return J


def rot_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """
    Returns the relative rotation angle between two rotation matrices.
    """
    R = Ra @ Rb.T
    val = (np.trace(R) - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    return np.arccos(val)


def tilt_R_around_Y(R_in: np.ndarray, deg: float, frame: str = "local") -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    R_y = np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ], dtype=float)

    if frame == "base":
        return R_y @ R_in
    elif frame == "local":
        return R_in @ R_y
    else:
        raise ValueError("frame must be 'base' or 'local'")


def ik_check(q: np.ndarray, target_pose: np.ndarray):
    T = franka_fk(q)
    pos_err = np.linalg.norm(T[:3, 3] - target_pose[:3, 3])
    rot_err = rot_angle(T[:3, :3], target_pose[:3, :3])
    ok = (pos_err < IK_POS_TOL) and (rot_err < IK_ROT_TOL)
    return ok, pos_err, rot_err


def franka_ik(
    target_pose: np.ndarray,
    q_init: np.ndarray,
    max_iter: int = 300,
    n_random_seeds: int = 20,
    n_random_samples: int = 500,
) -> np.ndarray:
    """
    Damped least-squares IK with multiple seeds.
    """
    p_tgt = target_pose[:3, 3]
    R_tgt = target_pose[:3, :3]

    lam = 1e-3
    k_null = 0.05
    step_limit = 0.10

    rng = np.random.default_rng(42)
    q_rand = rng.uniform(_JOINT_LIMITS[:, 0], _JOINT_LIMITS[:, 1], (n_random_samples, 7))
    pos_dists = np.array([
        np.linalg.norm(franka_fk(q)[:3, 3] - p_tgt) for q in q_rand
    ])
    top_idx = np.argsort(pos_dists)[:n_random_seeds]

    seeds = [q_init.copy(), _JOINT_CENTERS.copy()]
    seeds += [q_rand[i].copy() for i in top_idx]

    best_q = q_init.copy()
    best_err = np.inf

    for seed in seeds:
        q = np.clip(seed, _JOINT_LIMITS[:, 0], _JOINT_LIMITS[:, 1])

        for _ in range(max_iter):
            T = franka_fk(q)
            p_err = p_tgt - T[:3, 3]

            R_e = R_tgt @ T[:3, :3].T
            r_err = 0.5 * np.array([
                R_e[2, 1] - R_e[1, 2],
                R_e[0, 2] - R_e[2, 0],
                R_e[1, 0] - R_e[0, 1],
            ])

            dx = np.concatenate([p_err, r_err])
            err = np.linalg.norm(dx)

            if err < best_err:
                best_err = err
                best_q = q.copy()

            if np.linalg.norm(p_err) < 1e-5 and np.linalg.norm(r_err) < 1e-5:
                return q

            J = franka_jacobian(q)
            J_pinv = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(6))

            dq_task = J_pinv @ dx
            dq_null = (np.eye(7) - J_pinv @ J) @ (-k_null * (q - _JOINT_CENTERS))
            dq = dq_task + dq_null

            dq = np.clip(dq, -step_limit, step_limit)
            q = np.clip(q + dq, _JOINT_LIMITS[:, 0], _JOINT_LIMITS[:, 1])

    return best_q


def stream_joints(client, q: np.ndarray, dry_run: bool):
    if not dry_run:
        client.send_joint_position(q, blocking=False)


def move_joints(client, q_from: np.ndarray, q_to: np.ndarray, n_steps: int, dt: float, dry_run: bool):
    for i in range(n_steps):
        alpha = (i + 1) / n_steps
        q_cmd = (1.0 - alpha) * q_from + alpha * q_to
        stream_joints(client, q_cmd, dry_run)
        time.sleep(dt)


def wait_for_fresh_state(client, timeout=2.0, sleep_dt=0.01):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if client.latest_state is not None:
            return client.latest_state
        time.sleep(sleep_dt)
    return None


def main(args):
    print(f"Connecting to FrankaServer at {args.ip} ...")
    client = FrankaClient(server_ip=args.ip)
    client.start()

    try:
        if not client.wait_for_state(timeout=5.0):
            raise RuntimeError("Failed to receive initial robot state.")

        state = client.latest_state
        if state is None:
            raise RuntimeError("latest_state is None.")

        current_q = np.array(state.q, dtype=float)
        real_T0 = pose_to_matrix(state.O_T_EE)

        print(f"Connected.")
        print(f"Current q: {np.round(current_q, 4)}")
        print(f"Robot-reported EE position: {np.round(real_T0[:3, 3], 4)}")

        # =========================
        # 1) Model consistency check
        # =========================
        fk_T0 = franka_fk(current_q)
        model_pos_err = np.linalg.norm(fk_T0[:3, 3] - real_T0[:3, 3])
        model_rot_err = rot_angle(fk_T0[:3, :3], real_T0[:3, :3])

        print("\n[Model check]")
        print(f"  FK pos   = {np.round(fk_T0[:3, 3], 4)}")
        print(f"  Real pos = {np.round(real_T0[:3, 3], 4)}")
        print(f"  Position error    = {model_pos_err*1000:.2f} mm")
        print(f"  Orientation error = {np.rad2deg(model_rot_err):.2f} deg")

        if model_pos_err > MODEL_POS_TOL or model_rot_err > MODEL_ROT_TOL:
            raise RuntimeError(
                "Handwritten FK does NOT match robot-reported O_T_EE. "
                "Do not trust this IK result yet. "
                "Most likely your EE/tool transform or frame convention is inconsistent."
            )

        # =========================
        # 2) Build target pose
        # =========================
        target_R = tilt_R_around_Y(real_T0[:3, :3], args.tilt_deg, frame=args.tilt_frame)

        target_pose = np.eye(4)
        target_pose[:3, :3] = target_R
        target_pose[:3, 3] = np.array([args.target_x, args.target_y, args.target_z], dtype=float)

        print("\n[Target pose]")
        print(f"  Position = {np.round(target_pose[:3, 3], 4)}")
        print(f"  Tilt     = {args.tilt_deg:.2f} deg around {args.tilt_frame} Y")

        # =========================
        # 3) Solve IK
        # =========================
        print("\n[IK] Solving ...")
        q_target = franka_ik(target_pose, current_q)

        ik_ok, ik_pos_err, ik_rot_err = ik_check(q_target, target_pose)
        pred_T = franka_fk(q_target)
        print(q_target)
        print(f"  q_target = {np.round(q_target, 4)}")
        print(f"  Pred pos = {np.round(pred_T[:3, 3], 4)}")
        print(f"  IK pos err = {ik_pos_err*1000:.2f} mm")
        print(f"  IK rot err = {np.rad2deg(ik_rot_err):.2f} deg")
        print(f"  IK status  = {'PASS' if ik_ok else 'FAIL'}")

        if not ik_ok:
            raise RuntimeError("IK did not reach the requested target pose in the handwritten model.")

        if args.dry_run:
            print("\n[DRY RUN] No commands sent.")
            return

        ans = input("\nMove robot to q_target? (y/N): ").strip().lower()
        if ans != "y":
            print("Cancelled by user.")
            return

        # =========================
        # 4) Send motion
        # =========================
        print("\n[Motion] Switching to JOINT_POSITION mode ...")
        client.set_control_mode(ControlMode.JOINT_POSITION)
        time.sleep(0.2)

        dt = 1.0 / CONTROL_RATE
        n_steps = max(1, int(APPROACH_DURATION * CONTROL_RATE))

        print(f"[Motion] Moving over {APPROACH_DURATION:.2f} s ...")
        move_joints(client, current_q, q_target, n_steps, dt, dry_run=False)

        time.sleep(0.5)
        state_after = wait_for_fresh_state(client, timeout=2.0)
        if state_after is None:
            raise RuntimeError("No robot state received after motion.")

        real_T1 = pose_to_matrix(state_after.O_T_EE)
        final_q = np.array(state_after.q, dtype=float)

        # =========================
        # 5) Compare actual pose to target
        # =========================
        actual_pos_err = np.linalg.norm(real_T1[:3, 3] - target_pose[:3, 3])
        actual_rot_err = rot_angle(real_T1[:3, :3], target_pose[:3, :3])

        print("\n[Actual robot result]")
        print(f"  Final q       = {np.round(final_q, 4)}")
        print(f"  Actual pos    = {np.round(real_T1[:3, 3], 4)}")
        print(f"  Target pos    = {np.round(target_pose[:3, 3], 4)}")
        print(f"  Pos err       = {actual_pos_err*1000:.2f} mm")
        print(f"  Rot err       = {np.rad2deg(actual_rot_err):.2f} deg")

        print("\nStopping ...")
        client.set_control_mode(ControlMode.IDLE)
        print("Done.")

    finally:
        try:
            client.stop()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug IK move for Franka real robot")

    parser.add_argument("--ip", type=str, default="localhost", help="Franka server IP")
    parser.add_argument("--dry-run", action="store_true", help="Solve and print only, do not move robot")

    parser.add_argument("--target-x", type=float, default=TARGET_X)
    parser.add_argument("--target-y", type=float, default=TARGET_Y)
    parser.add_argument("--target-z", type=float, default=TARGET_Z)

    parser.add_argument("--tilt-deg", type=float, default=TILT_DEG)
    parser.add_argument(
        "--tilt-frame",
        type=str,
        default=TILT_FRAME,
        choices=["local", "base"],
        help="Apply Y-tilt in local EE frame or base frame",
    )

    main(parser.parse_args())