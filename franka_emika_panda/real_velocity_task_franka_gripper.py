#!/usr/bin/env python3
"""
Real robot code: end-effector sweeps back and forth along Y axis, keeping X=0.4, Z=TARGET_Z, tilted 15° around base Y axis.
Corresponds to the task in sim_velocity_task_franka_gripper copy.py.

Strategy:
    Phase 1  — Switch to Joint Position mode, slowly (~5s) interpolate from current pose to start pose.
    Phase 2  — Uniform reciprocating motion in Y direction via feedforward, X/Z/orientation held constant.
    Phase 3  — Lift and return to initial pose.

Notes (real robot vs simulation):
    * TARGET_Z is "Z in the base frame". In sim it is 1mm; on the real robot, adjust it based on your calibrated K frame and table /
      sandpaper surface height! Default is 0.02; use --dry-run to verify trajectory before contact.
    * O_T_EE is the flange→EE transform, assuming you configured NE_T_EE pointing to the tool tip in libfranka.
    * IK uses numerical optimization (L-BFGS-B) warm-started from the current joint state each step.
    * The code does not replicate the sim's IK + Jacobian (that is done on the server side); instead it directly sends
      joint position commands.

Usage:
    # First start the server (on the machine that controls the robot)
    python -m franka_server.server --ip 172.16.0.2

    # Dry run (no commands sent)
    python real_velocity_task_franka_gripper.py --ip <server_ip> --dry-run

    # Execute (will prompt for confirmation)
    python real_velocity_task_franka_gripper.py --ip <server_ip>
"""

import argparse
import time
import numpy as np
from scipy.optimize import minimize
from franka_server import (
    FrankaClient,
    ControlMode,
    pose_to_matrix,
    get_position,
)


# --- Trajectory parameters ---
TARGET_X = 0.4
TARGET_Z = 0.02       # For real robot, start with 2cm; adjust after manually aligning to table
TARGET_SPEED_Y = 0.05
MOVE_RANGE_Y = 0.1
TILT_DEG = 15

# --- Real robot parameters ---
CONTROL_RATE = 100          # Hz
APPROACH_DURATION = 5.0     # Duration to approach start pose
SWEEP_DURATION = 60.0       # Total sweep duration
LIFT_HEIGHT = 0.05          # Lift 5cm at the end

# --- Franka Panda DH parameters ---
_DH_A     = [0,       0,        0,       0.0825, -0.0825,  0,      0.088]
_DH_D     = [0.333,   0,        0.316,   0,       0.384,   0,      0    ]
_DH_ALPHA = [0,      -np.pi/2,  np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2]
_EE_D     = 0.107   # flange to EE along z

_JOINT_LIMITS = np.array([
    [-2.8973,  2.8973],
    [-1.7628,  1.7628],
    [-2.8973,  2.8973],
    [-3.0718, -0.0698],
    [-2.8973,  2.8973],
    [-0.0175,  3.7525],
    [-2.8973,  2.8973],
])


def _dh_mat(a, d, alpha, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,       sa,      ca,       d ],
        [0,        0,       0,       1 ],
    ])


def franka_fk(q: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for i in range(7):
        T = T @ _dh_mat(_DH_A[i], _DH_D[i], _DH_ALPHA[i], q[i])
    T_ee = np.eye(4)
    T_ee[2, 3] = _EE_D
    return T @ T_ee


def franka_ik(target_pose: np.ndarray, q_init: np.ndarray) -> np.ndarray:
    p_tgt = target_pose[:3, 3]
    R_tgt = target_pose[:3, :3]

    def cost(q):
        T = franka_fk(q)
        p_err = np.linalg.norm(T[:3, 3] - p_tgt) ** 2
        R_err = np.linalg.norm(T[:3, :3] - R_tgt, 'fro') ** 2
        return 10.0 * p_err + R_err

    bounds = [(_JOINT_LIMITS[i, 0], _JOINT_LIMITS[i, 1]) for i in range(7)]
    res = minimize(cost, q_init, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
    return res.x


def tilt_R_around_Y(R_in, deg):
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    R_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return R_y @ R_in


def slerp_R(R0, R1, alpha):
    """Rough matrix interpolation via SVD orthogonalization; sufficient for small rotations."""
    M = (1 - alpha) * R0 + alpha * R1
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def interpolate_pose(pose0, pose1, alpha):
    out = np.eye(4)
    out[:3, 3] = (1 - alpha) * pose0[:3, 3] + alpha * pose1[:3, 3]
    out[:3, :3] = slerp_R(pose0[:3, :3], pose1[:3, :3], alpha)
    return out


def stream_joints(client, q, dry_run):
    if not dry_run:
        client.send_joint_position(q, blocking=False)


def main(server_ip: str, dry_run: bool):
    print(f"Connecting to FrankaServer at {server_ip}...")
    client = FrankaClient(server_ip=server_ip)
    client.start()

    if not client.wait_for_state(timeout=5.0):
        print("Failed to receive state from server!")
        client.stop()
        return

    state = client.latest_state
    initial_pose = pose_to_matrix(state.O_T_EE)
    current_q = np.array(state.q)
    print(f"Connected! Initial EE pos: {get_position(state.O_T_EE).round(4)}")
    print(f"Initial joints: {current_q.round(4)}")

    # Start pose: tilt current orientation 15° around base Y, translate to (X=0.4, 0, TARGET_Z)
    target_R = tilt_R_around_Y(initial_pose[:3, :3], TILT_DEG)
    start_pose = np.eye(4)
    start_pose[:3, :3] = target_R
    start_pose[:3, 3] = [TARGET_X, 0.0, TARGET_Z]

    print(f"  start pose target: {start_pose[:3, 3].round(4)}")
    print(f"  Y range: [{-MOVE_RANGE_Y:+.3f}, {+MOVE_RANGE_Y:+.3f}] @ {TARGET_SPEED_Y} m/s")
    print(f"  tilt:    {TILT_DEG}° around base Y")

    if dry_run:
        print("\n[DRY RUN] No commands will be sent.")
    else:
        ans = input("\nConfirm execution? (y/N): ").strip().lower()
        if ans != "y":
            print("Cancelled.")
            client.stop()
            return

    dt = 1.0 / CONTROL_RATE

    # --- Phase 1: Switch mode ---
    print("\n[Phase 1] Setting JOINT_POSITION mode...")
    if not dry_run:
        client.set_control_mode(ControlMode.JOINT_POSITION)
    time.sleep(0.2)

    # --- Phase 2: Slowly interpolate to start pose ---
    print(f"[Phase 2] Approaching start pose over {APPROACH_DURATION:.1f}s...")
    n_approach = int(APPROACH_DURATION * CONTROL_RATE)
    for i in range(n_approach):
        alpha = (i + 1) / n_approach
        p = interpolate_pose(initial_pose, start_pose, alpha)
        current_q = franka_ik(p, current_q)
        stream_joints(client, current_q, dry_run)
        time.sleep(dt)
    time.sleep(0.5)  # settle

    # --- Phase 3: Y-axis reciprocating sweep ---
    print(f"[Phase 3] Y-sweep for {SWEEP_DURATION:.1f}s. Ctrl+C to stop.\n")
    y = 0.0
    direction = 1
    t0 = time.time()
    try:
        while time.time() - t0 < SWEEP_DURATION:
            step_start = time.time()

            y += TARGET_SPEED_Y * direction * dt
            if abs(y) >= MOVE_RANGE_Y:
                y = float(np.clip(y, -MOVE_RANGE_Y, MOVE_RANGE_Y))
                direction *= -1

            target_pose = np.eye(4)
            target_pose[:3, :3] = target_R
            target_pose[:3, 3] = [TARGET_X, y, TARGET_Z]

            cur = client.latest_state
            if cur is not None:
                current_q = np.array(cur.q)
            current_q = franka_ik(target_pose, current_q)
            stream_joints(client, current_q, dry_run)

            if cur is not None:
                cur_pos = get_position(cur.O_T_EE)
                f_ext = np.array(cur.O_F_ext_hat_K[:3])
                pos_err = np.linalg.norm(cur_pos - target_pose[:3, 3])
                print(
                    f"\rt={time.time()-t0:5.1f}s | y_cmd={y:+.3f} | "
                    f"pos_err={pos_err*1000:5.1f}mm | "
                    f"F_ext=[{f_ext[0]:+5.1f},{f_ext[1]:+5.1f},{f_ext[2]:+5.1f}]N",
                    end="",
                )

            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    # --- Phase 4: Lift + return to initial pose ---
    print("\n[Phase 4] Lifting and returning to initial pose...")
    cur = client.latest_state
    if cur is not None:
        cur_pose = pose_to_matrix(cur.O_T_EE)
        current_q = np.array(cur.q)
    else:
        cur_pose = start_pose.copy()

    lift_pose = cur_pose.copy()
    lift_pose[2, 3] += LIFT_HEIGHT

    n_lift = int(1.0 * CONTROL_RATE)
    for i in range(n_lift):
        alpha = (i + 1) / n_lift
        p = interpolate_pose(cur_pose, lift_pose, alpha)
        current_q = franka_ik(p, current_q)
        stream_joints(client, current_q, dry_run)
        time.sleep(dt)

    n_return = int(3.0 * CONTROL_RATE)
    for i in range(n_return):
        alpha = (i + 1) / n_return
        p = interpolate_pose(lift_pose, initial_pose, alpha)
        current_q = franka_ik(p, current_q)
        stream_joints(client, current_q, dry_run)
        time.sleep(dt)

    # --- Cleanup ---
    print("Stopping...")
    if not dry_run:
        client.set_control_mode(ControlMode.IDLE)
    client.stop()
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Franka real-robot Y-sweep with tilt (joint position control)"
    )
    parser.add_argument("--ip", type=str, default="localhost", help="Server IP")
    parser.add_argument("--dry-run", action="store_true", help="Print only, no commands sent")
    args = parser.parse_args()
    main(args.ip, args.dry_run)
