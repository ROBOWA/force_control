#!/usr/bin/env python3
"""Payload gravity-compensation calibration for the FT-sensor tool.

Recovers the lumped tool payload (mass + centre-of-mass) in the
``ft_sensor_site`` frame from FT-sensor readings taken at several different
end-effector orientations (gravity-dominated, no contact), then writes an MJCF
``<include>`` fragment that ``panda_impedance.xml`` loads to update the payload
mass/CoM on its ``ft_sensor_site``.

Why multi-pose: with the tool hanging free, the only wrench the sensor sees is
gravity. Across >=4 distinct orientations the gravity vector sweeps the sensor
frame, so a single least-squares fit recovers mass, CoM, and any residual
sensor bias jointly::

    F_obs = R_site_to_world^T @ [0, 0, -m*g]      - bias_force
    T_obs = com x (R_site_to_world^T @ [0,0,-mg]) - bias_torque

New data path (vs. the old ZMQ/msgpack collector): FT comes from the
shared-memory transport (scripts/ft_node.py -> /dev/shm/ft_wrench, read via
sensors.ft_shm.FTShmSource). The per-pose sensor->world rotation comes from the
SAME MuJoCo model that consumes the result: we set qpos = robot q, run FK, and
read ``data.site_xmat`` for ``ft_sensor_site``. This makes the calibrated CoM
self-consistent with the frame it is written into — no flange/sensor rotation
bookkeeping, and a frame mismatch shows up as a large fit residual.

Workflow (online collection on the control PC):
    # terminal 1 — ROS -> shared-memory bridge (full 400 Hz)
    python -m scripts.ft_node --config configs/real.yaml
    # terminal 2 — collect poses + fit + write fragment
    python -m scripts.calibrate_payload_gravity --collect --config configs/real.yaml

Offline re-fit from a saved raw dump:
    python -m scripts.calibrate_payload_gravity --data data/payload_poses.npz

libfranka allows only ONE process to hold the robot. Stop any controller before
--collect; this script only calls robot.read_once() (no control loop).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sensors.ft_shm import FTShmSource  # noqa: E402

G = 9.81  # m/s^2

# Defaults used when configs/real.yaml has no payload_calibration block.
_DEFAULTS = {
    "model": "franka_emika_panda/panda_impedance.xml",
    "site_name": "ft_sensor_site",
    "fragment_out": "franka_emika_panda/ft_payload.xml",
    "fit_npz": "data/payload_gravity_fit.npz",
    "body_name": "ft_payload",
    "diaginertia": [0.0011, 0.00275, 0.00187],  # carried over (fit gives no inertia)
    "num_poses": 6,
    "samples_per_pose": 300,
    "settle": 2.0,
    "mass_scale": 1.0,
}


# ── Solver ──────────────────────────────────────────────────────────────────

def calibrate_multi_pose(ft_readings: np.ndarray,
                         rotations: np.ndarray,
                         ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Least-squares fit of mass, CoM and residual bias from N gravity poses.

    Args:
        ft_readings: (N, 6) averaged FT readings per pose, no contact.
        rotations:   (N, 3, 3) ft_sensor_site -> world rotations (v_world = R @ v_site).

    Returns:
        mass        (kg), com (3,) sensor frame [m], bias_force (3,) [N],
        bias_torque (3,) [Nm], and a `report` dict with residual RMS values.

    Model (per pose i, g_row_i = R_i[2, :] = world-Z axis in sensor frame):
        F_obs_i = -mg * g_row_i               - bias_force
        T_obs_i =  mg * skew(g_row_i) @ com   - bias_torque
    """
    ft_readings = np.asarray(ft_readings, dtype=float)
    rotations = np.asarray(rotations, dtype=float)
    N = ft_readings.shape[0]
    if N != rotations.shape[0] or ft_readings.shape[1] != 6:
        raise ValueError("ft_readings (N,6) and rotations (N,3,3) must agree on N.")
    if N < 4:
        raise ValueError(f"Need at least 4 poses for the joint-bias fit, got {N}.")

    g_rows = rotations[:, 2, :]                          # (N, 3)

    # 1) Force eqs -> [mg, bias_force]:  F_obs = [-g_row, -I] @ [mg; bias_force]
    A_f = np.zeros((3 * N, 4))
    b_f = np.zeros(3 * N)
    for i in range(N):
        A_f[3 * i:3 * i + 3, 0]   = -g_rows[i]
        A_f[3 * i:3 * i + 3, 1:4] = -np.eye(3)
        b_f[3 * i:3 * i + 3]      = ft_readings[i, :3]
    sol_f, *_ = np.linalg.lstsq(A_f, b_f, rcond=None)
    mg = float(sol_f[0])
    bias_force = sol_f[1:4]
    mass = mg / G

    # 2) Torque eqs -> [com, bias_torque]:  T_obs = [mg*skew(g_row), -I] @ [com; bias_torque]
    A_t = np.zeros((3 * N, 6))
    b_t = np.zeros(3 * N)
    for i in range(N):
        gx, gy, gz = g_rows[i]
        skew = np.array([
            [0.0, -gz,  gy],
            [gz,  0.0, -gx],
            [-gy,  gx, 0.0],
        ])
        A_t[3 * i:3 * i + 3, 0:3] = mg * skew
        A_t[3 * i:3 * i + 3, 3:6] = -np.eye(3)
        b_t[3 * i:3 * i + 3]      = ft_readings[i, 3:]
    sol_t, *_ = np.linalg.lstsq(A_t, b_t, rcond=None)
    com = sol_t[0:3]
    bias_torque = sol_t[3:6]

    # Residuals
    pred_F = -mg * g_rows - bias_force
    res_F = ft_readings[:, :3] - pred_F
    pred_T = np.zeros((N, 3))
    for i in range(N):
        pred_T[i] = np.cross(com, -mg * g_rows[i]) - bias_torque
    res_T = ft_readings[:, 3:] - pred_T
    force_rms = float(np.sqrt((res_F ** 2).mean()))
    torque_rms = float(np.sqrt((res_T ** 2).mean()))

    report = {
        "n_poses": N,
        "force_rms": force_rms,
        "torque_rms": torque_rms,
        "bias_force_norm": float(np.linalg.norm(bias_force)),
        "bias_torque_norm": float(np.linalg.norm(bias_torque)),
    }

    print(f"Estimated mass        : {mass:.4f} kg  (mg = {mg:.4f} N)")
    print(f"Estimated com         : {np.round(com, 5).tolist()} (ft_sensor_site frame, m)")
    print(f"Estimated bias_force  : {np.round(bias_force, 4).tolist()} N "
          f"(||.|| = {report['bias_force_norm']:.3f})")
    print(f"Estimated bias_torque : {np.round(bias_torque, 5).tolist()} Nm "
          f"(||.|| = {report['bias_torque_norm']:.4f})")
    print(f"Force residual RMS    : {force_rms:.4f} N")
    print(f"Torque residual RMS   : {torque_rms:.5f} Nm")
    if mass <= 0:
        print("⚠️  Fitted mass is <= 0 — the FT sign convention is likely flipped. "
              "Set ft.sign: -1.0 in the config (or pass --sign -1) and re-fit.")
    if report["bias_force_norm"] > 0.5:
        print("⚠️  bias_force >= 0.5 N — the FT publisher's tare is non-zero; the fit "
              "absorbed it. Re-tare the sensor (no load) for a cleaner fit.")
    if force_rms > 0.6 or torque_rms > 0.03:
        print("⚠️  Large residual — poses may be too similar, the tool moved/contacted "
              "something, or ft_sensor_site orientation does not match the real sensor.")
    return mass, com, bias_force, bias_torque, report


def compute_gravity_offset(mass: float, com: np.ndarray,
                           R_site_to_world: np.ndarray,
                           bias_force: np.ndarray | None = None,
                           bias_torque: np.ndarray | None = None,
                           ) -> np.ndarray:
    """Predicted gravity FT reading (Fx..Tz) in the ft_sensor_site frame for a
    given pose — i.e. the wrench to subtract from a raw reading to isolate the
    contact wrench. Useful for validating a fit against a live reading."""
    g_world = np.array([0.0, 0.0, -mass * G])
    F = R_site_to_world.T @ g_world
    T = np.cross(np.asarray(com, dtype=float), F)
    w = np.concatenate([F, T])
    if bias_force is not None:
        w[:3] -= np.asarray(bias_force, dtype=float)
    if bias_torque is not None:
        w[3:] -= np.asarray(bias_torque, dtype=float)
    return w


# ── MJCF fragment writer ────────────────────────────────────────────────────

def write_payload_fragment(path: Path, body_name: str, mass: float,
                           com: np.ndarray, diaginertia: np.ndarray,
                           meta: dict) -> None:
    """(Over)write the <mujocoinclude> fragment that panda_impedance.xml loads."""
    com = np.asarray(com, dtype=float)
    di = np.asarray(diaginertia, dtype=float)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_lines = "\n".join(f"           {k:>16}: {v}" for k, v in meta.items())
    xml = f"""<mujocoinclude>
  <!--
    {path.name}  —  AUTO-GENERATED by scripts/calibrate_payload_gravity.py.
    DO NOT EDIT BY HAND; re-run the calibration to regenerate.

    Lumped tool payload (mass + CoM) in the ft_sensor_site frame. <include>d by
    panda_impedance.xml inside <body name="ft_frame"> after the site, so this
    body frame IS the ft_sensor_site frame and the CoM drops straight into pos.
    The FT sensor measures only the tool BELOW it — the 0.242 kg sensor body is
    modelled separately and must NOT be added here. Rotational inertia is not
    measured by the gravity fit (irrelevant to RNE); diaginertia is carried over.

    Fit metadata ({stamp}):
{meta_lines}
  -->
  <body name="{body_name}" pos="0 0 0">
    <inertial mass="{mass:.6f}" pos="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}"
              diaginertia="{di[0]:.6g} {di[1]:.6g} {di[2]:.6g}"/>
  </body>
</mujocoinclude>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml)
    print(f"\n✓ Wrote payload fragment -> {path}")
    print("  panda_impedance.xml <include>s this; the model now uses the new mass/CoM.")


def _finalize(ft_arr: np.ndarray, R_arr: np.ndarray, cfg: dict) -> None:
    """Solve, apply mass scale, and write the fragment + audit npz."""
    print("\n=== Solving multi-pose least squares ===")
    mass, com, bias_force, bias_torque, report = calibrate_multi_pose(ft_arr, R_arr)

    scale = float(cfg["mass_scale"])
    mass_out = mass * scale
    if scale != 1.0:
        print(f"\nApplying mass_scale={scale}: {mass:.4f} kg -> {mass_out:.4f} kg "
              "(empirical uplift for unmodelled cable/screw mass).")

    meta = {
        "n_poses": report["n_poses"],
        "mass_raw_kg": round(mass, 5),
        "mass_scale": scale,
        "mass_kg": round(mass_out, 5),
        "com_m": [round(float(c), 6) for c in com],
        "force_rms_N": round(report["force_rms"], 4),
        "torque_rms_Nm": round(report["torque_rms"], 5),
        "bias_force_N": [round(float(b), 4) for b in bias_force],
        "bias_torque_Nm": [round(float(b), 5) for b in bias_torque],
    }

    write_payload_fragment(
        Path(cfg["fragment_out"]), cfg["body_name"], mass_out, com,
        cfg["diaginertia"], meta,
    )

    fit_npz = Path(cfg["fit_npz"])
    fit_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        fit_npz, mass=mass_out, mass_raw=mass, mass_scale=scale, com=com,
        diaginertia=np.asarray(cfg["diaginertia"], dtype=float),
        bias_force=bias_force, bias_torque=bias_torque,
        ft=ft_arr, rotations=R_arr,
    )
    print(f"✓ Wrote fit/audit data  -> {fit_npz}  (mass, com, bias, raw ft+rotations)")


# ── Online collection (control PC) ──────────────────────────────────────────

def _collect_ft_mean(ft_source: FTShmSource, num_samples: int, sign: float,
                     timeout: float = 5.0) -> tuple[np.ndarray, np.ndarray] | None:
    """Average `num_samples` DISTINCT shared-memory FT samples (by seq).

    Returns (mean (6,), std (6,)) with `sign` applied, or None if the writer
    stalls (no new sample within `timeout`)."""
    samples: list[np.ndarray] = []
    last_seq = None
    t_last = time.perf_counter()
    while len(samples) < num_samples:
        s = ft_source.get_latest()
        if s.valid and s.seq != last_seq:
            samples.append(np.asarray(s.wrench, dtype=float) * sign)
            last_seq = s.seq
            t_last = time.perf_counter()
        elif time.perf_counter() - t_last > timeout:
            print("    ⚠️  FT shared memory stalled — is scripts.ft_node running?")
            return None
        else:
            time.sleep(0.001)
    arr = np.asarray(samples, dtype=float)
    return arr.mean(axis=0), arr.std(axis=0)


def _capture_pose(ft_source, robot, model, data, site_id,
                  num_samples: int, settle: float, sign: float
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Settle, average FT (shm), read q (libfranka), FK -> ft_sensor_site rotation."""
    import mujoco  # noqa: PLC0415

    print(f"  Settling for {settle:.1f}s ...")
    time.sleep(settle)

    print(f"  Averaging {num_samples} FT samples ...")
    res = _collect_ft_mean(ft_source, num_samples, sign)
    if res is None:
        return None
    ft_mean, ft_std = res

    try:
        state = robot.read_once()
    except Exception as e:
        print(f"    ⚠️  robot.read_once() failed: {e}")
        return None
    q = np.asarray(state.q, dtype=float)

    data.qpos[:7] = q
    data.qvel[:] = 0.0
    mujoco.mj_kinematics(model, data)
    R_site_to_world = data.site_xmat[site_id].reshape(3, 3).copy()

    print(f"  ft_mean = {np.round(ft_mean, 3).tolist()}")
    print(f"  ft_std  = {np.round(ft_std, 4).tolist()}  (large std => motion/vibration)")
    print(f"  R_site_to_world =\n{np.round(R_site_to_world, 3)}")
    return ft_mean, R_site_to_world, ft_std


def collect_online(robot_ip: str, shm_path: str, sign: float, cfg: dict,
                   raw_output: Path | None) -> None:
    """Interactive online collection: guide the arm by hand between captures."""
    import mujoco  # noqa: PLC0415
    from pylibfranka import Robot  # noqa: PLC0415

    print("=== FT Payload Gravity Calibration (online, shared-memory FT) ===")
    print(f"Robot (libfranka direct) : {robot_ip}")
    print(f"FT shared memory         : {shm_path}")
    print(f"Model (FK frame source)  : {cfg['model']}  site={cfg['site_name']}")
    print(f"Num poses / samples      : {cfg['num_poses']} / {cfg['samples_per_pose']}")
    print()
    print("Before you start:")
    print("  • scripts.ft_node must be feeding the shared-memory buffer.")
    print("  • No other process may hold the robot (one libfranka connection only).")
    print("  • Hold the white guiding button on the wrist to move the arm by hand.")
    print("  • Each pose: tool hangs free, NO contact; vary orientation between poses.\n")

    model = mujoco.MjModel.from_xml_path(cfg["model"])
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg["site_name"])
    if site_id < 0:
        print(f"❌ site '{cfg['site_name']}' not found in {cfg['model']}.")
        return

    ft_source = FTShmSource(shm_path=shm_path)
    try:
        ft_source.start()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    # Wait for the first valid sample so we fail fast if the bridge is down.
    t0 = time.perf_counter()
    while not ft_source.get_latest().valid:
        if time.perf_counter() - t0 > 3.0:
            print("❌ No FT data in shared memory. Start: python -m scripts.ft_node")
            ft_source.stop()
            return
        time.sleep(0.05)

    try:
        robot = Robot(robot_ip)
    except Exception as e:
        print(f"❌ Failed to connect to robot at {robot_ip}: {e}")
        ft_source.stop()
        return
    print(f"✓ Connected to robot at {robot_ip}\n")

    ft_list: list[np.ndarray] = []
    R_list: list[np.ndarray] = []
    num_poses = int(cfg["num_poses"])
    try:
        i = 0
        while i < num_poses:
            print(f"--- Pose {i + 1}/{num_poses} ---")
            print("Move the arm to a NEW orientation; tool free, no contact.")
            cmd = input("Enter = capture / s = skip / r = redo last / q = finish & fit: ").strip().lower()
            if cmd == "q":
                break
            if cmd == "s":
                continue
            if cmd == "r":
                if ft_list:
                    ft_list.pop(); R_list.pop(); i -= 1
                    print("Dropped last capture.")
                else:
                    print("Nothing captured yet.")
                continue

            result = _capture_pose(ft_source, robot, model, data, site_id,
                                   int(cfg["samples_per_pose"]), float(cfg["settle"]), sign)
            if result is None:
                print("  ✗ Capture failed; skipping.")
                continue
            ft_mean, R, ft_std = result
            if float(np.max(ft_std)) > 0.5:
                ans = input("  ⚠️  FT std high (>0.5) — keep this pose? [y/N] ").strip().lower()
                if ans != "y":
                    print("  Dropped.")
                    continue
            ft_list.append(ft_mean); R_list.append(R); i += 1
            print(f"  ✓ Recorded ({i}/{num_poses})\n")
    finally:
        ft_source.stop()
        del robot  # pylibfranka.Robot has no explicit close

    if len(ft_list) < 4:
        print(f"\n❌ Only {len(ft_list)} poses collected; need at least 4.")
        return

    ft_arr = np.stack(ft_list, axis=0)
    R_arr = np.stack(R_list, axis=0)
    if raw_output is not None:
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(raw_output, ft=ft_arr, rotations=R_arr)
        print(f"\nRaw poses saved -> {raw_output}  (re-fit with --data {raw_output})")

    _finalize(ft_arr, R_arr, cfg)


# ── CLI ─────────────────────────────────────────────────────────────────────

def _build_cfg(args) -> tuple[dict, str, str, float]:
    """Merge config file + payload_calibration block + CLI overrides."""
    full = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        full = yaml.safe_load(cfg_path.read_text()) or {}
    else:
        print(f"(config {cfg_path} not found — using defaults)")

    pc = dict(_DEFAULTS)
    pc.update(full.get("payload_calibration", {}) or {})

    # CLI overrides
    if args.model:            pc["model"] = args.model
    if args.site_name:        pc["site_name"] = args.site_name
    if args.output:           pc["fragment_out"] = args.output
    if args.fit_npz:          pc["fit_npz"] = args.fit_npz
    if args.num_poses:        pc["num_poses"] = args.num_poses
    if args.samples_per_pose: pc["samples_per_pose"] = args.samples_per_pose
    if args.settle is not None:     pc["settle"] = args.settle
    if args.mass_scale is not None: pc["mass_scale"] = args.mass_scale
    if args.diaginertia:      pc["diaginertia"] = args.diaginertia

    # Resolve relative paths against the project root.
    for key in ("model", "fragment_out", "fit_npz"):
        p = Path(pc[key])
        pc[key] = str(p if p.is_absolute() else PROJECT_ROOT / p)

    ft = full.get("ft", {}) or {}
    robot_ip = args.robot_ip or full.get("robot_ip", "172.16.0.2")
    shm_path = args.shm_path or ft.get("shm_path", "/dev/shm/ft_wrench")
    sign = args.sign if args.sign is not None else float(ft.get("sign", 1.0))
    return pc, robot_ip, shm_path, sign


def main() -> None:
    p = argparse.ArgumentParser(
        description="Payload gravity calibration -> MJCF include fragment for "
                    "panda_impedance.xml (ft_sensor_site mass + CoM).")
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "real.yaml"))

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--collect", action="store_true",
                      help="Interactive online collection (needs robot + ft_node).")
    mode.add_argument("--data", default=None,
                      help="Offline re-fit from .npz with 'ft' (N,6) and 'rotations' (N,3,3).")

    # Online
    p.add_argument("--robot-ip", default=None, help="Robot IP (default config robot_ip).")
    p.add_argument("--shm-path", default=None, help="FT shared-memory path (default ft.shm_path).")
    p.add_argument("--sign", type=float, default=None, help="FT sign multiplier (default ft.sign).")
    p.add_argument("--num-poses", type=int, default=None, help="Poses to collect (>=4).")
    p.add_argument("--samples-per-pose", type=int, default=None, help="FT frames averaged per pose.")
    p.add_argument("--settle", type=float, default=None, help="Settle time per pose [s].")
    p.add_argument("--raw-output", default=None, help="Optional .npz dump of raw (ft, rotations).")

    # Shared
    p.add_argument("--model", default=None, help="MJCF whose site defines the fit frame.")
    p.add_argument("--site-name", default=None, help="Site name (default ft_sensor_site).")
    p.add_argument("--output", default=None, help="Output fragment .xml (default franka_emika_panda/ft_payload.xml).")
    p.add_argument("--fit-npz", default=None, help="Output audit .npz path.")
    p.add_argument("--mass-scale", type=float, default=None,
                   help="Multiply fitted mass before writing (empirical uplift; default 1.0).")
    p.add_argument("--diaginertia", type=float, nargs=3, default=None,
                   help="Override carried-over diaginertia (ixx iyy izz).")
    args = p.parse_args()

    cfg, robot_ip, shm_path, sign = _build_cfg(args)

    if args.collect:
        if int(cfg["num_poses"]) < 4:
            raise SystemExit("--num-poses must be >= 4 for the joint-bias fit.")
        raw_out = Path(args.raw_output) if args.raw_output else None
        collect_online(robot_ip, shm_path, sign, cfg, raw_out)
    else:
        data = np.load(args.data)
        if "ft" not in data or "rotations" not in data:
            raise SystemExit("Input .npz must contain 'ft' (N,6) and 'rotations' (N,3,3).")
        _finalize(np.asarray(data["ft"]), np.asarray(data["rotations"]), cfg)


if __name__ == "__main__":
    main()
