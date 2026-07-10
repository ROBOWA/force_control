"""Flange F_ext drift test: move between IK waypoints, log wrench per phase.

Purpose
-------
Verify WHERE the flange force reading drifts:
  * pose-dependent bias   -> the STATIC reading (HOLD_i) differs between poses
  * motion-induced offset -> the reading during MOVE segments is offset from the
    surrounding holds (and flips sign with direction: down-move vs up-move)

The test replicates the normal pipeline exactly:
  * waypoints are defined the same way as the initial pose (ik.target_pos +
    target_euler_xyz semantics, keyframe orientation reference) and solved with
    the same DLS IK;
  * motion uses the same min-jerk + JointPDController (MOVE_JOINT-style);
  * the same FlangeFTProcessor (sign / butterworth) and payload-gravity
    feed-forward are active;
  * a TARE is performed at the FIRST waypoint, like the state machine does.

Sequence:  MOVE->W0, TARE@W0, HOLD@W0, MOVE W0->W1, HOLD@W1, ...  ping-pong
through the waypoint list `cycles` times, then a final hold and clean finish.

Every 1 kHz tick logs: t, q(7), tip xyz, RAW O_F_ext_hat_K (6), processed
wrench (6), phase label.  Keep ALL waypoints contact-free — this is a free-air
test; any contact invalidates the bias measurement.

Run from the repo root (Ubuntu Panda PC):

    # dry run: solve IK, print waypoints, no motion
    python -m scripts.test_flange_ft_drift --config configs/real_flange.yaml --dry-run

    # execute (asks for confirmation)
    python -m scripts.test_flange_ft_drift --config configs/real_flange.yaml --execute

    # offline analysis of a recorded CSV (runs anywhere)
    python -m scripts.test_flange_ft_drift --analyze data/logs/ft_drift_test_XXXX.csv

Waypoints / durations are read from the `ft_pose_test` section of the config.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.controller import JointPDController                      # noqa: E402
from core.ft_processor_flange import FlangeFTProcessor             # noqa: E402
from core.ik import euler_xyz_to_R, min_jerk, solve_ik_dls         # noqa: E402
from core.kinematics import SiteKinematics                         # noqa: E402
from core.payload_gravity import PayloadGravityCompensator         # noqa: E402
from core.safety import saturate_torque_rate                       # noqa: E402
from sensors.ft_flange import FTFlangeSource                       # noqa: E402


# ── Segment plan ─────────────────────────────────────────────────────────────

class Segment:
    """One phase of the test: a min-jerk MOVE, or a stationary HOLD/TARE."""

    def __init__(self, name: str, kind: str, duration: float,
                 q_from: np.ndarray, q_to: np.ndarray):
        self.name = name
        self.kind = kind            # "move" | "hold" | "tare"
        self.duration = float(duration)
        self.q_from = q_from.copy()
        self.q_to = q_to.copy()

    def q_des(self, t_seg: float) -> tuple[np.ndarray, np.ndarray]:
        if self.kind != "move":
            return self.q_to.copy(), np.zeros(7)
        s, ds = min_jerk(t_seg, self.duration)
        return (self.q_from + s * (self.q_to - self.q_from),
                ds * (self.q_to - self.q_from))


def build_plan(q_start: np.ndarray, q_wps: list[np.ndarray],
               move_time: float, hold_time: float, tare_time: float,
               cycles: int) -> list[Segment]:
    """MOVE->W0, TARE@W0, HOLD@W0, then ping-pong the list `cycles` times."""
    plan = [Segment("MOVE_START>W0", "move", move_time, q_start, q_wps[0]),
            Segment("TARE@W0", "tare", tare_time, q_wps[0], q_wps[0]),
            Segment("HOLD@W0_c0", "hold", hold_time, q_wps[0], q_wps[0])]

    order = list(range(len(q_wps)))
    for c in range(cycles):
        # forward W0->..->Wn, then backward ..->W0 (skip repeating endpoints)
        path = order[1:] + order[-2::-1] if len(q_wps) > 1 else []
        prev = 0
        for k in path:
            plan.append(Segment(f"MOVE_W{prev}>W{k}_c{c}", "move",
                                move_time, q_wps[prev], q_wps[k]))
            plan.append(Segment(f"HOLD@W{k}_c{c}", "hold",
                                hold_time, q_wps[k], q_wps[k]))
            prev = k
    plan.append(Segment("FINAL_HOLD", "hold", hold_time,
                        plan[-1].q_to, plan[-1].q_to))
    return plan


# ── Preallocated per-tick logger ─────────────────────────────────────────────

class DriftLogger:
    COLUMNS = (["t"] + [f"q{i+1}" for i in range(7)] + ["x", "y", "z"]
               + [f"raw_{a}" for a in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")]
               + ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"] + ["phase"])

    def __init__(self, capacity: int, phase_names: list[str]):
        self._buf = np.zeros((capacity, 23), dtype=np.float64)
        self._code = np.zeros(capacity, dtype=np.int32)
        self._names = phase_names
        self._n = 0
        self._cap = capacity
        self._overflowed = False

    def log(self, t, q, x, raw_w, proc_w, code) -> None:
        i = self._n
        if i >= self._cap:
            if not self._overflowed:
                print("⚠️  DriftLogger full — logging stopped.")
                self._overflowed = True
            return
        b = self._buf[i]
        b[0] = t
        b[1:8] = q
        b[8:11] = x
        b[11:17] = raw_w
        b[17:23] = proc_w
        self._code[i] = code
        self._n = i + 1

    def save(self, output_dir: str, prefix: str) -> str | None:
        if self._n == 0:
            print("DriftLogger: nothing recorded.")
            return None
        os.makedirs(output_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        path = os.path.join(output_dir, f"{prefix}_{stamp}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.COLUMNS)
            for i in range(self._n):
                row = [f"{v:.6f}" for v in self._buf[i]]
                row.append(self._names[int(self._code[i])])
                w.writerow(row)
        print(f"DriftLogger: wrote {self._n} rows -> {path}")
        return path


# ── IK for the waypoint list (same semantics as the state machine) ──────────

def solve_waypoints(cfg: dict, wp_cfg: list[dict], q_seed: np.ndarray):
    """Solve IK for each waypoint; returns (q_list, model, site_name)."""
    import mujoco

    ik_cfg = cfg["ik"]
    model = mujoco.MjModel.from_xml_path(ik_cfg["mjcf"])
    data = mujoco.MjData(model)
    site_name = ik_cfg["site_name"]
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"site '{site_name}' not found in {ik_cfg['mjcf']}")

    # Reference orientation from the home keyframe (state-machine default).
    scratch = mujoco.MjData(model)
    if model.nkey > 0:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, scratch, max(key_id, 0))
    mujoco.mj_forward(model, scratch)
    R_ref = scratch.site_xmat[site_id].copy().reshape(3, 3)

    q_list = []
    q_init = q_seed.copy()
    for i, wp in enumerate(wp_cfg):
        target_pos = np.array(wp["pos"], dtype=float)
        target_R = R_ref @ euler_xyz_to_R(wp.get("euler", [0.0, 0.0, 0.0]))
        data.qpos[:7] = q_init
        mujoco.mj_forward(model, data)
        q, n_iter, err, ok = solve_ik_dls(model, data, site_id, q_init,
                                          target_pos, target_R)
        print(f"  W{i}: pos={target_pos.tolist()} -> IK "
              f"{'OK' if ok else 'FAILED'} (iter={n_iter}, err={err:.5f})  "
              f"q={np.round(q, 3).tolist()}")
        if not ok:
            raise RuntimeError(f"IK failed for waypoint {i}")
        q_list.append(q.copy())
        q_init = q.copy()      # seed the next solve for branch consistency
    return q_list, model, site_name


# ── Test runner ──────────────────────────────────────────────────────────────

def run_test(cfg: dict, dry_run: bool) -> None:
    tp = cfg.get("ft_pose_test", {})
    wp_cfg = tp.get("waypoints")
    if not wp_cfg:
        raise ValueError("config has no ft_pose_test.waypoints")
    move_time = float(tp.get("move_time", 6.0))
    hold_time = float(tp.get("hold_time", 4.0))
    cycles = int(tp.get("cycles", 2))
    tare_time = float(cfg.get("ft", {}).get("tare_duration", 5.0))

    from pylibfranka import Robot, Torques

    print(f"Connecting to Franka at {cfg['robot_ip']} …")
    robot = Robot(cfg["robot_ip"])
    robot.set_collision_behavior(
        [40.0, 40.0, 38.0, 38.0, 30.0, 25.0, 20.0],
        [40.0, 40.0, 38.0, 38.0, 30.0, 25.0, 20.0],
        [40.0, 40.0, 40.0, 50.0, 50.0, 50.0],
        [40.0, 40.0, 40.0, 50.0, 50.0, 50.0],
    )
    state0 = robot.read_once()
    q0 = np.asarray(state0.q, dtype=np.float64)
    print(f"Initial q: {q0.round(4)}")

    print("Solving IK for waypoints:")
    q_wps, ik_model, site_name = solve_waypoints(cfg, wp_cfg, q0)

    plan = build_plan(q0, q_wps, move_time, hold_time, tare_time, cycles)
    total = sum(s.duration for s in plan)
    print(f"\nPlan ({len(plan)} segments, ~{total:.0f}s total):")
    for s in plan:
        print(f"  {s.name:<18} {s.kind:<5} {s.duration:.1f}s")

    if dry_run:
        print("[DRY RUN] IK + plan OK. No motion commanded.")
        return
    ans = input("\n⚠️  FREE-AIR test — confirm all waypoints are contact-free. "
                "Execute? (y/N): ").strip().lower()
    if ans != "y":
        print("Cancelled.")
        return

    # Same processing chain as the real pipeline.
    ft_cfg = cfg.get("ft", {})
    filt = ft_cfg.get("filter", {})
    ft_source = FTFlangeSource(frame=str(ft_cfg.get("frame", "base")))
    ft_proc = FlangeFTProcessor(
        sign=ft_cfg.get("sign", 1.0),
        filter_type=str(filt.get("type", "butterworth")),
        cutoff_hz=float(filt.get("cutoff_hz", 20.0)),
        sample_rate_hz=float(filt.get("sample_rate_hz", 1000.0)),
        ewa_alpha=float(filt.get("alpha", 0.1)),
        despike_window=int(filt.get("despike_window", 1)),
    )
    joint_pd = JointPDController(cfg["joint_pd"])
    tip_kin = SiteKinematics(ik_model, site_name)

    payload_comp = None
    if cfg.get("payload_gravity", {}).get("enabled", False):
        payload_comp = PayloadGravityCompensator(cfg)
        print("Payload gravity compensation enabled (same as pipeline).")

    safety = cfg.get("safety", {})
    max_torque = np.array(safety.get(
        "max_torque", [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]))
    max_rate = float(safety.get("max_torque_rate", 1.0))

    capacity = int(total * 1000 * 1.2) + 20_000
    logger = DriftLogger(capacity, [s.name for s in plan])

    # Callback state (closure)
    st = {"t": 0.0, "seg": 0, "seg_t0": 0.0, "tau_prev": np.zeros(7),
          "tare_sum": np.zeros(6), "tare_n": 0, "done": False}

    def callback(robot_state, duration):
        dt = duration.to_sec() if hasattr(duration, "to_sec") else 1e-3
        st["t"] += dt
        t = st["t"]

        q = np.asarray(robot_state.q, dtype=np.float64)
        dq = np.asarray(robot_state.dq, dtype=np.float64)

        raw = ft_source.update(robot_state, t=t).wrench
        proc = ft_proc.process(raw)

        # Segment bookkeeping
        seg = plan[st["seg"]]
        t_seg = t - st["seg_t0"]
        if seg.kind == "tare":
            st["tare_sum"] += raw
            st["tare_n"] += 1
        if t_seg >= seg.duration:
            if seg.kind == "tare" and st["tare_n"] > 0:
                ft_proc.tare(st["tare_sum"] / st["tare_n"])
                print(f"\nTared over {st['tare_n']} samples.")
            if st["seg"] + 1 < len(plan):
                st["seg"] += 1
                st["seg_t0"] = t
                seg = plan[st["seg"]]
                t_seg = 0.0
                print(f"→ {seg.name}")
            else:
                st["done"] = True

        q_des, dq_des = seg.q_des(t_seg)
        tau = joint_pd.compute(q, dq, q_des, dq_des)
        if payload_comp is not None:
            tau += payload_comp.compute(q)
        tau = np.clip(tau, -max_torque, max_torque)
        tau = saturate_torque_rate(tau, st["tau_prev"], max_rate)
        st["tau_prev"] = tau.copy()

        x_tip, _, _, _, _ = tip_kin.compute(q, dq)
        logger.log(t, q, x_tip, raw, proc, st["seg"])

        if st["done"]:
            return Torques.finished(tau.tolist())
        return Torques(tau.tolist())

    print("Torque control active — Ctrl+C aborts (log is saved either way).")
    try:
        robot.control_torques(callback)
        print("Sequence complete.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        path = logger.save(tp.get("output_dir", "data/logs"),
                           tp.get("prefix", "ft_drift_test"))
        if path:
            print(f"Analyze with:\n  python -m scripts.test_flange_ft_drift "
                  f"--analyze {path}")


# ── Offline analysis ─────────────────────────────────────────────────────────

def analyze(path: str) -> None:
    """Per-phase Fz stats: static pose bias vs motion-induced offset."""
    rows_t, rows_z, rows_raw, rows_fz, rows_ph = [], [], [], [], []
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        i_z, i_raw, i_fz, i_ph = (header.index("z"), header.index("raw_Fz"),
                                  header.index("Fz"), header.index("phase"))
        for row in r:
            rows_t.append(float(row[0]))
            rows_z.append(float(row[i_z]))
            rows_raw.append(float(row[i_raw]))
            rows_fz.append(float(row[i_fz]))
            rows_ph.append(row[i_ph])
    t = np.array(rows_t)
    z = np.array(rows_z)
    raw = np.array(rows_raw)
    fz = np.array(rows_fz)
    ph = np.array(rows_ph)

    # Contiguous phase runs, in order.
    runs = []
    start = 0
    for i in range(1, len(ph) + 1):
        if i == len(ph) or ph[i] != ph[start]:
            runs.append((ph[start], start, i))
            start = i

    # Per-run means (runs are contiguous segments in time order; names can
    # repeat across ping-pong passes, so key everything by run index).
    print(f"\n{'phase':<20} {'dur[s]':>7} {'z range [m]':>18} "
          f"{'Fz mean±std [N]':>18} {'raw Fz mean':>12}")
    means = []
    for name, a, b in runs:
        # settle: use the last 60% of holds, middle 60% of moves
        n = b - a
        if name.startswith("MOVE"):
            s = slice(a + int(0.2 * n), b - int(0.2 * n))
        else:
            s = slice(a + int(0.4 * n), b)
        m, sd = float(fz[s].mean()), float(fz[s].std())
        means.append(m)
        print(f"{name:<20} {t[b-1]-t[a]:>7.1f} "
              f"[{z[a]:+.3f} → {z[b-1]:+.3f}] {m:>+10.2f} ± {sd:<5.2f} "
              f"{raw[s].mean():>+12.2f}")

    hold_vals = [means[i] for i, (n, _, _) in enumerate(runs)
                 if n.startswith("HOLD") or n == "FINAL_HOLD"]
    if hold_vals:
        print(f"\n静态位姿间偏置 (pose-dependent bias): HOLD 段 Fz 范围 "
              f"{min(hold_vals):+.2f} … {max(hold_vals):+.2f} N "
              f"(峰峰 {max(hold_vals) - min(hold_vals):.2f} N)")

    # Motion offset = move mean minus the mean of its bracketing static runs.
    offs = []
    for i, (n, _, _) in enumerate(runs):
        if not n.startswith("MOVE") or "START" in n:
            continue
        neigh = [means[j] for j in (i - 1, i + 1)
                 if 0 <= j < len(runs) and not runs[j][0].startswith("MOVE")]
        if neigh:
            offs.append((n, means[i] - float(np.mean(neigh))))
    if offs:
        print("运动段相对前后静止段的偏移 (motion-induced offset):")
        for n, o in offs:
            print(f"  {n:<20} {o:+.2f} N")
        dn = [o for n, o in offs if _is_down(n)]
        up = [o for n, o in offs if not _is_down(n)]
        if dn and up:
            print(f"  下行均值 {np.mean(dn):+.2f} N vs 上行均值 {np.mean(up):+.2f} N")
            if np.sign(np.mean(dn)) != np.sign(np.mean(up)):
                print("  → 偏移随运动方向反号：符合摩擦/指令力矩(τ_d 被扣除)机制")


def _is_down(name: str) -> bool:
    """MOVE_Wa>Wb: heuristic — larger waypoint index = later in list = lower z."""
    try:
        core = name.split("_", 1)[1].split(">")
        a = int(core[0].lstrip("W"))
        b = int(core[1].split("_")[0].lstrip("W"))
        return b > a
    except (IndexError, ValueError):
        return False


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Flange F_ext pose/motion drift test")
    p.add_argument("--config", default="configs/real_flange.yaml")
    p.add_argument("--dry-run", action="store_true",
                   help="solve IK and print the plan; no motion")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--analyze", metavar="CSV",
                   help="analyze a recorded CSV instead of running the robot")
    args = p.parse_args()

    if args.analyze:
        analyze(args.analyze)
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_test(cfg, dry_run=args.dry_run or not args.execute)


if __name__ == "__main__":
    main()
