#!/usr/bin/env python3
"""Gravity-compensation 'float test' on the real Franka.

Commands ONLY the payload gravity-compensation torque — no joint PD, no target,
no state machine — so the arm should hang in free-float equilibrium:

    tau_cmd = G_full(q) - G_zero(q)          (PayloadGravityCompensator.compute)

libfranka already cancels the arm's own gravity internally; this delta cancels
the extra tool payload (the calibrated ft_payload + the 0.242 kg sensor body)
that libfranka does not know about. If the payload model (mass + CoM from
scripts/calibrate_payload_gravity.py) is correct, the net static torque is ~0:
the arm holds its pose and is freely backdrivable. If the payload is wrong, the
uncompensated tool weight pulls the arm down — the joints (typically J2 / J4)
SAG, and the drift below grows over time.

How to read the result:
    • small, settling drift (<~1-2 deg/joint, tip <~10 mm)  -> good compensation
    • large or steadily growing drift, esp. J2 / J4 dropping -> under-compensated
          (raise payload mass / mass_scale, or re-check the calibrated CoM)
    • push the arm by hand: it should feel weightless and stay where you leave it

Run TWICE to quantify the benefit:
    python -m scripts.test_gravity_comp --config configs/real.yaml              # comp ON
    python -m scripts.test_gravity_comp --config configs/real.yaml --no-comp    # baseline (tau=0)

The --no-comp baseline commands zero torque, so the tool WILL droop under its
own weight — keep a hand near the arm and be ready to Ctrl+C / E-stop.

This script only runs torque control; stop any other process holding the robot
first (one libfranka connection at a time).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.payload_gravity import PayloadGravityCompensator  # noqa: E402
from core.safety import saturate_torque_rate                # noqa: E402

# Conservative collision thresholds (same as FrankaBackend.connect).
_COLLISION_TORQUE = [40.0, 40.0, 38.0, 38.0, 30.0, 25.0, 20.0]
_COLLISION_FORCE = [40.0, 40.0, 40.0, 50.0, 50.0, 50.0]


class GravityCompTest:
    """Minimal torque loop that floats the arm on payload gravity comp only."""

    def __init__(self, cfg: dict, seconds: float, use_comp: bool):
        self._cfg = cfg
        self._seconds = float(seconds)
        self._use_comp = use_comp

        safety = cfg.get("safety", {})
        self._max_torque = np.array(
            safety.get("max_torque", [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]),
            dtype=float,
        )
        self._max_torque_rate = float(safety.get("max_torque_rate", 1.0))

        self._comp = PayloadGravityCompensator(cfg) if use_comp else None

        # Preallocated log (single writer = RT callback; no per-tick alloc).
        cap = int(self._seconds * 1100) + 200
        self._t = np.zeros(cap)
        self._q = np.zeros((cap, 7))
        self._dq = np.zeros((cap, 7))
        self._tau = np.zeros((cap, 7))
        self._n = 0

        self._tau_prev = np.zeros(7)
        self._t_ctrl = 0.0
        self._last_print = 0.0
        self._Torques = None

    # ------------------------------------------------------------------
    def _callback(self, robot_state, duration):
        q = np.asarray(robot_state.q, dtype=float)
        dq = np.asarray(robot_state.dq, dtype=float)
        dt = duration.to_sec() if hasattr(duration, "to_sec") else 1e-3
        self._t_ctrl += dt

        tau = self._comp.compute(q) if self._comp is not None else np.zeros(7)

        # Safety: clip, then rate-limit (ramps from 0 at start).
        tau = np.clip(tau, -self._max_torque, self._max_torque)
        tau = saturate_torque_rate(tau, self._tau_prev, self._max_torque_rate)
        self._tau_prev = tau.copy()

        # Record.
        i = self._n
        if i < self._t.shape[0]:
            self._t[i] = self._t_ctrl
            self._q[i] = q
            self._dq[i] = dq
            self._tau[i] = tau
            self._n += 1

        # ~2 Hz live drift heartbeat.
        if self._t_ctrl - self._last_print >= 0.5 and self._n > 1:
            self._last_print = self._t_ctrl
            drift = np.degrees(q - self._q[0])
            j = int(np.argmax(np.abs(drift)))
            print(f"  t={self._t_ctrl:5.1f}s  max drift J{j+1}={drift[j]:+6.2f} deg  "
                  f"|dq|max={np.max(np.abs(dq)):.3f} rad/s")

        if self._t_ctrl >= self._seconds:
            return self._Torques.finished(tau.tolist())
        return self._Torques(tau.tolist())

    # ------------------------------------------------------------------
    def run(self, robot_ip: str) -> None:
        from pylibfranka import Robot, Torques  # noqa: PLC0415

        self._Torques = Torques
        print(f"Connecting to Franka at {robot_ip} ...")
        robot = Robot(robot_ip)
        robot.set_collision_behavior(
            _COLLISION_TORQUE, _COLLISION_TORQUE, _COLLISION_FORCE, _COLLISION_FORCE)
        q0 = np.asarray(robot.read_once().q, dtype=float)
        print(f"Initial q [deg]: {np.degrees(q0).round(2).tolist()}")

        if self._comp is not None:
            tau0 = self._comp.compute(q0)
            print(f"Payload comp torque @start [Nm]: {tau0.round(3).tolist()}")
        else:
            print("Comp DISABLED (--no-comp): commanding zero torque — arm will droop.")

        mode = "PAYLOAD GRAVITY COMP" if self._comp is not None else "ZERO TORQUE (baseline)"
        print(f"\n>>> Floating for {self._seconds:.0f}s in {mode} mode. Ctrl+C to stop early.\n")
        try:
            robot.control_torques(self._callback)
        except KeyboardInterrupt:
            print("\nInterrupted.")
        del robot
        self._report()

    # ------------------------------------------------------------------
    def _report(self) -> None:
        n = self._n
        if n < 2:
            print("No data recorded.")
            return
        t = self._t[:n]
        q = self._q[:n]
        dq = self._dq[:n]
        tau = self._tau[:n]
        dur = t[-1] - t[0]

        drift = q - q[0]                          # (n,7) rad
        drift_deg = np.degrees(drift)
        final = drift_deg[-1]
        max_abs = np.max(np.abs(drift_deg), axis=0)
        worst_j = int(np.argmax(max_abs))
        rms = float(np.sqrt((final ** 2).mean()))

        print("\n" + "=" * 64)
        print(f"GRAVITY-COMP FLOAT TEST  ({'comp ON' if self._comp else 'NO comp'})")
        print("=" * 64)
        print(f"duration {dur:.1f}s   ticks {n}   rate {n / dur:.0f} Hz")
        print("\nPer-joint drift from start [deg]   (final / peak |drift|):")
        for j in range(7):
            flag = "  <-- worst" if j == worst_j else ""
            print(f"  J{j+1}: final {final[j]:+7.2f}   peak {max_abs[j]:6.2f}{flag}")
        print(f"\nFinal drift RMS over joints : {rms:.2f} deg")
        print(f"Worst joint                 : J{worst_j+1} "
              f"(peak {max_abs[worst_j]:.2f} deg)")
        print(f"Peak joint speed |dq|max    : {np.max(np.abs(dq)):.3f} rad/s")

        # Settling vs growing: compare drift magnitude in 1s checkpoints.
        marks = np.arange(1.0, dur + 1e-6, 1.0)
        if len(marks) >= 2:
            mags = []
            for m in marks:
                k = int(np.searchsorted(t - t[0], m))
                k = min(k, n - 1)
                mags.append(float(np.max(np.abs(drift_deg[k]))))
            trend = "GROWING (under-compensated)" if mags[-1] > mags[-2] + 0.05 \
                else "settling/stable"
            print(f"Max-drift trend             : "
                  f"{' -> '.join(f'{x:.2f}' for x in mags)} deg  [{trend}]")

        # How hard the comp is working.
        if self._comp is not None:
            print(f"\nPayload comp torque |tau| mean/max per joint [Nm]:")
            print(f"  mean {np.abs(tau).mean(axis=0).round(2).tolist()}")
            print(f"  max  {np.abs(tau).max(axis=0).round(2).tolist()}")

        # Tip drift via FK on the IK / full model.
        tip = self._tip_drift(q[0], q[-1])
        if tip is not None:
            print(f"\nTool-tip position drift     : {tip*1000:.1f} mm")
        print("=" * 64)
        print("Good comp -> small, settling drift and a weightless, backdrivable arm.")
        print("Big/growing J2 or J4 drop -> raise payload mass / mass_scale, "
              "or re-check the calibrated CoM.")

    def _tip_drift(self, q0: np.ndarray, q1: np.ndarray) -> float | None:
        try:
            import mujoco  # noqa: PLC0415
        except Exception:
            return None
        ik = self._cfg.get("ik", {})
        xml = ik.get("mjcf") or self._cfg.get("payload_gravity", {}).get("model_full")
        site = ik.get("site_name", "stick_tip")
        if not xml:
            return None
        xml_p = Path(xml)
        if not xml_p.is_absolute():
            xml_p = PROJECT_ROOT / xml_p
        m = mujoco.MjModel.from_xml_path(str(xml_p))
        d = mujoco.MjData(m)
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
        if sid < 0:
            return None
        d.qpos[:7] = q0; mujoco.mj_kinematics(m, d)
        p0 = d.site_xpos[sid].copy()
        d.qpos[:7] = q1; mujoco.mj_kinematics(m, d)
        p1 = d.site_xpos[sid].copy()
        return float(np.linalg.norm(p1 - p0))

    # ------------------------------------------------------------------
    def save_log(self, path: Path) -> None:
        n = self._n
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, t=self._t[:n], q=self._q[:n], dq=self._dq[:n], tau=self._tau[:n],
                 use_comp=self._use_comp)
        print(f"Log saved -> {path}  ({n} ticks)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Float test: run payload gravity comp only and measure drift/sag.")
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "real.yaml"))
    p.add_argument("--seconds", type=float, default=10.0, help="Float duration [s].")
    p.add_argument("--no-comp", action="store_true",
                   help="Command zero torque (baseline) instead of payload comp.")
    p.add_argument("--log", default=None, help="Optional .npz log of t,q,dq,tau.")
    p.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    robot_ip = cfg.get("robot_ip", "172.16.0.2")

    print("=== Gravity-Compensation Float Test ===")
    print(f"Robot     : {robot_ip}")
    print(f"Mode      : {'ZERO TORQUE (baseline)' if args.no_comp else 'payload gravity comp'}")
    print(f"Duration  : {args.seconds:.0f}s")
    print("\nSafety: the arm will be in free-float. Clear the workspace, keep a hand "
          "near the E-stop, and ensure no other process holds the robot.")
    if not args.yes:
        ans = input("Engage torque control now? (y/N): ").strip().lower()
        if ans != "y":
            print("Cancelled.")
            return

    test = GravityCompTest(cfg, args.seconds, use_comp=not args.no_comp)
    test.run(robot_ip)
    if args.log:
        test.save_log(Path(args.log))


if __name__ == "__main__":
    main()
