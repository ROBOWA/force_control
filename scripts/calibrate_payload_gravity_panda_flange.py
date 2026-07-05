"""Single-pose tool-payload calibration from the Panda's built-in flange sensing.

Companion to scripts/calibrate_payload_gravity.py (multi-pose fit of an external
Bota FT sensor via shared memory).  This variant needs NO external FT hardware:
it uses libfranka's own external-wrench estimate ``K_F_ext_hat_K`` (computed
from the joint-torque sensors, reported about the stiffness frame K) while the
arm holds one static pose — nominally home — and averages for ``--duration``
seconds, then writes franka_emika_panda/flange_payload.xml.

Procedure (control PC)
----------------------
1. Mount the tool.  Move the arm to home (Desk guiding, or
   ``python -m scripts.run_real_joint_move``).  The script does NOT move the
   robot; in idle mode the arm holds its pose on its own.
2. Run::

       python -m scripts.calibrate_payload_gravity_panda_flange \
           --ip 172.16.0.2 --duration 5

   The script zeroes the robot's configured load first (otherwise the tool is
   already compensated and invisible in F_ext), collects, fits, writes the
   fragment, then restores the previous load config — or uploads the new fit
   to the robot instead if ``--set-load`` is given (libfranka then gravity-
   compensates the tool internally, plan step E).

Physics & observability (single static pose)
--------------------------------------------
With the load config zeroed, a rigid tool of mass m and CoM c (flange frame)
produces the flange-frame external wrench

    f   = m * g_F              g_F : gravity vector in the flange frame
    tau = c x f                torque about the flange origin

so::

    mass   = |f| / 9.81                    observable
    c_perp = (f x tau) / |f|^2             CoM component PERPENDICULAR to
                                           gravity: observable
    c_par  = (c . g_hat) g_hat             CoM component ALONG gravity:
                                           NOT observable from one pose

c_par is filled from a prior: ``--com-prior``, else the CoM already stored in
the output fragment, else zero — and is reported loudly.  At home the flange
z-axis is nearly parallel to gravity, so the prior fills ~c_z.  If you need
c_z measured, calibrate at a tilted pose or use the multi-pose script.

Bias caveat: libfranka's F_ext carries a pose-dependent model-error bias that
one pose cannot separate from the payload; it is folded into the fit.

Frames: the wrench is reported about K.  The script reads F_T_EE and EE_T_K
from the robot state and transforms the wrench to the flange frame F, so a
Desk-configured tool transform (NE_T_EE / EE_T_K) does not corrupt the fit.
The fragment's frame is the ft_frame body of panda_impedance_tool.xml —
(0, 0, 0.107) from link7, identity orientation = the canonical flange.

Offline checks (no robot needed, run anywhere)::

    python -m scripts.calibrate_payload_gravity_panda_flange --selftest
    python -m scripts.calibrate_payload_gravity_panda_flange --source mujoco
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np

G = 9.81  # m/s^2

# "home" keyframe of the MJCF models (arm joints 1-7) — used only for a warning.
HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

DEFAULT_OUTPUT = "franka_emika_panda/flange_payload.xml"
DEFAULT_BODY_NAME = "flange_payload"
DEFAULT_MODEL = "franka_emika_panda/panda_impedance_tool.xml"
DEFAULT_IP = "172.16.0.2"                       # configs/real.yaml :: robot_ip
DEFAULT_DIAGINERTIA = np.array([0.0011, 0.00275, 0.00187])  # carried over; RNE gravity ignores it

MIN_FORCE_N = 0.5      # |f| below this → refuse to fit (tool too light / load not zeroed)
STATIC_DQ = 0.01       # [rad/s] max |dq| for the arm to count as static
HOME_TOL = 0.05        # [rad] per-joint warning threshold vs HOME_QPOS


# ── Math (pure numpy — covered by --selftest) ───────────────────────────────

def colmajor44(a) -> np.ndarray:
    """libfranka 4x4 transforms are column-major (16,) arrays."""
    return np.asarray(a, dtype=float).reshape(4, 4, order="F")


def wrench_K_to_flange(w_K: np.ndarray, F_T_K: np.ndarray) -> np.ndarray:
    """Re-express a wrench (about K, in K) about the flange origin, in flange axes.

    f_F   = R @ f_K
    tau_F = R @ tau_K + p x f_F        p: K origin in flange frame
    """
    R, p = F_T_K[:3, :3], F_T_K[:3, 3]
    f = R @ w_K[:3]
    tau = R @ w_K[3:] + np.cross(p, f)
    return np.concatenate([f, tau])


def fit_single_pose(wrench_F: np.ndarray, com_prior: np.ndarray) -> dict:
    """Fit mass + CoM (flange frame) from one static gravity-only wrench.

    f = m*g_F and tau = c x f  ⇒  mass = |f|/g,  c_perp = (f x tau)/|f|²
    (minimal-norm CoM ⊥ gravity); the component along g_hat is unobservable
    and is filled from com_prior's projection.
    """
    f, tau = wrench_F[:3], wrench_F[3:]
    f_norm = float(np.linalg.norm(f))
    if f_norm < MIN_FORCE_N:
        raise RuntimeError(
            f"|force| = {f_norm:.3f} N — too small to fit a payload. "
            "Is the tool mounted and the robot's load config zeroed?"
        )
    mass = f_norm / G
    ghat = f / f_norm
    c_perp = np.cross(f, tau) / f_norm**2
    c_par = float(np.dot(com_prior, ghat)) * ghat
    com = c_perp + c_par
    tau_res = tau - np.cross(com, f)            # = component of tau along f
    return {
        "mass": mass,
        "com": com,
        "ghat": ghat,
        "c_perp": c_perp,
        "c_par": c_par,
        "torque_residual_Nm": float(np.linalg.norm(tau_res)),
    }


def read_fragment_prior(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Parse (com, diaginertia) out of an existing payload fragment, if any."""
    if not path.is_file():
        return None, None
    text = path.read_text()
    m_pos = re.search(r'<inertial[^>]*\bpos="([^"]+)"', text, re.S)
    m_di = re.search(r'\bdiaginertia="([^"]+)"', text)
    com = np.array([float(v) for v in m_pos.group(1).split()]) if m_pos else None
    di = np.array([float(v) for v in m_di.group(1).split()]) if m_di else None
    return com, di


# ── MJCF fragment writer ────────────────────────────────────────────────────

def write_flange_fragment(path: Path, body_name: str, mass: float,
                          com: np.ndarray, diaginertia: np.ndarray,
                          meta: dict) -> None:
    """(Over)write the <mujocoinclude> fragment that panda_impedance_tool.xml loads."""
    com = np.asarray(com, dtype=float)
    di = np.asarray(diaginertia, dtype=float)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_lines = "\n".join(f"           {k:>18}: {v}" for k, v in meta.items())
    xml = f"""<mujocoinclude>
  <!--
    {path.name}  —  AUTO-GENERATED by scripts/calibrate_payload_gravity_panda_flange.py.
    DO NOT EDIT BY HAND; re-run the calibration to regenerate.

    Lumped tool payload (mass + CoM) in the FLANGE frame: the ft_frame body of
    panda_impedance_tool.xml at (0, 0, 0.107) from link7, identity orientation
    — matching the real Franka flange F.  <include>d inside <body name="ft_frame">
    after ft_sensor_site, so this body frame IS the flange frame and the CoM
    drops straight into pos.

    Source: single-pose fit of the Panda's own external-wrench estimate
    (K_F_ext_hat_K, transformed K -> flange), arm static, load config zeroed.
    The CoM component ALONG gravity is unobservable at one pose and is filled
    from a prior (com_par_prior below).  Rotational inertia is not measured
    (irrelevant to gravity RNE); diaginertia is carried over.

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


# ── Collection: real robot ──────────────────────────────────────────────────

def collect_franka(ip: str, duration: float, keep_load: bool,
                   set_load_after: bool, diaginertia: np.ndarray):
    """Read K_F_ext_hat_K for `duration` s; returns (wrench_K_mean, std, F_T_K, q, extras).

    Handles the robot's load configuration around the window:
    zero it before (so the tool shows up in F_ext), restore or update after.
    """
    from pylibfranka import Robot   # Linux control PC only

    robot = Robot(ip)
    s0 = robot.read_once()
    q0 = np.asarray(s0.q, dtype=float)

    # Pose sanity — warn only; the fit math is pose-general.
    dev = np.abs(q0 - HOME_QPOS)
    if np.any(dev > HOME_TOL):
        j = int(np.argmax(dev)) + 1
        print(f"⚠️  Arm is NOT at the home keyframe (joint{j} off by {dev.max():.3f} rad).")
        print("    Proceeding — the fit is valid at any static pose, but remember the")
        print("    unobservable CoM axis is whatever is parallel to gravity HERE.")

    # Flange <- K transform from the robot's own configuration.
    F_T_K = colmajor44(s0.F_T_EE) @ colmajor44(s0.EE_T_K)
    if not np.allclose(F_T_K, np.eye(4), atol=1e-6):
        print("ℹ️  K frame != flange (Desk tool transform is set); wrench will be")
        print(f"    transformed to the flange. F_T_K translation = {F_T_K[:3, 3].round(4).tolist()}")

    # Zero the configured load so the tool appears in F_ext.
    orig_load = (float(s0.m_load),
                 np.asarray(s0.F_x_Cload, dtype=float).copy(),
                 np.asarray(s0.I_load, dtype=float).copy())
    zeroed = False
    if orig_load[0] > 1e-3:
        if keep_load:
            raise RuntimeError(
                f"Robot load config is non-zero (m_load = {orig_load[0]:.3f} kg) and "
                "--keep-configured-load was given: F_ext would hide the tool. "
                "Zero the load (drop the flag, or Desk > Settings > End effector)."
            )
        print(f"ℹ️  Zeroing robot load config for collection (was m_load = {orig_load[0]:.3f} kg).")
        robot.set_load(0.0, [0.0, 0.0, 0.0], [0.0] * 9)
        zeroed = True
        time.sleep(1.0)                       # let the estimate settle

    # Collect.
    print(f"Collecting K_F_ext_hat_K for {duration:.1f} s — keep the arm untouched…")
    w_K, w_O, dq_max = [], [], 0.0
    t_end = time.monotonic() + duration
    while time.monotonic() < t_end:
        s = robot.read_once()
        w_K.append(np.asarray(s.K_F_ext_hat_K, dtype=float))
        w_O.append(np.asarray(s.O_F_ext_hat_K, dtype=float))
        dq_max = max(dq_max, float(np.max(np.abs(np.asarray(s.dq, dtype=float)))))
        time.sleep(0.005)
    w_K, w_O = np.asarray(w_K), np.asarray(w_O)
    print(f"  {len(w_K)} samples, max |dq| = {dq_max:.4f} rad/s")
    if dq_max > STATIC_DQ:
        print(f"⚠️  Arm was moving (|dq| > {STATIC_DQ}); the fit assumes a static pose.")

    # Sign: F_ext acts ON the robot, so a hanging tool reads O-frame Fz < 0.
    fz_O = float(w_O[:, 2].mean())
    if abs(fz_O) < MIN_FORCE_N:
        raise RuntimeError(
            f"Base-frame Fz = {fz_O:.3f} N is ambiguous — tool too light or load not zeroed."
        )
    sign = 1.0 if fz_O < 0.0 else -1.0
    print(f"  sign auto-detect: base-frame Fz = {fz_O:+.3f} N -> sign = {sign:+.0f}")

    def _finish():
        if set_load_after:
            return  # caller uploads the fit
        if zeroed:
            print(f"ℹ️  Restoring previous load config (m_load = {orig_load[0]:.3f} kg).")
            robot.set_load(orig_load[0], orig_load[1].tolist(), orig_load[2].tolist())

    extras = {"robot": robot, "finish": _finish, "q": q0,
              "wrench_O_mean": w_O.mean(axis=0)}
    return sign * w_K.mean(axis=0), w_K.std(axis=0), F_T_K, extras


# ── Collection: MuJoCo end-to-end check (no hardware) ───────────────────────

def collect_mujoco(model_path: str, site_name: str = "ft_sensor_site"):
    """Emulate the procedure in sim: hold home perfectly, read the flange sensors.

    Validates the whole pipeline offline — the fit should recover the mass/CoM
    of the payload fragment currently <include>d by the model.
    """
    import mujoco

    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    d.qfrc_applied[:] = d.qfrc_bias           # perfect gravity hold -> qacc ~ 0
    mujoco.mj_forward(m, d)

    def _adr(name):
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise ValueError(f"Sensor '{name}' not found in {model_path}")
        return int(m.sensor_adr[sid])

    fadr, tadr = _adr("ft_force"), _adr("ft_torque")
    w_site = np.concatenate([d.sensordata[fadr:fadr + 3], d.sensordata[tadr:tadr + 3]])

    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
    R_WS = d.site_xmat[site_id].reshape(3, 3)
    fz_world = float((R_WS @ w_site[:3])[2])
    if abs(fz_world) < 1e-9:
        raise RuntimeError("Sim flange sensor reads ~zero force — no payload in the model?")
    sign = 1.0 if fz_world < 0.0 else -1.0    # want f = m * g_F (gravity points down)
    print(f"  sim sensor: |f| = {np.linalg.norm(w_site[:3]):.4f} N, "
          f"world Fz = {fz_world:+.4f} N -> sign = {sign:+.0f}")

    extras = {"q": d.qpos[:7].copy()}
    return sign * w_site, np.zeros(6), np.eye(4), extras


# ── Self-test (pure math, no hardware, no MuJoCo) ───────────────────────────

def run_selftest() -> None:
    rng = np.random.default_rng(0)
    m_true, c_true = 0.8542, np.array([0.004321, -0.003973, 0.040417])

    def wrench_at(R_WF):
        g_F = R_WF.T @ np.array([0.0, 0.0, -G])
        f = m_true * g_F
        return np.concatenate([f, np.cross(c_true, f)])

    def rot(axis, ang):
        axis = np.asarray(axis, float) / np.linalg.norm(axis)
        K = np.cross(np.eye(3), axis)
        return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K

    # 1. Tilted pose, exact prior -> exact recovery (all 3 CoM components).
    R = rot([1, 1, 0], 0.7)
    fit = fit_single_pose(wrench_at(R), com_prior=c_true)
    assert abs(fit["mass"] - m_true) < 1e-9
    assert np.allclose(fit["com"], c_true, atol=1e-12)

    # 2. Gravity along flange z (home-like): cx, cy measured, cz from prior.
    fit = fit_single_pose(wrench_at(np.eye(3)), com_prior=np.array([9.9, 9.9, 0.040417]))
    assert np.allclose(fit["com"], c_true, atol=1e-12), "perp part must come from data"

    # 3. Wrong prior only corrupts the unobservable (∥ gravity) component.
    fit = fit_single_pose(wrench_at(np.eye(3)), com_prior=np.zeros(3))
    assert np.allclose(fit["com"][:2], c_true[:2], atol=1e-12) and abs(fit["com"][2]) < 1e-12

    # 4. K != flange: transform must undo an offset + rotation exactly.
    R_FK, p_FK = rot([0, 0, 1], 0.8), np.array([0.01, -0.02, 0.05])
    w_F = wrench_at(rot([1, 0, 1], 0.5))
    f_K = R_FK.T @ w_F[:3]
    tau_K = R_FK.T @ (w_F[3:] - np.cross(p_FK, w_F[:3]))
    F_T_K = np.eye(4); F_T_K[:3, :3] = R_FK; F_T_K[:3, 3] = p_FK
    assert np.allclose(wrench_K_to_flange(np.concatenate([f_K, tau_K]), F_T_K), w_F, atol=1e-12)

    # 5. Noise robustness: averaged noisy wrench stays close.
    w = wrench_at(R) + rng.normal(0, [0.05] * 3 + [0.002] * 3, size=(500, 6)).mean(axis=0)
    fit = fit_single_pose(w, com_prior=c_true)
    assert abs(fit["mass"] - m_true) < 5e-3 and np.linalg.norm(fit["com"] - c_true) < 2e-3

    print("✓ selftest passed (fit math + K->flange transform)")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Single-pose tool calibration from the Panda's flange F_ext "
                    "(hold home, average t seconds, write flange_payload.xml)")
    p.add_argument("--source", choices=["franka", "mujoco"], default="franka",
                   help="franka: real robot (control PC); mujoco: offline sim check")
    p.add_argument("--ip", default=DEFAULT_IP, help="robot IP (source=franka)")
    p.add_argument("-t", "--duration", type=float, default=5.0,
                   help="averaging window [s] (source=franka)")
    p.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                   help="payload fragment to write")
    p.add_argument("--body-name", default=DEFAULT_BODY_NAME)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="MJCF with the flange sensors (source=mujoco)")
    p.add_argument("--com-prior", nargs=3, type=float, metavar=("X", "Y", "Z"),
                   help="CoM prior for the unobservable (along-gravity) axis "
                        "[m, flange frame]; default: existing fragment, else 0")
    p.add_argument("--set-load", action="store_true",
                   help="after the fit, upload mass/CoM to the robot via set_load() "
                        "so libfranka gravity-compensates the tool internally")
    p.add_argument("--keep-configured-load", action="store_true",
                   help="refuse to touch the robot's load config (abort if non-zero)")
    p.add_argument("--selftest", action="store_true", help="run offline math checks and exit")
    args = p.parse_args()

    if args.selftest:
        run_selftest()
        return

    out_path = Path(args.output)
    prior_com, prior_di = read_fragment_prior(out_path)
    com_prior = (np.array(args.com_prior) if args.com_prior is not None
                 else prior_com if prior_com is not None else np.zeros(3))
    diaginertia = prior_di if prior_di is not None else DEFAULT_DIAGINERTIA
    src = ("--com-prior" if args.com_prior is not None
           else f"existing {out_path.name}" if prior_com is not None else "zeros")
    print(f"CoM prior (fills the along-gravity axis): {com_prior.round(6).tolist()}  [{src}]")

    if args.source == "mujoco":
        wrench_K, wrench_std, F_T_K, extras = collect_mujoco(args.model)
    else:
        wrench_K, wrench_std, F_T_K, extras = collect_franka(
            args.ip, args.duration, args.keep_configured_load,
            args.set_load, diaginertia)

    wrench_F = wrench_K_to_flange(wrench_K, F_T_K)
    fit = fit_single_pose(wrench_F, com_prior)

    print(f"\nWrench (flange frame): f = {wrench_F[:3].round(4).tolist()} N, "
          f"tau = {wrench_F[3:].round(5).tolist()} Nm")
    print(f"Estimated mass        : {fit['mass']:.6f} kg")
    print(f"Estimated CoM         : {fit['com'].round(6).tolist()} m (flange frame)")
    print(f"  measured  (⊥ gravity): {fit['c_perp'].round(6).tolist()}")
    print(f"  from prior (∥ gravity): {fit['c_par'].round(6).tolist()}")
    print(f"Gravity dir in flange : {fit['ghat'].round(4).tolist()}")
    print(f"Torque residual       : {fit['torque_residual_Nm']:.6f} Nm")

    meta = {
        "source": f"{args.source} single-pose flange F_ext",
        "duration_s": args.duration if args.source == "franka" else "n/a (sim)",
        "q_pose_rad": np.round(extras["q"], 5).tolist(),
        "wrench_std": np.round(wrench_std, 4).tolist(),
        "ghat_flange": np.round(fit["ghat"], 5).tolist(),
        "mass_kg": round(fit["mass"], 6),
        "com_m": np.round(fit["com"], 6).tolist(),
        "com_perp_measured_m": np.round(fit["c_perp"], 6).tolist(),
        "com_par_prior_m": np.round(fit["c_par"], 6).tolist(),
        "com_prior_source": src,
        "torque_rms_Nm": round(fit["torque_residual_Nm"], 6),
    }
    write_flange_fragment(out_path, args.body_name, fit["mass"], fit["com"],
                          diaginertia, meta)

    if args.source == "franka":
        if args.set_load:
            I9 = np.diag(diaginertia).flatten().tolist()
            extras["robot"].set_load(fit["mass"], fit["com"].tolist(), I9)
            print(f"✓ set_load({fit['mass']:.4f} kg, {fit['com'].round(5).tolist()}) uploaded — "
                  "the Franka now compensates the tool; F_ext at rest should read ~0.")
        else:
            extras["finish"]()
            print("ℹ️  Load config left as before. To have the robot compensate the tool")
            print("    internally, re-run with --set-load (or configure it in Desk).")


if __name__ == "__main__":
    main()
