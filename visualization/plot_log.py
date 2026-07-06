"""Post-run analysis for sim and real-robot trajectory logs.

Reads the newest log in data/logs/ by default and produces five figures:
  1. Time-series  — Fz, position, velocity with phase shading
  2. Force error  — (Fz - F_desired) during FORCE_HOLD / SLIDE phases
  3. XY overhead  — top-down tip path colored by phase (shows slide direction)
  4. Phase Gantt  — horizontal bar chart of phase durations
  5. 3-D path     — time-colored scatter of full tip trajectory

Usage:
    python visualization/plot_log.py
    python visualization/plot_log.py data/logs/sim_20260611_120000.csv
    python visualization/plot_log.py data/logs/real_20260609_013301.csv
    python visualization/plot_log.py --save plots/ --no-show
    python visualization/plot_log.py --fz-desired 3.0
"""

from __future__ import annotations
import argparse
import glob
import os

import numpy as np

try:
    import matplotlib
    import sys
    # Force non-interactive backend only on Linux without an X display.
    # On macOS the native backend works without DISPLAY being set.
    if sys.platform == "linux" and not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except ImportError:
    raise SystemExit("matplotlib is required: pip install matplotlib")

# --------------------------------------------------------------------------- #
# Phase colours (consistent with scripts/plot_trajectory.py)
# --------------------------------------------------------------------------- #
_PHASE_COLORS = {
    "HOLD":                  "#dddddd",
    "WAIT_USER_CONFIRM":     "#cfe8ff",
    "MOVE_JOINT":            "#ffe1b3",
    "TARE":                  "#e3d4ff",
    "WAIT_APPROACH_CONFIRM": "#cfe8ff",
    "APPROACH":              "#c9f0c9",
    "FORCE_HOLD":            "#f9c9c9",
    "SLIDE":                 "#ffd6a5",
    "FAILED":                "#ff8888",
}
_PHASE_ORDER = [
    "HOLD", "WAIT_USER_CONFIRM", "MOVE_JOINT", "TARE",
    "WAIT_APPROACH_CONFIRM", "APPROACH", "FORCE_HOLD", "SLIDE", "FAILED",
]


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #

def load_log(path: str) -> dict:
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    rows = np.atleast_1d(rows)
    data = {name: rows[name] for name in rows.dtype.names}
    if "phase" in data:
        data["phase"] = np.array([str(p) for p in data["phase"]])
    return data


def _newest_log(directory: str = "data/logs") -> str:
    files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV logs found in {directory}/")
    return files[-1]


def _phase_spans(t: np.ndarray, phase: np.ndarray):
    """Yield (name, t_start, t_end) for each contiguous phase block."""
    if len(t) == 0:
        return
    start = 0
    for i in range(1, len(phase)):
        if phase[i] != phase[start]:
            yield phase[start], t[start], t[i]
            start = i
    yield phase[start], t[start], t[-1]


def _shade_phases(ax, t, phase, seen: set):
    for name, t0, t1 in _phase_spans(t, phase):
        color = _PHASE_COLORS.get(name, "#f2f2f2")
        lbl = name if name not in seen else None
        seen.add(name)
        ax.axvspan(t0, t1, color=color, alpha=0.40, lw=0, label=lbl)


def summary(data: dict) -> None:
    t = data["t"]
    dur = (t[-1] - t[0]) if len(t) > 1 else 0.0
    print(f"rows={len(t)}  duration={dur:.2f}s  "
          f"Fz[min/max]={data['Fz'].min():+.2f}/{data['Fz'].max():+.2f} N")
    if "phase" in data:
        names = [n for n, _, _ in _phase_spans(t, data["phase"])]
        unique = list(dict.fromkeys(names))
        print("phases:", " → ".join(unique))

        for ph in ("FORCE_HOLD", "SLIDE"):
            mask = data["phase"] == ph
            if mask.any():
                fz_ph = data["Fz"][mask]
                print(f"  {ph}: Fz mean={fz_ph.mean():+.3f} N  "
                      f"std={fz_ph.std():.3f} N  "
                      f"min={fz_ph.min():+.3f}  max={fz_ph.max():+.3f}")


# --------------------------------------------------------------------------- #
# Figure 1 — time-series (Fz, position, velocity)
# --------------------------------------------------------------------------- #

def plot_timeseries(data: dict, save_dir: str | None, show: bool, stem: str) -> None:
    t = data["t"] - data["t"][0]
    phase = data.get("phase", np.array(["?"] * len(t)))

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"Simulation run — {stem}", fontsize=11)

    seen: set = set()
    ax = axes[0]
    _shade_phases(ax, t, phase, seen)
    ax.plot(t, data["Fz"], color="#c0392b", lw=1.0, label="Fz")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_ylabel("Fz  [N]")
    ax.set_title("Contact force (world z)")
    ax.legend(loc="upper right", fontsize=8, ncol=5)
    ax.grid(alpha=0.25)

    ax = axes[1]
    _shade_phases(ax, t, phase, set())
    for k, c in zip("xyz", ("#1f77b4", "#2ca02c", "#9467bd")):
        ax.plot(t, data[k], color=c, lw=1.0, label=k)
    ax.set_ylabel("position  [m]")
    ax.set_title("Tip position (world)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[2]
    _shade_phases(ax, t, phase, set())
    for k, c in zip(("vx", "vy", "vz"), ("#1f77b4", "#2ca02c", "#9467bd")):
        ax.plot(t, data[k], color=c, lw=0.9, label=k)
    speed = np.sqrt(data["vx"] ** 2 + data["vy"] ** 2 + data["vz"] ** 2)
    ax.plot(t, speed, color="k", lw=1.1, ls="--", label="|v|")
    ax.set_ylabel("velocity  [m/s]")
    ax.set_xlabel("time  [s]")
    ax.set_title("Tip velocity (world)")
    ax.legend(loc="upper right", fontsize=8, ncol=4)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    _save_or_show(fig, save_dir, f"{stem}_timeseries.png", show)


# --------------------------------------------------------------------------- #
# Figure 2 — force error during contact phases
# --------------------------------------------------------------------------- #

def plot_force_error(
    data: dict, fz_desired: float,
    save_dir: str | None, show: bool, stem: str,
) -> None:
    t = data["t"] - data["t"][0]
    phase = data.get("phase", np.array(["?"] * len(t)))

    contact_phases = {"FORCE_HOLD", "SLIDE"}
    mask = np.isin(phase, list(contact_phases))
    if not mask.any():
        print("  [force_error] no FORCE_HOLD/SLIDE data — skipping figure")
        return

    error = data["Fz"] - fz_desired

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f"Force regulation — {stem}  (F_desired={fz_desired:.1f} N)", fontsize=11)

    seen: set = set()
    ax = axes[0]
    _shade_phases(ax, t, phase, seen)
    ax.plot(t, data["Fz"], color="#c0392b", lw=1.0, label="Fz measured")
    ax.axhline(fz_desired, color="#27ae60", lw=1.2, ls="--", label=f"desired {fz_desired} N")
    ax.set_ylabel("Fz  [N]")
    ax.set_title("Measured vs desired contact force")
    ax.legend(loc="upper right", fontsize=8, ncol=4)
    ax.grid(alpha=0.25)

    ax = axes[1]
    _shade_phases(ax, t, phase, set())
    ax.fill_between(t, error, where=mask, color="#e74c3c", alpha=0.25)
    ax.plot(t[mask], error[mask], color="#e74c3c", lw=0.9, label="Fz error (contact only)")
    ax.axhline(0, color="k", lw=0.7, ls=":")
    e_contact = error[mask]
    ax.axhline(e_contact.mean(), color="#8e44ad", lw=1.0, ls="--",
               label=f"mean {e_contact.mean():+.3f} N")
    ax.set_ylabel("Fz − F_desired  [N]")
    ax.set_xlabel("time  [s]")
    ax.set_title("Force tracking error")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    _save_or_show(fig, save_dir, f"{stem}_force_error.png", show)


# --------------------------------------------------------------------------- #
# Figure 3 — XY overhead trajectory (great for visualising slide direction)
# --------------------------------------------------------------------------- #

def plot_xy_overhead(data: dict, save_dir: str | None, show: bool, stem: str) -> None:
    x, y = data["x"], data["y"]
    phase = data.get("phase", np.array(["?"] * len(x)))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_title(f"Tip path — top-down XY view  ({stem})", fontsize=11)

    # Plot each phase segment with its colour
    seen_phases: set = set()
    seen: set = set()
    for name, *_ in _phase_spans(data["t"], phase):
        seen_phases.add(name)

    for name in _PHASE_ORDER:
        if name not in seen_phases:
            continue
        mask = phase == name
        color = _PHASE_COLORS.get(name, "#cccccc")
        lbl = name if name not in seen else None
        seen.add(name)
        ax.plot(x[mask], y[mask], ".", color=color, ms=2, alpha=0.7, label=lbl)

    # Start / end markers
    ax.scatter(x[0], y[0], color="green", s=60, zorder=5, label="start")
    ax.scatter(x[-1], y[-1], color="red", s=60, zorder=5, label="end")

    ax.set_xlabel("x  [m]")
    ax.set_ylabel("y  [m]")
    ax.legend(loc="best", fontsize=8, markerscale=3)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    _save_or_show(fig, save_dir, f"{stem}_xy_overhead.png", show)


# --------------------------------------------------------------------------- #
# Figure 4 — phase Gantt chart
# --------------------------------------------------------------------------- #

def plot_phase_gantt(data: dict, save_dir: str | None, show: bool, stem: str) -> None:
    t = data["t"] - data["t"][0]
    phase = data.get("phase", np.array(["?"] * len(t)))

    spans = list(_phase_spans(t, phase))
    if not spans:
        return

    # Assign a y-slot per unique phase in encounter order
    encountered = list(dict.fromkeys(name for name, *_ in spans))
    y_map = {name: i for i, name in enumerate(encountered)}

    fig, ax = plt.subplots(figsize=(12, max(3, len(encountered) * 0.7 + 1)))
    fig.suptitle(f"Phase timeline — {stem}", fontsize=11)

    for name, t0, t1 in spans:
        color = _PHASE_COLORS.get(name, "#eeeeee")
        y = y_map[name]
        ax.barh(y, t1 - t0, left=t0, height=0.6, color=color,
                edgecolor="#555555", linewidth=0.5)
        dur = t1 - t0
        if dur > 0.3:
            ax.text(t0 + dur / 2, y, f"{dur:.1f}s",
                    va="center", ha="center", fontsize=7)

    ax.set_yticks(range(len(encountered)))
    ax.set_yticklabels(encountered, fontsize=9)
    ax.set_xlabel("time  [s]")
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()

    fig.tight_layout()
    _save_or_show(fig, save_dir, f"{stem}_phase_gantt.png", show)


# --------------------------------------------------------------------------- #
# Figure 5 — 3-D tip path
# --------------------------------------------------------------------------- #

def plot_3d(data: dict, save_dir: str | None, show: bool, stem: str) -> None:
    x, y, z = data["x"], data["y"], data["z"]
    t_rel = data["t"] - data["t"][0]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=t_rel, cmap="viridis", s=3)
    ax.plot(x, y, z, color="gray", lw=0.4, alpha=0.5)
    ax.scatter(x[0], y[0], z[0], color="green", s=50, label="start")
    ax.scatter(x[-1], y[-1], z[-1], color="red", s=50, label="end")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(f"Tip path 3-D — {stem}", fontsize=11)
    ax.legend(fontsize=8)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="time [s]")

    fig.tight_layout()
    _save_or_show(fig, save_dir, f"{stem}_path3d.png", show)


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #

def _save_or_show(fig, save_dir: str | None, filename: str, show: bool) -> None:
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, filename)
        fig.savefig(out, dpi=130)
        print(f"  saved {out}")
    if not show:
        plt.close(fig)
    # When show=True, leave figures open and let main() call plt.show() once.


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Post-run analysis for MuJoCo sim logs")
    ap.add_argument("log", nargs="?", default=None,
                    help="CSV path (default: newest sim_*.csv in data/logs/)")
    ap.add_argument("--save", default=None, metavar="DIR",
                    help="save PNGs to DIR (default: don't save)")
    ap.add_argument("--no-show", dest="show", action="store_false",
                    help="save only, do not open interactive windows")
    ap.add_argument("--fz-desired", type=float, default=3.0,
                    help="desired contact force in N (default: 3.0, from sim.yaml)")
    args = ap.parse_args()

    path = args.log or _newest_log()
    print(f"loading {path}")
    data = load_log(path)
    summary(data)

    stem = os.path.splitext(os.path.basename(path))[0]
    save_dir = args.save  # None = don't save; set with --save DIR

    print("generating figures…")
    plot_timeseries(data, save_dir, args.show, stem)
    plot_force_error(data, args.fz_desired, save_dir, args.show, stem)
    plot_xy_overhead(data, save_dir, args.show, stem)
    plot_phase_gantt(data, save_dir, args.show, stem)
    plot_3d(data, save_dir, args.show, stem)
    print("done.")
    if args.show:
        plt.show()  # block once with all figures open simultaneously


if __name__ == "__main__":
    main()
