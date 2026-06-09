"""Read a trajectory CSV (written by core.data_logger.TrajectoryLogger) and
export a visualization of Fz, tip position, and tip velocity over time.

CSV columns: t, x, y, z, vx, vy, vz, Fz, phase

Usage:
    # newest log in data/logs, save PNGs next to it, and show interactively
    python scripts/plot_trajectory.py
    python scripts/plot_trajectory.py data/logs/real_20260609_141233.csv
    python scripts/plot_trajectory.py <file.csv> --save plots/ --no-show
"""

from __future__ import annotations
import argparse
import glob
import os

import numpy as np

try:
    import matplotlib
    # Pick a non-interactive backend when there is no display (headless / CI).
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    raise SystemExit("matplotlib is required: pip install matplotlib")


# Distinct color per phase for the timeline shading.
_PHASE_COLORS = {
    "HOLD":                  "#dddddd",
    "WAIT_USER_CONFIRM":     "#cfe8ff",
    "MOVE_JOINT":            "#ffe1b3",
    "TARE":                  "#e3d4ff",
    "WAIT_APPROACH_CONFIRM": "#cfe8ff",
    "APPROACH":              "#c9f0c9",
    "FORCE_HOLD":            "#f9c9c9",
    "FAILED":                "#ff8888",
}


def load_log(path: str) -> dict:
    """Load the CSV into a dict of arrays. 'phase' is an array of strings."""
    rows = np.genfromtxt(
        path, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    rows = np.atleast_1d(rows)
    data = {name: rows[name] for name in rows.dtype.names}
    # Normalize phase to plain python strings.
    if "phase" in data:
        data["phase"] = np.array([str(p) for p in data["phase"]])
    return data


def _newest_log(directory: str = "data/logs") -> str:
    files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV logs found in {directory}/")
    return files[-1]


def _phase_spans(t: np.ndarray, phase: np.ndarray):
    """Yield (phase_name, t_start, t_end) for each contiguous phase block."""
    if len(t) == 0:
        return
    start = 0
    for i in range(1, len(phase)):
        if phase[i] != phase[start]:
            yield phase[start], t[start], t[i]
            start = i
    yield phase[start], t[start], t[-1]


def _shade_phases(ax, t, phase, label_once: set):
    for name, t0, t1 in _phase_spans(t, phase):
        color = _PHASE_COLORS.get(name, "#f2f2f2")
        lbl = name if name not in label_once else None
        label_once.add(name)
        ax.axvspan(t0, t1, color=color, alpha=0.45, lw=0, label=lbl)


def plot_timeseries(data: dict, save_dir: str | None, show: bool, stem: str):
    """Fz, position, velocity (+ speed) vs time, with phase shading."""
    t = data["t"]
    t0 = t[0] if len(t) else 0.0
    t = t - t0
    phase = data.get("phase", np.array(["?"] * len(t)))

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    # --- Fz ---
    ax = axes[0]
    seen: set = set()
    _shade_phases(ax, t, phase, seen)
    ax.plot(t, data["Fz"], color="#c0392b", lw=1.0, label="Fz")
    ax.axhline(0.0, color="k", lw=0.6, ls=":")
    ax.set_ylabel("Fz  [N]")
    ax.set_title("Contact force (world z)")
    ax.legend(loc="upper right", fontsize=8, ncol=4)
    ax.grid(alpha=0.3)

    # --- position ---
    ax = axes[1]
    _shade_phases(ax, t, phase, set())
    for k, c in zip("xyz", ("#1f77b4", "#2ca02c", "#9467bd")):
        ax.plot(t, data[k], color=c, lw=1.0, label=k)
    ax.set_ylabel("position  [m]")
    ax.set_title("Tip position (world)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # --- velocity + speed ---
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
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, f"{stem}_timeseries.png")
        fig.savefig(out, dpi=130)
        print(f"saved {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_3d(data: dict, save_dir: str | None, show: bool, stem: str):
    """3D tip path, colored by time."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    x, y, z = data["x"], data["y"], data["z"]
    sc = ax.scatter(x, y, z, c=data["t"] - data["t"][0], cmap="viridis", s=3)
    ax.plot(x, y, z, color="gray", lw=0.4, alpha=0.6)
    ax.scatter(x[0], y[0], z[0], color="green", s=40, label="start")
    ax.scatter(x[-1], y[-1], z[-1], color="red", s=40, label="end")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title("Tip path (world)")
    ax.legend(fontsize=8)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="time [s]")

    fig.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, f"{stem}_path3d.png")
        fig.savefig(out, dpi=130)
        print(f"saved {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def summary(data: dict) -> None:
    t = data["t"]
    dur = (t[-1] - t[0]) if len(t) > 1 else 0.0
    print(f"rows={len(t)}  duration={dur:.2f}s  "
          f"Fz[min/max]={data['Fz'].min():+.2f}/{data['Fz'].max():+.2f} N")
    if "phase" in data:
        names = [n for n, _, _ in _phase_spans(t, data["phase"])]
        print("phases:", " -> ".join(dict.fromkeys(names)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot a trajectory CSV log")
    ap.add_argument("log", nargs="?", default=None,
                    help="CSV path (default: newest in data/logs)")
    ap.add_argument("--save", default=None,
                    help="directory to write PNGs (default: alongside the CSV)")
    ap.add_argument("--no-show", dest="show", action="store_false",
                    help="don't open interactive windows (just save)")
    args = ap.parse_args()

    path = args.log or _newest_log()
    print(f"loading {path}")
    data = load_log(path)
    summary(data)

    stem = os.path.splitext(os.path.basename(path))[0]
    save_dir = args.save if args.save is not None else os.path.dirname(path) or "."

    plot_timeseries(data, save_dir, args.show, stem)
    plot_3d(data, save_dir, args.show, stem)


if __name__ == "__main__":
    main()
