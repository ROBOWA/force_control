"""Watch data/logs/ and auto-plot the moment a new sim CSV appears.

Run this in a second terminal before or during the simulation. When the
control loop finishes and TrajectoryLogger writes the CSV, this script
detects it and immediately calls sim_analysis to display all figures.

Usage:
    python visualization/live_monitor.py               # watch data/logs/
    python visualization/live_monitor.py --dir /tmp/logs --interval 1.0
    python visualization/live_monitor.py --no-show --save plots/
"""

from __future__ import annotations
import argparse
import glob
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import visualization.plot_log as sa


def _current_sim_logs(directory: str) -> set[str]:
    return set(glob.glob(os.path.join(directory, "sim_*.csv")))


def watch(
    directory: str = "data/logs",
    interval: float = 1.0,
    save_dir: str | None = None,
    show: bool = True,
    fz_desired: float = 3.0,
) -> None:
    abs_dir = os.path.join(_ROOT, directory) if not os.path.isabs(directory) else directory
    os.makedirs(abs_dir, exist_ok=True)

    print(f"watching {abs_dir}  (polling every {interval}s) — Ctrl-C to quit")
    known = _current_sim_logs(abs_dir)
    print(f"  {len(known)} existing sim log(s) ignored")

    try:
        while True:
            time.sleep(interval)
            current = _current_sim_logs(abs_dir)
            new = current - known
            for path in sorted(new):
                print(f"\nnew log detected: {path}")
                try:
                    data = sa.load_log(path)
                    sa.summary(data)
                    stem = os.path.splitext(os.path.basename(path))[0]
                    out_dir = save_dir or os.path.dirname(path) or "."
                    print("generating figures…")
                    sa.plot_timeseries(data, out_dir, show, stem)
                    sa.plot_force_error(data, fz_desired, out_dir, show, stem)
                    sa.plot_xy_overhead(data, out_dir, show, stem)
                    sa.plot_phase_gantt(data, out_dir, show, stem)
                    sa.plot_3d(data, out_dir, show, stem)
                    if show:
                        sa.plt.show()
                    print("done — resuming watch\n")
                except Exception as exc:
                    print(f"  error plotting {path}: {exc}")
            known = current
    except KeyboardInterrupt:
        print("\nstopped.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-plot new sim logs as they appear")
    ap.add_argument("--dir", default="data/logs",
                    help="directory to watch (default: data/logs)")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="polling interval in seconds (default: 1.0)")
    ap.add_argument("--save", default=None,
                    help="save PNGs to this directory (default: alongside CSV)")
    ap.add_argument("--no-show", dest="show", action="store_false",
                    help="save only, do not open interactive windows")
    ap.add_argument("--fz-desired", type=float, default=3.0,
                    help="desired contact force in N (default: 3.0)")
    args = ap.parse_args()

    watch(
        directory=args.dir,
        interval=args.interval,
        save_dir=args.save,
        show=args.show,
        fz_desired=args.fz_desired,
    )


if __name__ == "__main__":
    main()
