"""Offline log analysis and plotting.

Usage:
    python -m force_control.scripts.analyze_log data/logs/run_001.csv
    python -m force_control.scripts.analyze_log data/logs/run_001.csv --save plots/
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np


# Column groups expected in the CSV log
_STATE_COLS  = ["t", "q0","q1","q2","q3","q4","q5","q6",
                 "dq0","dq1","dq2","dq3","dq4","dq5","dq6"]
_WRENCH_COLS = ["ft_t", "fx","fy","fz","tx","ty","tz"]
_TAU_COLS    = ["tau0","tau1","tau2","tau3","tau4","tau5","tau6"]
_PHASE_COL   = "phase"


def load_log(path: str) -> dict[str, np.ndarray]:
    """Load a CSV log into a dict of numpy arrays keyed by column name."""
    # TODO: implement
    raise NotImplementedError


def plot_forces(data: dict[str, np.ndarray], save_dir: str | None = None) -> None:
    """Plot Fx/Fy/Fz over time."""
    # TODO: implement with matplotlib
    raise NotImplementedError


def plot_torques(data: dict[str, np.ndarray], save_dir: str | None = None) -> None:
    """Plot joint torques over time."""
    # TODO: implement with matplotlib
    raise NotImplementedError


def plot_ee_trajectory(data: dict[str, np.ndarray], save_dir: str | None = None) -> None:
    """Plot EE x/y/z position over time."""
    # TODO: implement with matplotlib
    raise NotImplementedError


def summary_stats(data: dict[str, np.ndarray]) -> None:
    """Print mean/std/max of key signals."""
    # TODO: implement
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse a force-control CSV log")
    parser.add_argument("log", type=str, help="Path to CSV log file")
    parser.add_argument("--save", type=str, default=None,
                        help="Directory to save plot images (optional)")
    args = parser.parse_args()

    data = load_log(args.log)
    summary_stats(data)
    plot_forces(data, save_dir=args.save)
    plot_torques(data, save_dir=args.save)
    plot_ee_trajectory(data, save_dir=args.save)


if __name__ == "__main__":
    main()
