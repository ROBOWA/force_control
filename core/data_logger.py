"""Lightweight trajectory logger for the 1 kHz control loop.

Records, per tick: time, world-frame tip position (3), world-frame tip linear
velocity (3), the z contact force Fz, and the state-machine phase.

RT-safe: a fixed-size numpy buffer is preallocated up front and each log() call
only writes one row by index — no disk I/O, no allocation in the hot path.
The CSV is written once by save() after the control loop has stopped.
"""

from __future__ import annotations
import csv
import os
import time
import numpy as np


class TrajectoryLogger:
    """Preallocated per-tick logger; flush to CSV with save().

    Columns written: t, x, y, z, vx, vy, vz, Fz, phase
    """

    COLUMNS = ["t", "x", "y", "z", "vx", "vy", "vz", "Fz", "phase"]
    _NUMERIC = 8  # t,x,y,z,vx,vy,vz,Fz  (phase stored separately as an int code)

    def __init__(
        self,
        output_dir: str,
        prefix: str = "run",
        capacity: int = 1_500_000,
        phase_names: dict[int, str] | None = None,
    ):
        """
        Args:
            output_dir:  directory the CSV is written to (created if missing).
            prefix:      filename prefix; final name is <prefix>_<timestamp>.csv.
            capacity:    max rows preallocated (~25 min at 1 kHz by default).
                         Logging stops (with a one-time warning) once full.
            phase_names: optional {code: name} map so the phase column is written
                         as a readable name instead of an integer code.
        """
        self._dir = output_dir
        self._prefix = prefix
        self._phase_names = phase_names or {}

        self._buf = np.zeros((capacity, self._NUMERIC), dtype=np.float64)
        self._phase = np.zeros(capacity, dtype=np.int32)
        self._cap = capacity
        self._n = 0
        self._overflowed = False

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------

    def log(
        self,
        t: float,
        x: np.ndarray,
        v: np.ndarray,
        Fz: float,
        phase_code: int = 0,
    ) -> None:
        """Append one sample. O(1), no allocation, no I/O."""
        i = self._n
        if i >= self._cap:
            if not self._overflowed:
                print(f"⚠️  TrajectoryLogger full ({self._cap} rows) — "
                      "logging stopped. Increase logging.capacity.")
                self._overflowed = True
            return
        b = self._buf[i]
        b[0] = t
        b[1] = x[0]; b[2] = x[1]; b[3] = x[2]
        b[4] = v[0]; b[5] = v[1]; b[6] = v[2]
        b[7] = Fz
        self._phase[i] = phase_code
        self._n = i + 1

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> str | None:
        """Write the recorded rows to CSV. Returns the path, or None if empty."""
        if self._n == 0:
            print("TrajectoryLogger: nothing recorded — no file written.")
            return None

        if path is None:
            os.makedirs(self._dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            path = os.path.join(self._dir, f"{self._prefix}_{stamp}.csv")

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.COLUMNS)
            for i in range(self._n):
                r = self._buf[i]
                code = int(self._phase[i])
                phase = self._phase_names.get(code, str(code))
                w.writerow([
                    f"{r[0]:.6f}", f"{r[1]:.6f}", f"{r[2]:.6f}", f"{r[3]:.6f}",
                    f"{r[4]:.6f}", f"{r[5]:.6f}", f"{r[6]:.6f}", f"{r[7]:.6f}",
                    phase,
                ])
        print(f"TrajectoryLogger: wrote {self._n} rows -> {path}")
        return path
