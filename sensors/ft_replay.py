"""CSV replay FT source: streams pre-recorded FT + target data.

Useful for offline debugging without MuJoCo or a real Panda.
No MuJoCo, pylibfranka, or ROS imports.
"""

from __future__ import annotations
import csv
import time
import numpy as np

from core.types import WrenchSample, TargetSample


class FTReplaySource:
    """Replays FT samples from a CSV log at their original timestamps.

    Expected CSV columns (header row required):
        t, fx, fy, fz, tx, ty, tz

    Usage::

        source = FTReplaySource("data/logs/run_001.csv")
        source.load()
        while True:
            sample = source.get_latest(time.time())
    """

    def __init__(self, csv_path: str, realtime: bool = True):
        """
        Args:
            csv_path: path to the CSV log file
            realtime: if True, block until the sample's timestamp is reached
        """
        self._path = csv_path
        self._realtime = realtime
        self._samples: list[WrenchSample] = []
        self._idx = 0
        self._t0_wall: float | None = None
        self._t0_log: float | None = None

    def load(self) -> None:
        """Read all rows from the CSV into memory."""
        # TODO: implement
        # with open(self._path) as f:
        #     reader = csv.DictReader(f)
        #     for i, row in enumerate(reader):
        #         self._samples.append(WrenchSample(
        #             t=float(row["t"]),
        #             wrench=np.array([float(row[k]) for k in ("fx","fy","fz","tx","ty","tz")]),
        #             seq=i,
        #         ))
        raise NotImplementedError

    def start(self, t_wall: float) -> None:
        """Record the wall-clock origin so replay timestamps align."""
        self._t0_wall = t_wall
        self._t0_log = self._samples[0].t if self._samples else 0.0

    def get_latest(self, t_wall: float) -> WrenchSample | None:
        """Return the most recent sample whose log-time ≤ current replay time.

        Args:
            t_wall: current wall-clock time [s]

        Returns:
            WrenchSample or None if replay hasn't started yet.
        """
        # TODO: implement
        # t_log = self._t0_log + (t_wall - self._t0_wall)
        # advance self._idx while next sample's t <= t_log
        raise NotImplementedError

    @property
    def done(self) -> bool:
        """True when all samples have been played back."""
        return self._idx >= len(self._samples)
