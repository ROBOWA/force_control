"""Shared-memory FT transport: decouple ROS from the RT control loop.

A Python ROS subscriber that lives in the SAME process as the 1 kHz libfranka
control loop is starved by that loop's SCHED_FIFO thread (GIL + RT scheduling),
so it only ingests ~45 Hz of a 400 Hz stream. The fix is to move the subscriber
into its OWN process:

    scripts/ft_node.py   (separate process) — subscribes at full 400 Hz and
                           writes each sample into a small mmap buffer in /dev/shm
    FTShmSource          (control process)  — reads the latest sample from that
                           buffer directly: no ROS, no background thread, no GIL
                           contention. Drop-in for FTRosSource.

Concurrency: single writer, single reader, lock-free. The writer writes the
payload first and the monotonic seq counter LAST; on x86 (TSO) stores are not
reordered, so a reader that sees a new seq also sees the matching payload. The
reader reads the fields directly and never blocks or retries.

This deliberately avoids a seqlock. An earlier seqlock version returned a STALE
cached sample whenever it caught the writer mid-update — and because the writer
is a separate, lower-priority process that gets descheduled mid-write, its
"in-progress" window is open most of the time. A dense reader still caught the
brief clean windows, but the sparse 1 kHz control loop mostly landed in the
in-progress window and was starved to ~50 Hz off a 400 Hz writer. Reading single
atomic fields directly fixes that.

Buffer layout (float64 x 16):
    [0]     unused      (was a seqlock counter)
    [1]     t           ROS header stamp [s]
    [2]     msg_seq     monotonic message counter, written LAST = commit marker
    [3]     valid       1.0 once at least one message has been written
    [4:10]  wrench      Fx Fy Fz Tx Ty Tz
    [10:16] reserved
"""

from __future__ import annotations
import mmap
import os

import numpy as np

from core.types import WrenchSample

_N_SLOTS = 16
_NBYTES = _N_SLOTS * 8
_ZERO_SAMPLE = WrenchSample(t=0.0, wrench=np.zeros(6), seq=0, valid=False)


def _open_buffer(path: str, create: bool):
    """Open (or create) the mmap and return (mmap, writable float64 view)."""
    if create:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
        os.ftruncate(fd, _NBYTES)
    else:
        fd = os.open(path, os.O_RDWR)
    try:
        mm = mmap.mmap(fd, _NBYTES)
    finally:
        os.close(fd)  # mmap holds its own reference to the mapping
    buf = np.ndarray((_N_SLOTS,), dtype=np.float64, buffer=mm)
    return mm, buf


class FTShmWriter:
    """Writer side — used by the standalone ROS node process."""

    def __init__(self, path: str = "/dev/shm/ft_wrench"):
        self._mm, self._buf = _open_buffer(path, create=True)
        self._buf[:] = 0.0
        self._seq = 0

    def write(self, t: float, wrench: np.ndarray) -> None:
        """Publish one sample. Payload first, seq counter LAST (commit marker):
        on x86 a reader that sees the new seq is guaranteed to see this payload.
        """
        b = self._buf
        if b is None:                       # closed — ignore a late callback
            return
        self._seq += 1
        b[1] = t
        b[4:10] = wrench
        b[3] = 1.0
        b[2] = float(self._seq)             # published LAST = commit marker

    def close(self) -> None:
        if self._mm is None:
            return
        self._buf = None        # block any further writes before unmapping
        self._mm.close()
        self._mm = None


class FTShmSource:
    """Reader side — drop-in replacement for FTRosSource in the control process.

    No ROS, no background thread: get_latest() reads the mmap directly and is
    safe to call from the 1 kHz RT callback (non-blocking, bounded retries).
    """

    def __init__(self, shm_path: str = "/dev/shm/ft_wrench"):
        self._path = shm_path
        self._mm = None
        self._buf = None
        self._last: WrenchSample = _ZERO_SAMPLE

    # Common FT source interface -------------------------------------------

    def start(self) -> None:
        """Attach to the shared-memory buffer created by the FT node."""
        if not os.path.exists(self._path):
            raise FileNotFoundError(
                f"FT shared memory '{self._path}' not found. "
                f"Start the FT node first:\n"
                f"    python -m scripts.ft_node --config configs/real.yaml"
            )
        self._mm, self._buf = _open_buffer(self._path, create=False)

    def get_latest(self) -> WrenchSample:
        """Return the latest sample. Non-blocking, no retries — safe in the RT
        callback. Reads the commit marker (seq) and payload directly; single
        8-byte fields are atomic on x86, so this never returns a stale cached
        sample, even when the writer is descheduled mid-update.
        """
        b = self._buf
        if b is None:
            return self._last
        if b[3] >= 0.5:                     # valid: at least one sample written
            self._last = WrenchSample(
                t=float(b[1]), wrench=b[4:10].copy(), seq=int(b[2]), valid=True
            )
        return self._last

    def stop(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
            self._buf = None
