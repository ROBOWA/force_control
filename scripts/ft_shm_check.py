"""Measure what a fresh FTShmSource reader observes from the live /dev/shm buffer.

Run this in a THIRD terminal WHILE scripts.ft_node is feeding the buffer
(and optionally while the controller runs). It isolates the shm transport/reader
from the control loop:

    python -m scripts.ft_shm_check --config configs/real.yaml --seconds 3

Interpretation:
  distinct seq/s ~= ft_node's bridge rate (e.g. ~400)  -> transport+reader are
      fine; the controller's lower number is a control-loop / stale-attach issue.
  distinct seq/s ~= ~50 (much less than ft_node)        -> the cross-process read
      itself is the bottleneck (e.g. controller attached to a stale inode, or a
      seqlock/visibility problem).
"""

from __future__ import annotations
import argparse
import time

import yaml

from sensors.ft_shm import FTShmSource


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure live FTShmSource read rate")
    ap.add_argument("--config", default="configs/real.yaml")
    ap.add_argument("--shm-path", default=None)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--hz", type=float, default=0.0,
                    help="read rate cap [Hz]; 0 = read as fast as possible")
    args = ap.parse_args()

    ft = {}
    try:
        ft = (yaml.safe_load(open(args.config)) or {}).get("ft", {})
    except FileNotFoundError:
        pass
    path = args.shm_path or ft.get("shm_path", "/dev/shm/ft_wrench")

    print(f"Attaching reader to: {path}")
    src = FTShmSource(shm_path=path)
    src.start()

    reads = 0
    distinct_seq = 0
    distinct_t = 0
    prev_seq = None
    prev_t = None
    first_seq = None
    last_seq = None
    period = (1.0 / args.hz) if args.hz > 0 else 0.0

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        s = src.get_latest()
        reads += 1
        if s.valid:
            if first_seq is None:
                first_seq = s.seq
            last_seq = s.seq
            if s.seq != prev_seq:
                distinct_seq += 1
                prev_seq = s.seq
            if s.t != prev_t:
                distinct_t += 1
                prev_t = s.t
        if period:
            time.sleep(period)
    dt = time.perf_counter() - t0
    src.stop()

    print(f"reads/s         : {reads / dt:.0f}")
    print(f"distinct seq/s  : {distinct_seq / dt:.0f}   <- writer updates this reader actually saw")
    print(f"distinct t/s    : {distinct_t / dt:.0f}")
    if first_seq is not None:
        span = (last_seq - first_seq) / dt
        print(f"writer seq span : {span:.0f} /s   <- (last_seq-first_seq)/time = true writer rate")
        if distinct_seq / dt < 0.5 * span:
            print("NOTE: reader saw far fewer distinct values than the writer advanced "
                  "-> reads are missing updates (stale reads), not a slow writer.")


if __name__ == "__main__":
    main()
