"""Diagnose WHERE the FT burst comes from: the publisher, or the delivery.

Subscribes with rospy.AnyMsg (like `rostopic` — needs NO geometry_msgs, works in
any env where rostopic works) and records, per message, the wall-clock arrival
time and the publisher's header stamp (parsed from the raw bytes). Then compares:

  - arrivals BURSTY (many ~0 gaps + occasional ~20 ms gaps) -> messages are
    delivered in bursts. Re-run under `chrt` (RT priority): if arrivals become
    smooth, the burst was the subscriber being descheduled (fixable downstream by
    running ft_node under chrt). If still bursty, the publisher/network batches
    (fix on the FT PC).
  - the header-stamp gaps show whether the SENSOR was sampled evenly (~one period)
    regardless of how the messages were delivered.

Run like ft_node (NOT inside a venv that lacks ROS), controller can be stopped:

    python -m scripts.ft_arrival_check --config configs/real.yaml --seconds 3
    sudo -E chrt -f 80 $(which python) -m scripts.ft_arrival_check --seconds 3
"""

from __future__ import annotations
import argparse
import struct
import time

import numpy as np
import yaml


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose FT burst source")
    ap.add_argument("--config", default="configs/real.yaml")
    ap.add_argument("--topic", default=None)
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()

    ft = {}
    try:
        ft = (yaml.safe_load(open(args.config)) or {}).get("ft", {})
    except FileNotFoundError:
        pass
    topic = args.topic or ft.get("topic", "/ft_sensor/wrench")

    import rospy  # AnyMsg avoids needing the message package (geometry_msgs)

    arrivals: list[float] = []
    stamps: list[float] = []

    def cb(msg) -> None:
        arrivals.append(time.perf_counter())
        # WrenchStamped raw layout: Header{seq u32, sec u32, nsec u32, frame_id...}
        # -> stamp seconds at bytes [4:12]. Best-effort; ignore if it doesn't parse.
        try:
            sec, nsec = struct.unpack_from("<II", msg._buff, 4)
            stamps.append(sec + nsec * 1e-9)
        except Exception:
            stamps.append(0.0)

    rospy.init_node("ft_arrival_check", anonymous=True, disable_signals=True)
    rospy.Subscriber(topic, rospy.AnyMsg, cb, tcp_nodelay=True, buff_size=2 ** 24)
    print(f"Measuring {topic} for {args.seconds:.0f}s ...")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.seconds and not rospy.is_shutdown():
        time.sleep(0.05)

    n = len(arrivals)
    if n < 10:
        print(f"Only {n} messages — is the publisher up and ROS_MASTER_URI set?")
        return

    aw = np.diff(np.array(arrivals)) * 1e3    # inter-arrival [ms]
    dur = arrivals[-1] - arrivals[0]
    print(f"\nmessages            : {n} over {dur:.2f}s  (avg {n / dur:.0f}/s)\n")

    print("inter-ARRIVAL gaps (wall clock) [ms]:")
    print(f"  median {np.median(aw):6.2f}   p95 {np.percentile(aw, 95):6.2f}   max {aw.max():6.2f}")
    print(f"  back-to-back (<0.5ms, inside a burst): {100 * np.mean(aw < 0.5):3.0f}%")
    print(f"  big gaps     (>10ms, between bursts) : {100 * np.mean(aw > 10):3.0f}%\n")

    st = np.array(stamps)
    if np.any(st > 0):
        ast = np.diff(st[st > 0]) * 1e3
        ast = ast[(ast > -1000) & (ast < 1000)]   # drop any parse glitches
        if len(ast):
            print("inter-STAMP gaps (publisher's own clock) [ms]:")
            print(f"  median {np.median(ast):6.2f}   p95 {np.percentile(ast, 95):6.2f}   max {ast.max():6.2f}\n")

    bursty = np.mean(aw < 0.5) > 0.3 or np.percentile(aw, 95) > 10
    if bursty:
        print(">> ARRIVALS are BURSTY. Re-run under RT priority:")
        print("     sudo -E chrt -f 80 $(which python) -m scripts.ft_arrival_check --seconds 3")
        print("   If it becomes smooth -> run ft_node under chrt for fresh data.")
        print("   If still bursty     -> publisher/network batches; fix the FT PC.")
    else:
        print(">> Arrivals are SMOOTH at this layer (median ~ one sensor period).")


if __name__ == "__main__":
    main()
