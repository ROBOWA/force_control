"""Standalone check that the real FT-sensor reading path works.

Runs the SAME code the controller uses (sensors.ft_ros.FTRosSource +
core.ft_processor.FTProcessor) and reports:
    1. whether any message arrives at all   (valid flag)
    2. the publish rate                      (seq advance over wall-clock)
    3. live raw + processed wrench           (push on the sensor and watch it move)

It does NOT touch the robot — safe to run any time the FT publisher is up.

Run from the repo root on the Panda/FT PC (needs rospy + the FT publisher running):

    python -m scripts.check_ft --config configs/real.yaml
    python -m scripts.check_ft --topic /ft_sensor/wrench --seconds 8
"""

from __future__ import annotations
import argparse
import sys
import time

import yaml

from sensors.ft_ros import FTRosSource
from core.ft_processor import FTProcessor


def main() -> None:
    ap = argparse.ArgumentParser(description="Check the real FT-sensor reading path")
    ap.add_argument("--config", default="configs/real.yaml",
                    help="YAML config; reads ft.topic / ft.sign / ft.lowpass_alpha")
    ap.add_argument("--topic", default=None, help="Override ft.topic from the config")
    ap.add_argument("--seconds", type=float, default=6.0,
                    help="How long to live-print the wrench")
    ap.add_argument("--wait", type=float, default=5.0,
                    help="Seconds to wait for the first message before failing")
    args = ap.parse_args()

    ft_cfg = {}
    try:
        with open(args.config) as f:
            ft_cfg = (yaml.safe_load(f) or {}).get("ft", {})
    except FileNotFoundError:
        print(f"(config {args.config} not found — using defaults)")

    topic = args.topic or ft_cfg.get("topic", "/ft_sensor/wrench")
    sign = float(ft_cfg.get("sign", 1.0))
    alpha = float(ft_cfg.get("lowpass_alpha", 1.0))

    print(f"Topic           : {topic}")
    print(f"sign / alpha    : {sign} / {alpha}")

    src = FTRosSource(topic=topic)
    proc = FTProcessor(sign=sign, lowpass_alpha=alpha)
    src.start()

    try:
        # ---- 1) Wait for the first valid sample -----------------------------
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.wait:
            if src.get_latest().valid:
                break
            time.sleep(0.05)
        else:
            print(f"\nFAIL: no message on '{topic}' within {args.wait:.0f}s (valid stayed False).")
            print("  Check, in order:")
            print(f"    rostopic list | grep ft         # does '{topic}' exist?")
            print(f"    rostopic hz   {topic}            # is it actually publishing?")
            print("    echo $ROS_MASTER_URI / $ROS_IP   # is this PC pointed at the right master?")
            return _exit(src, 1)

        print("\n[1/3] First valid sample received  ->  subscriber + callback are wired up.")

        # ---- 2) Measure the rate via seq advance ----------------------------
        s0 = src.get_latest()
        seq0, tw0 = s0.seq, time.perf_counter()
        time.sleep(2.0)
        s1 = src.get_latest()
        dseq = s1.seq - seq0
        dt = time.perf_counter() - tw0
        hz = dseq / dt if dt > 0 else 0.0

        if dseq == 0:
            print("[2/3] FAIL: a sample exists but seq is NOT advancing.")
            print("        -> the callback fired once then stopped, or the publisher died.")
            print(f"        -> confirm with: rostopic hz {topic}")
            return _exit(src, 1)
        print(f"[2/3] Rate ~{hz:.0f} Hz ({dseq} new samples in {dt:.1f}s)  ->  stream is live.")

        # ---- 3) Live values so you can excite the sensor by hand ------------
        print(f"\n[3/3] Live wrench for {args.seconds:.0f}s — PUSH on the sensor and watch Fx/Fy/Fz move.")
        print("        seq    | Fx     Fy     Fz     Tx     Ty     Tz   (raw)")
        print("               | Fx     Fy     Fz     Tx     Ty     Tz   (processed: sign+bias+lowpass)")
        prev_seq = None
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.seconds:
            s = src.get_latest()
            w = proc.process(s.wrench)
            stale = "  <-- STALE (no new sample since last print)" if s.seq == prev_seq else ""
            prev_seq = s.seq
            raw = " ".join(f"{v:6.2f}" for v in s.wrench)
            prc = " ".join(f"{v:6.2f}" for v in w)
            print(f"  {s.seq:8d} | {raw}{stale}")
            print(f"           | {prc}")
            time.sleep(0.25)

        print("\nPASS: FT reading path works (valid=True, seq advancing, values live).")
        print("      If the numbers did NOT change when you pushed, the sensor/driver side is the")
        print("      issue, not this reading code — verify with `rostopic echo " + topic + "`.")
        return _exit(src, 0)

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return _exit(src, 0)


def _exit(src: FTRosSource, code: int) -> None:
    src.stop()
    sys.exit(code)


if __name__ == "__main__":
    main()
