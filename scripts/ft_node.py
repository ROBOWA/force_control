"""Standalone FT ROS->shared-memory bridge (runs in its OWN process).

Because a Python ROS subscriber sharing a process with the 1 kHz libfranka
control loop is starved to ~45 Hz, this node runs separately — its rospy thread
is scheduled normally and ingests the full 400 Hz — and writes each sample into
the /dev/shm buffer that the controller reads via sensors.ft_shm.FTShmSource.

Start this BEFORE the controller, in its own terminal:

    python -m scripts.ft_node --config configs/real.yaml
    python -m scripts.ft_node --topic /ft_sensor/wrench --shm-path /dev/shm/ft_wrench

Leave it running for the whole session; Ctrl+C to stop.
"""

from __future__ import annotations
import argparse
import time

import numpy as np
import yaml

from sensors.ft_shm import FTShmWriter


def main() -> None:
    ap = argparse.ArgumentParser(description="FT ROS -> shared-memory bridge")
    ap.add_argument("--config", default="configs/real.yaml",
                    help="YAML config; reads ft.topic / ft.shm_path")
    ap.add_argument("--topic", default=None, help="Override ft.topic")
    ap.add_argument("--shm-path", default=None, help="Override ft.shm_path")
    args = ap.parse_args()

    ft_cfg = {}
    try:
        with open(args.config) as f:
            ft_cfg = (yaml.safe_load(f) or {}).get("ft", {})
    except FileNotFoundError:
        print(f"(config {args.config} not found — using defaults)")

    topic = args.topic or ft_cfg.get("topic", "/ft_sensor/wrench")
    shm_path = args.shm_path or ft_cfg.get("shm_path", "/dev/shm/ft_wrench")

    import rospy
    from geometry_msgs.msg import WrenchStamped

    writer = FTShmWriter(shm_path)
    rospy.init_node("ft_shm_bridge", anonymous=True, disable_signals=True)

    n = [0]

    def cb(msg) -> None:
        w = msg.wrench
        writer.write(
            msg.header.stamp.to_sec(),
            np.array(
                [w.force.x, w.force.y, w.force.z,
                 w.torque.x, w.torque.y, w.torque.z],
                dtype=np.float64,
            ),
        )
        n[0] += 1

    # NO queue_size (=> rospy None): keep EVERY message in each received batch,
    # exactly like `rostopic hz`. A non-None queue_size makes rospy discard all
    # but the last N messages of each batch (deserialize_messages) — that is what
    # throttled the bridge to ~80 Hz. Large buff_size lets one recv drain the
    # socket; tcp_nodelay disables Nagle on the publisher side.
    sub = rospy.Subscriber(
        topic, WrenchStamped, cb, tcp_nodelay=True, buff_size=2 ** 24
    )
    print(f"FT bridge running: {topic}  ->  {shm_path}   (Ctrl+C to stop)")

    # Periodic liveness print so you can confirm the bridge ingests full rate.
    try:
        rate = rospy.Rate(1.0)
        last = 0
        while not rospy.is_shutdown():
            rate.sleep()
            print(f"  bridged {n[0] - last} msg/s (total {n[0]})")
            last = n[0]
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass
    finally:
        # Stop callbacks BEFORE unmapping the buffer, or a late callback writes to
        # freed memory and segfaults. unregister + a short settle does that.
        sub.unregister()
        time.sleep(0.2)
        writer.close()
        # NOTE: deliberately do NOT unlink the shm file. Keeping a stable inode at
        # the path means a controller that is already attached stays connected when
        # this bridge is restarted (a new run reuses + re-truncates the same file).
        # Unlinking would orphan any attached reader on the next restart.
        print("FT bridge stopped (shm left in place for attached readers).")


if __name__ == "__main__":
    main()
