"""ROS FT source: subscribes to /ft_sensor/wrench from the FT PC.

Only imported on the Ubuntu Panda PC. Keeps a single latest-sample buffer;
never blocks the 1 kHz callback.
"""

from __future__ import annotations
import threading
import numpy as np

from force_control.core.types import WrenchSample


class FTRosSource:
    """Non-blocking ROS subscriber that caches the latest WrenchSample.

    Designed for use from a separate thread; the 1 kHz callback reads
    self.latest without any lock (Python GIL makes the reference swap
    effectively atomic for our purposes).

    Usage::

        ft_source = FTRosSource(topic="/ft_sensor/wrench")
        ft_source.start()          # starts rospy.spin() in a daemon thread
        ...
        sample = ft_source.latest  # read from 1 kHz callback
    """

    def __init__(self, topic: str = "/ft_sensor/wrench"):
        self._topic = topic
        self.latest: WrenchSample | None = None
        self._seq = 0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the ROS subscriber in a background daemon thread."""
        # TODO: implement
        # import rospy
        # self._thread = threading.Thread(target=self._spin, daemon=True)
        # self._thread.start()
        raise NotImplementedError

    def _spin(self) -> None:
        """Background thread: subscribe and spin."""
        # TODO: implement
        # import rospy
        # from geometry_msgs.msg import WrenchStamped
        # rospy.Subscriber(self._topic, WrenchStamped, self._cb,
        #                  queue_size=1, tcp_nodelay=True)
        # rospy.spin()
        raise NotImplementedError

    def _cb(self, msg) -> None:
        """ROS callback — called at ~400 Hz by rospy spin thread."""
        # TODO: implement
        # w = msg.wrench
        # self._seq += 1
        # self.latest = WrenchSample(
        #     t=msg.header.stamp.to_sec(),
        #     wrench=np.array([w.force.x, w.force.y, w.force.z,
        #                      w.torque.x, w.torque.y, w.torque.z]),
        #     seq=self._seq,
        # )
        raise NotImplementedError
