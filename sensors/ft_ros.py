"""ROS FT source: subscribes to a geometry_msgs/WrenchStamped topic.

Only imported on the Ubuntu Panda PC (requires rospy + geometry_msgs).
The 1 kHz Franka callback must never block waiting for a new message —
this class keeps a cached WrenchSample updated by a background thread.
"""

from __future__ import annotations
import threading
import numpy as np

from core.types import WrenchSample

# Sentinel returned before the first ROS message arrives.
_ZERO_SAMPLE = WrenchSample(t=0.0, wrench=np.zeros(6), seq=0, valid=False)


class FTRosSource:
    """Non-blocking ROS subscriber that caches the latest WrenchSample.

    The background thread subscribes to a WrenchStamped topic and stores
    each message as a WrenchSample under a lock.  get_latest() returns
    the cached value immediately — suitable for the real-time 1 kHz callback.

    Interface (same as FTMuJoCo)::

        ft = FTRosSource(topic="/ft_sensor/wrench")
        ft.start()                 # starts background subscriber thread
        sample = ft.get_latest()   # non-blocking; valid=False until first msg
        ft.stop()                  # signals background thread to exit
    """

    def __init__(
        self,
        topic: str = "/ft_sensor/wrench",
    ):
        """
        Args:
            topic:  ROS topic name publishing geometry_msgs/WrenchStamped.
        """
        self._topic = topic
        self._seq = 0
        self._lock = threading.Lock()
        self._latest: WrenchSample = _ZERO_SAMPLE
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Common FT source interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the ROS subscriber in a background daemon thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._spin, daemon=True, name="ft_ros_spin"
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to exit."""
        self._stop_event.set()

    def get_latest(self) -> WrenchSample:
        """Return the last received sample. Non-blocking.

        Returns a zero sample with valid=False if no message has arrived yet.
        """
        with self._lock:
            return self._latest

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _spin(self) -> None:
        """Background thread: subscribe and block until stop() is called."""
        # Import inside the thread — rospy must already be init'd by the caller.
        import rospy  # noqa: PLC0415
        from geometry_msgs.msg import WrenchStamped  # noqa: PLC0415

        rospy.Subscriber(
            self._topic,
            WrenchStamped,
            self._cb,
            queue_size=1,
            tcp_nodelay=True,
        )
        # Block here; rospy dispatches _cb in its own callback thread.
        # _stop_event.wait() unblocks when stop() is called.
        self._stop_event.wait()

    def _cb(self, msg) -> None:
        """ROS callback — called at ~400 Hz by rospy's internal thread.

        Converts the WrenchStamped message to a WrenchSample and stores it.
        Must not allocate heavily or block.
        """
        w = msg.wrench
        self._seq += 1
        sample = WrenchSample(
            t=msg.header.stamp.to_sec(),
            wrench=np.array(
                [w.force.x,  w.force.y,  w.force.z,
                 w.torque.x, w.torque.y, w.torque.z],
                dtype=np.float64,
            ),
            seq=self._seq,
            valid=True,
        )
        with self._lock:
            self._latest = sample
