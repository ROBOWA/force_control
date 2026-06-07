"""Franka real-robot backend — Ubuntu Panda PC only.

This module is intentionally NOT imported on Mac. It wraps pylibfranka and
the ROS FT receiver. All pylibfranka and rospy imports are local so that
import errors only occur if this backend is actually instantiated.

Threading model:
    Thread A (ROS spin): receives FT at ~400 Hz → writes _ft_buffer
    Thread B (outer loop, 400 Hz): reads FT, runs StateMachine → writes _target_buffer
    Thread C (pylibfranka 1 kHz callback): reads buffers, runs ControllerCore

The 1 kHz callback MUST return within ~1 ms. It must not block on any lock,
socket, or ROS primitive.
"""

from __future__ import annotations
import threading
import time
import numpy as np

from force_control.core.types import RobotStateLite, TargetSample, WrenchSample
from force_control.core.controller import ControllerCore
from force_control.core.state_machine import StateMachine
from force_control.core.safety import saturate_torque_rate


class FrankaBackend:
    """Connects to a real Panda and runs the 1 kHz torque-control loop.

    Usage::

        backend = FrankaBackend("172.16.0.2", config)
        backend.start_ft_thread()
        backend.start_outer_loop()
        backend.run()   # blocks inside robot.control_torques()
    """

    def __init__(self, robot_ip: str, config: dict):
        """
        Args:
            robot_ip: Panda controller IP address (e.g. "172.16.0.2")
            config:   merged real config dict (from configs/real.yaml)
        """
        self._robot_ip = robot_ip
        self._cfg = config

        # Shared buffers — written by outer threads, read by 1 kHz callback.
        # Python GIL makes single-object assignment effectively atomic for
        # reads/writes of references, which is sufficient here.
        self._ft_buffer: WrenchSample | None = None
        self._target_buffer: TargetSample | None = None

        self._stop_event = threading.Event()
        self._controller: ControllerCore | None = None
        self._state_machine: StateMachine | None = None

        # pylibfranka objects — initialised in connect()
        self._robot = None
        self._model = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to the robot and load the dynamics model.

        Import pylibfranka here so Mac never touches it.
        """
        # TODO: implement
        # from pylibfranka import Robot, Frame
        # self._robot = Robot(self._robot_ip)
        # self._model = self._robot.load_model()   # or get_model()
        raise NotImplementedError

    def start_ft_thread(self) -> threading.Thread:
        """Start the ROS FT subscriber thread (Thread A).

        Import rospy here so Mac never touches it.
        """
        # TODO: implement
        # thread = threading.Thread(target=self._ros_ft_loop, daemon=True)
        # thread.start()
        # return thread
        raise NotImplementedError

    def start_outer_loop(self) -> threading.Thread:
        """Start the 400 Hz outer-loop thread (Thread B)."""
        # TODO: implement
        # thread = threading.Thread(target=self._outer_loop, daemon=True)
        # thread.start()
        # return thread
        raise NotImplementedError

    def run(self) -> None:
        """Enter robot.control_torques() — blocks until finished or Ctrl+C."""
        # TODO: implement
        # from pylibfranka import Torques
        # initial_state = self._robot.read_once()
        # self._controller.reset(tau_init=np.zeros(7))
        # try:
        #     self._robot.control_torques(self._torque_callback)
        # except KeyboardInterrupt:
        #     pass
        raise NotImplementedError

    def stop(self) -> None:
        """Signal all background threads to exit."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # 1 kHz torque callback (Thread C)
    # ------------------------------------------------------------------

    def _torque_callback(self, robot_state, duration) -> object:
        """Called by pylibfranka at ~1000 Hz.

        MUST return a Torques object within ~1 ms.
        Must NOT block, print, write files, or wait on any synchronisation.
        """
        # TODO: implement
        # from pylibfranka import Torques
        # state   = self._build_state(robot_state, duration.to_sec())
        # target  = self._target_buffer   # atomic reference read
        # wrench  = self._ft_buffer       # atomic reference read
        # if target is None or wrench is None:
        #     return Torques(np.zeros(7).tolist())
        # tau = self._controller.step(state, target, wrench, 1e-3,
        #                             contact=self._state_machine.in_contact)
        # if <finished>:
        #     return Torques.finished(tau.tolist())
        # return Torques(tau.tolist())
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Thread A: ROS FT receiver
    # ------------------------------------------------------------------

    def _ros_ft_loop(self) -> None:
        """Subscribe to /ft_sensor/wrench and update _ft_buffer.

        queue_size=1, tcpNoDelay=True — only the latest sample is kept.
        """
        # TODO: implement
        # import rospy
        # from geometry_msgs.msg import WrenchStamped
        # rospy.Subscriber(
        #     "/ft_sensor/wrench", WrenchStamped, self._ft_ros_cb,
        #     queue_size=1, tcp_nodelay=True,
        # )
        # rospy.spin()
        raise NotImplementedError

    def _ft_ros_cb(self, msg) -> None:
        """ROS callback: unpack WrenchStamped into WrenchSample."""
        # TODO: implement
        # w = msg.wrench
        # self._ft_buffer = WrenchSample(
        #     t=msg.header.stamp.to_sec(),
        #     wrench=np.array([w.force.x, w.force.y, w.force.z,
        #                      w.torque.x, w.torque.y, w.torque.z]),
        #     seq=msg.header.seq,
        # )
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Thread B: 400 Hz outer loop
    # ------------------------------------------------------------------

    def _outer_loop(self) -> None:
        """Run StateMachine at ~400 Hz and write TargetSamples to _target_buffer."""
        # TODO: implement
        # dt = 1.0 / 400.0
        # while not self._stop_event.is_set():
        #     t0 = time.perf_counter()
        #     ft     = self._ft_buffer
        #     state  = self._last_state   # cached from callback
        #     if ft is not None and state is not None:
        #         target = self._state_machine.update(time.time(), state, ft,
        #                                              self._target_buffer)
        #         self._target_buffer = target
        #     elapsed = time.perf_counter() - t0
        #     time.sleep(max(0.0, dt - elapsed))
        raise NotImplementedError

    # ------------------------------------------------------------------
    # State conversion
    # ------------------------------------------------------------------

    def _build_state(self, robot_state, dt: float) -> RobotStateLite:
        """Convert pylibfranka RobotState → RobotStateLite.

        Computes J and coriolis via the Franka dynamics model.
        """
        # TODO: implement
        # from pylibfranka import Frame
        # q        = np.array(robot_state.q)
        # dq       = np.array(robot_state.dq)
        # O_T_EE   = np.array(self._model.pose(Frame.EndEffector, robot_state)).reshape(4, 4)
        # J_full   = np.array(self._model.zero_jacobian(Frame.EndEffector, robot_state)).reshape(6, 7)
        # coriolis = np.array(self._model.coriolis(robot_state))
        # t        = time.perf_counter()
        raise NotImplementedError
