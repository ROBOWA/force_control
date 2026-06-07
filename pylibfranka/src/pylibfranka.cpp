// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE

// C++ standard library headers
#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <string>

// Third-party library headers
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Libfranka
#include <franka/control_types.h>
#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/model.h>
#include <franka/robot.h>

namespace py = pybind11;

namespace pylibfranka {

PYBIND11_MODULE(_pylibfranka, m) {
  m.doc() = "Python bindings for Franka Emika Robot Control Library (libfranka)";

  // Bind exceptions
  py::register_exception<franka::Exception>(m, "FrankaException");
  py::register_exception<franka::CommandException>(m, "CommandException", PyExc_RuntimeError);
  py::register_exception<franka::NetworkException>(m, "NetworkException", PyExc_RuntimeError);
  py::register_exception<franka::ControlException>(m, "ControlException", PyExc_RuntimeError);
  py::register_exception<franka::InvalidOperationException>(m, "InvalidOperationException",
                                                            PyExc_RuntimeError);
  py::register_exception<franka::RealtimeException>(m, "RealtimeException", PyExc_RuntimeError);

  // Bind Duration
  py::class_<franka::Duration>(m, "Duration", R"pbdoc(
        Time duration representation.
    )pbdoc")
      .def("to_sec", &franka::Duration::toSec, R"pbdoc(
        Convert duration to seconds.

        @return Duration in seconds as float
    )pbdoc")
      .def("to_msec", &franka::Duration::toMSec, R"pbdoc(
        Convert duration to milliseconds.

        @return Duration in milliseconds as integer
    )pbdoc");

  // Bind enums
  py::enum_<franka::ControllerMode>(m, "ControllerMode", R"pbdoc(
        Controller mode for motion control.
    )pbdoc")
      .value("JointImpedance", franka::ControllerMode::kJointImpedance,
             "Joint impedance control mode")
      .value("CartesianImpedance", franka::ControllerMode::kCartesianImpedance,
             "Cartesian impedance control mode");

  // Bind RealtimeConfig enum
  py::enum_<franka::RealtimeConfig>(m, "RealtimeConfig", R"pbdoc(
        Real-time configuration options.
    )pbdoc")
      .value("kEnforce", franka::RealtimeConfig::kEnforce, "Enforce real-time requirements")
      .value("kIgnore", franka::RealtimeConfig::kIgnore, "Ignore real-time requirements");

  // Bind RobotMode enum
  py::enum_<franka::RobotMode>(m, "RobotMode", R"pbdoc(
        Robot operating mode.
    )pbdoc")
      .value("Other", franka::RobotMode::kOther, "Other mode")
      .value("Idle", franka::RobotMode::kIdle, "Idle mode")
      .value("Move", franka::RobotMode::kMove, "Move mode")
      .value("Guiding", franka::RobotMode::kGuiding, "Guiding mode")
      .value("Reflex", franka::RobotMode::kReflex, "Reflex mode")
      .value("UserStopped", franka::RobotMode::kUserStopped, "User stopped mode")
      .value("AutomaticErrorRecovery", franka::RobotMode::kAutomaticErrorRecovery,
             "Automatic error recovery mode");

  // Bind Frame enum
  py::enum_<franka::Frame>(m, "Frame", R"pbdoc(
        Reference frames for computing Jacobians.
    )pbdoc")
      .value("Joint1", franka::Frame::kJoint1, "Joint 1 frame")
      .value("Joint2", franka::Frame::kJoint2, "Joint 2 frame")
      .value("Joint3", franka::Frame::kJoint3, "Joint 3 frame")
      .value("Joint4", franka::Frame::kJoint4, "Joint 4 frame")
      .value("Joint5", franka::Frame::kJoint5, "Joint 5 frame")
      .value("Joint6", franka::Frame::kJoint6, "Joint 6 frame")
      .value("Joint7", franka::Frame::kJoint7, "Joint 7 frame")
      .value("Flange", franka::Frame::kFlange, "Flange frame")
      .value("EndEffector", franka::Frame::kEndEffector, "End effector frame")
      .value("Stiffness", franka::Frame::kStiffness, "Stiffness frame");

  // Bind Errors struct
  py::class_<franka::Errors>(m, "Errors", R"pbdoc(
        Robot error state containing boolean flags for all possible errors.
    )pbdoc")
      .def(py::init<>())
      .def("__bool__", &franka::Errors::operator bool, R"pbdoc(
        Check if any error is present.

        @return True if any error flag is set
    )pbdoc")
      .def("__str__", &franka::Errors::operator std::string, R"pbdoc(
        Get string representation of active errors.

        @return Comma-separated list of active error names
    )pbdoc")
      .def_property_readonly(
          "joint_position_limits_violation",
          [](const franka::Errors& self) { return self.joint_position_limits_violation; })
      .def_property_readonly(
          "cartesian_position_limits_violation",
          [](const franka::Errors& self) { return self.cartesian_position_limits_violation; })
      .def_property_readonly(
          "self_collision_avoidance_violation",
          [](const franka::Errors& self) { return self.self_collision_avoidance_violation; })
      .def_property_readonly(
          "joint_velocity_violation",
          [](const franka::Errors& self) { return self.joint_velocity_violation; })
      .def_property_readonly(
          "cartesian_velocity_violation",
          [](const franka::Errors& self) { return self.cartesian_velocity_violation; })
      .def_property_readonly(
          "force_control_safety_violation",
          [](const franka::Errors& self) { return self.force_control_safety_violation; })
      .def_property_readonly("joint_reflex",
                             [](const franka::Errors& self) { return self.joint_reflex; })
      .def_property_readonly("cartesian_reflex",
                             [](const franka::Errors& self) { return self.cartesian_reflex; })
      .def_property_readonly(
          "max_goal_pose_deviation_violation",
          [](const franka::Errors& self) { return self.max_goal_pose_deviation_violation; })
      .def_property_readonly(
          "max_path_pose_deviation_violation",
          [](const franka::Errors& self) { return self.max_path_pose_deviation_violation; })
      .def_property_readonly("cartesian_velocity_profile_safety_violation",
                             [](const franka::Errors& self) {
                               return self.cartesian_velocity_profile_safety_violation;
                             })
      .def_property_readonly("joint_position_motion_generator_start_pose_invalid",
                             [](const franka::Errors& self) {
                               return self.joint_position_motion_generator_start_pose_invalid;
                             })
      .def_property_readonly("joint_motion_generator_position_limits_violation",
                             [](const franka::Errors& self) {
                               return self.joint_motion_generator_position_limits_violation;
                             })
      .def_property_readonly("joint_motion_generator_velocity_limits_violation",
                             [](const franka::Errors& self) {
                               return self.joint_motion_generator_velocity_limits_violation;
                             })
      .def_property_readonly("joint_motion_generator_velocity_discontinuity",
                             [](const franka::Errors& self) {
                               return self.joint_motion_generator_velocity_discontinuity;
                             })
      .def_property_readonly("joint_motion_generator_acceleration_discontinuity",
                             [](const franka::Errors& self) {
                               return self.joint_motion_generator_acceleration_discontinuity;
                             })
      .def_property_readonly("cartesian_position_motion_generator_start_pose_invalid",
                             [](const franka::Errors& self) {
                               return self.cartesian_position_motion_generator_start_pose_invalid;
                             })
      .def_property_readonly("cartesian_motion_generator_elbow_limit_violation",
                             [](const franka::Errors& self) {
                               return self.cartesian_motion_generator_elbow_limit_violation;
                             })
      .def_property_readonly("cartesian_motion_generator_velocity_limits_violation",
                             [](const franka::Errors& self) {
                               return self.cartesian_motion_generator_velocity_limits_violation;
                             })
      .def_property_readonly("cartesian_motion_generator_velocity_discontinuity",
                             [](const franka::Errors& self) {
                               return self.cartesian_motion_generator_velocity_discontinuity;
                             })
      .def_property_readonly("cartesian_motion_generator_acceleration_discontinuity",
                             [](const franka::Errors& self) {
                               return self.cartesian_motion_generator_acceleration_discontinuity;
                             })
      .def_property_readonly("cartesian_motion_generator_elbow_sign_inconsistent",
                             [](const franka::Errors& self) {
                               return self.cartesian_motion_generator_elbow_sign_inconsistent;
                             })
      .def_property_readonly("cartesian_motion_generator_start_elbow_invalid",
                             [](const franka::Errors& self) {
                               return self.cartesian_motion_generator_start_elbow_invalid;
                             })
      .def_property_readonly(
          "cartesian_motion_generator_joint_position_limits_violation",
          [](const franka::Errors& self) {
            return self.cartesian_motion_generator_joint_position_limits_violation;
          })
      .def_property_readonly(
          "cartesian_motion_generator_joint_velocity_limits_violation",
          [](const franka::Errors& self) {
            return self.cartesian_motion_generator_joint_velocity_limits_violation;
          })
      .def_property_readonly("cartesian_motion_generator_joint_velocity_discontinuity",
                             [](const franka::Errors& self) {
                               return self.cartesian_motion_generator_joint_velocity_discontinuity;
                             })
      .def_property_readonly(
          "cartesian_motion_generator_joint_acceleration_discontinuity",
          [](const franka::Errors& self) {
            return self.cartesian_motion_generator_joint_acceleration_discontinuity;
          })
      .def_property_readonly("cartesian_position_motion_generator_invalid_frame",
                             [](const franka::Errors& self) {
                               return self.cartesian_position_motion_generator_invalid_frame;
                             })
      .def_property_readonly("force_controller_desired_force_tolerance_violation",
                             [](const franka::Errors& self) {
                               return self.force_controller_desired_force_tolerance_violation;
                             })
      .def_property_readonly(
          "controller_torque_discontinuity",
          [](const franka::Errors& self) { return self.controller_torque_discontinuity; })
      .def_property_readonly(
          "start_elbow_sign_inconsistent",
          [](const franka::Errors& self) { return self.start_elbow_sign_inconsistent; })
      .def_property_readonly(
          "communication_constraints_violation",
          [](const franka::Errors& self) { return self.communication_constraints_violation; })
      .def_property_readonly("power_limit_violation",
                             [](const franka::Errors& self) { return self.power_limit_violation; })
      .def_property_readonly("joint_p2p_insufficient_torque_for_planning",
                             [](const franka::Errors& self) {
                               return self.joint_p2p_insufficient_torque_for_planning;
                             })
      .def_property_readonly("tau_j_range_violation",
                             [](const franka::Errors& self) { return self.tau_j_range_violation; })
      .def_property_readonly("instability_detected",
                             [](const franka::Errors& self) { return self.instability_detected; });

  // Bind franka::RobotState
  py::class_<franka::RobotState>(m, "RobotState", R"pbdoc(
        Current state of the Franka robot.

        Contains all robot state information including joint positions, velocities,
        torques, Cartesian poses, and error states. All arrays are NumPy arrays.
    )pbdoc")
      .def_readwrite("q", &franka::RobotState::q, "Joint positions [rad] (7,)")
      .def_readwrite("q_d", &franka::RobotState::q_d, "Desired joint positions [rad] (7,)")
      .def_readwrite("dq", &franka::RobotState::dq, "Joint velocities [rad/s] (7,)")
      .def_readwrite("dq_d", &franka::RobotState::dq_d, "Desired joint velocities [rad/s] (7,)")
      .def_readwrite("ddq_d", &franka::RobotState::ddq_d,
                     "Desired joint accelerations [rad/s²] (7,)")
      .def_readwrite("tau_J", &franka::RobotState::tau_J, "Measured joint torques [Nm] (7,)")
      .def_readwrite("tau_J_d", &franka::RobotState::tau_J_d, "Desired joint torques [Nm] (7,)")
      .def_readwrite("dtau_J", &franka::RobotState::dtau_J, "Joint torque derivatives [Nm/s] (7,)")
      .def_readwrite("tau_ext_hat_filtered", &franka::RobotState::tau_ext_hat_filtered,
                     "Filtered external torque [Nm] (7,)")
      .def_readwrite("theta", &franka::RobotState::theta, "Motor positions [rad] (7,)")
      .def_readwrite("dtheta", &franka::RobotState::dtheta, "Motor velocities [rad/s] (7,)")
      .def_readwrite("O_T_EE", &franka::RobotState::O_T_EE,
                     "End effector pose in base frame (16,) column-major")
      .def_readwrite("O_T_EE_d", &franka::RobotState::O_T_EE_d,
                     "Desired end effector pose (16,) column-major")
      .def_readwrite("O_T_EE_c", &franka::RobotState::O_T_EE_c,
                     "Commanded end effector pose (16,) column-major")
      .def_readwrite("F_T_EE", &franka::RobotState::F_T_EE, "Flange to EE transform (16,)")
      .def_readwrite("F_T_NE", &franka::RobotState::F_T_NE,
                     "Flange to nominal end effector transform (16,)")
      .def_readwrite("NE_T_EE", &franka::RobotState::NE_T_EE,
                     "Nominal EE to EE transform (16,)")
      .def_readwrite("EE_T_K", &franka::RobotState::EE_T_K,
                     "EE to stiffness frame transform (16,)")
      .def_readwrite("m_ee", &franka::RobotState::m_ee, "End effector mass [kg]")
      .def_readwrite("I_ee", &franka::RobotState::I_ee, "End effector inertia (9,)")
      .def_readwrite("F_x_Cee", &franka::RobotState::F_x_Cee, "EE center of mass in flange (3,)")
      .def_readwrite("m_load", &franka::RobotState::m_load, "External load mass [kg]")
      .def_readwrite("I_load", &franka::RobotState::I_load, "External load inertia (9,)")
      .def_readwrite("F_x_Cload", &franka::RobotState::F_x_Cload,
                     "Load center of mass in flange (3,)")
      .def_readwrite("m_total", &franka::RobotState::m_total, "Total mass [kg]")
      .def_readwrite("I_total", &franka::RobotState::I_total, "Total inertia (9,)")
      .def_readwrite("F_x_Ctotal", &franka::RobotState::F_x_Ctotal,
                     "Total center of mass in flange (3,)")
      .def_readwrite("elbow", &franka::RobotState::elbow, "Elbow configuration (2,)")
      .def_readwrite("elbow_d", &franka::RobotState::elbow_d, "Desired elbow configuration (2,)")
      .def_readwrite("elbow_c", &franka::RobotState::elbow_c, "Commanded elbow configuration (2,)")
      .def_readwrite("delbow_c", &franka::RobotState::delbow_c, "Commanded elbow velocity (2,)")
      .def_readwrite("ddelbow_c", &franka::RobotState::ddelbow_c,
                     "Commanded elbow acceleration (2,)")
      .def_readwrite("joint_contact", &franka::RobotState::joint_contact,
                     "Joint contact flags (7,)")
      .def_readwrite("cartesian_contact", &franka::RobotState::cartesian_contact,
                     "Cartesian contact flags (6,)")
      .def_readwrite("joint_collision", &franka::RobotState::joint_collision,
                     "Joint collision flags (7,)")
      .def_readwrite("cartesian_collision", &franka::RobotState::cartesian_collision,
                     "Cartesian collision flags (6,)")
      .def_readwrite("O_F_ext_hat_K", &franka::RobotState::O_F_ext_hat_K,
                     "External wrench in base frame (6,)")
      .def_readwrite("K_F_ext_hat_K", &franka::RobotState::K_F_ext_hat_K,
                     "External wrench in stiffness frame (6,)")
      .def_readwrite("O_dP_EE_d", &franka::RobotState::O_dP_EE_d,
                     "Desired EE twist in base (6,)")
      .def_readwrite("O_dP_EE_c", &franka::RobotState::O_dP_EE_c,
                     "Commanded EE twist in base (6,)")
      .def_readwrite("O_ddP_EE_c", &franka::RobotState::O_ddP_EE_c,
                     "Commanded EE acceleration (6,)")
      .def_readwrite("current_errors", &franka::RobotState::current_errors, "Current errors")
      .def_readwrite("last_motion_errors", &franka::RobotState::last_motion_errors,
                     "Last motion errors")
      .def_readwrite("control_command_success_rate",
                     &franka::RobotState::control_command_success_rate,
                     "Control command success rate [0, 1]")
      .def_readwrite("robot_mode", &franka::RobotState::robot_mode, "Current robot mode")
      .def_readwrite("time", &franka::RobotState::time, "Time since robot start");

  // Bind control types
  py::class_<franka::Torques>(m, "Torques", R"pbdoc(
        Torque control command.
    )pbdoc")
      .def(py::init<const std::array<double, 7>&>(), py::arg("tau_J"), R"pbdoc(
        Create torque command.

        @param tau_J Joint torques [Nm] (7,)
    )pbdoc")
      .def_readwrite("tau_J", &franka::Torques::tau_J, "Joint torques [Nm] (7,)")
      .def_readwrite("motion_finished", &franka::Torques::motion_finished,
                     "Set to True to finish motion")
      .def_static(
          "finished",
          [](const std::array<double, 7>& tau_J) {
            franka::Torques t(tau_J);
            t.motion_finished = true;
            return t;
          },
          py::arg("tau_J"), "Create finished torque command");

  py::class_<franka::JointPositions>(m, "JointPositions", R"pbdoc(
        Joint position control command.
    )pbdoc")
      .def(py::init<const std::array<double, 7>&>(), py::arg("q"), R"pbdoc(
        Create joint position command.

        @param q Joint positions [rad] (7,)
    )pbdoc")
      .def_readwrite("q", &franka::JointPositions::q, "Joint positions [rad] (7,)")
      .def_readwrite("motion_finished", &franka::JointPositions::motion_finished,
                     "Set to True to finish motion")
      .def_static(
          "finished",
          [](const std::array<double, 7>& q) {
            franka::JointPositions jp(q);
            jp.motion_finished = true;
            return jp;
          },
          py::arg("q"), "Create finished joint position command");

  py::class_<franka::JointVelocities>(m, "JointVelocities", R"pbdoc(
        Joint velocity control command.
    )pbdoc")
      .def(py::init<const std::array<double, 7>&>(), py::arg("dq"), R"pbdoc(
        Create joint velocity command.

        @param dq Joint velocities [rad/s] (7,)
    )pbdoc")
      .def_readwrite("dq", &franka::JointVelocities::dq, "Joint velocities [rad/s] (7,)")
      .def_readwrite("motion_finished", &franka::JointVelocities::motion_finished,
                     "Set to True to finish motion")
      .def_static(
          "finished",
          [](const std::array<double, 7>& dq) {
            franka::JointVelocities jv(dq);
            jv.motion_finished = true;
            return jv;
          },
          py::arg("dq"), "Create finished joint velocity command");

  py::class_<franka::CartesianPose>(m, "CartesianPose", R"pbdoc(
        Cartesian pose control command.
    )pbdoc")
      .def(py::init<const std::array<double, 16>&>(), py::arg("O_T_EE"), R"pbdoc(
        Create Cartesian pose command.

        @param O_T_EE Homogeneous transformation matrix (16,) in column-major order
    )pbdoc")
      .def(py::init<const std::array<double, 16>&, const std::array<double, 2>&>(),
           py::arg("O_T_EE"), py::arg("elbow"), R"pbdoc(
        Create Cartesian pose command with elbow configuration.

        @param O_T_EE Homogeneous transformation matrix (16,) in column-major order
        @param elbow Elbow configuration (2,)
    )pbdoc")
      .def_readwrite("O_T_EE", &franka::CartesianPose::O_T_EE,
                     "End effector pose (16,) column-major")
      .def_readwrite("elbow", &franka::CartesianPose::elbow, "Elbow configuration (2,)")
      .def_readwrite("motion_finished", &franka::CartesianPose::motion_finished,
                     "Set to True to finish motion")
      .def_static(
          "finished",
          [](const std::array<double, 16>& O_T_EE) {
            franka::CartesianPose cp(O_T_EE);
            cp.motion_finished = true;
            return cp;
          },
          py::arg("O_T_EE"), "Create finished Cartesian pose command");

  py::class_<franka::CartesianVelocities>(m, "CartesianVelocities", R"pbdoc(
        Cartesian velocity control command.
    )pbdoc")
      .def(py::init<const std::array<double, 6>&>(), py::arg("O_dP_EE"), R"pbdoc(
        Create Cartesian velocity command.

        @param O_dP_EE End effector twist [vx, vy, vz, wx, wy, wz] in [m/s, rad/s] (6,)
    )pbdoc")
      .def(py::init<const std::array<double, 6>&, const std::array<double, 2>&>(),
           py::arg("O_dP_EE"), py::arg("elbow"), R"pbdoc(
        Create Cartesian velocity command with elbow configuration.

        @param O_dP_EE End effector twist (6,)
        @param elbow Elbow configuration (2,)
    )pbdoc")
      .def_readwrite("O_dP_EE", &franka::CartesianVelocities::O_dP_EE,
                     "End effector twist [m/s, rad/s] (6,)")
      .def_readwrite("elbow", &franka::CartesianVelocities::elbow, "Elbow configuration (2,)")
      .def_readwrite("motion_finished", &franka::CartesianVelocities::motion_finished,
                     "Set to True to finish motion")
      .def_static(
          "finished",
          [](const std::array<double, 6>& O_dP_EE) {
            franka::CartesianVelocities cv(O_dP_EE);
            cv.motion_finished = true;
            return cv;
          },
          py::arg("O_dP_EE"), "Create finished Cartesian velocity command");

  // Bind franka::Model
  py::class_<franka::Model>(m, "Model", R"pbdoc(
        Robot dynamics model for computing dynamics quantities.
    )pbdoc")
      .def(
          "coriolis",
          [](franka::Model& self, const franka::RobotState& robot_state) {
            return self.coriolis(robot_state);
          },
          py::arg("robot_state"), R"pbdoc(
        Compute Coriolis force vector.

        @param robot_state Current robot state
        @return Coriolis force vector [Nm] as numpy array (7,)
    )pbdoc")
      .def(
          "gravity",
          [](franka::Model& self, const franka::RobotState& robot_state,
             const std::array<double, 3>& gravity_earth) {
            return self.gravity(robot_state, gravity_earth);
          },
          py::arg("robot_state"),
          py::arg("gravity_earth") = std::array<double, 3>{{0., 0., -9.81}}, R"pbdoc(
        Compute gravity force vector.

        @param robot_state Current robot state
        @param gravity_earth Gravity vector in base frame [m/s²] (default: [0, 0, -9.81])
        @return Gravity force vector [Nm] as numpy array (7,)
    )pbdoc")
      .def(
          "mass",
          [](franka::Model& self, const franka::RobotState& robot_state) {
            return self.mass(robot_state);
          },
          py::arg("robot_state"), R"pbdoc(
        Compute mass matrix.

        @param robot_state Current robot state
        @return Mass matrix [kg*m²] as numpy array (7, 7)
    )pbdoc")
      .def(
          "pose",
          [](franka::Model& self, franka::Frame frame, const franka::RobotState& robot_state) {
            return self.pose(frame, robot_state);
          },
          py::arg("frame"), py::arg("robot_state"), R"pbdoc(
        Compute pose of specified frame.

        @param frame Frame to compute pose for
        @param robot_state Current robot state
        @return Homogeneous transformation matrix (16,) in column-major order
    )pbdoc")
      .def(
          "body_jacobian",
          [](franka::Model& self, franka::Frame frame, const franka::RobotState& robot_state) {
            return self.bodyJacobian(frame, robot_state);
          },
          py::arg("frame"), py::arg("robot_state"), R"pbdoc(
        Compute body Jacobian for specified frame.

        @param frame Frame to compute Jacobian for
        @param robot_state Current robot state
        @return Body Jacobian matrix as numpy array (6, 7)
    )pbdoc")
      .def(
          "zero_jacobian",
          [](franka::Model& self, franka::Frame frame, const franka::RobotState& robot_state) {
            return self.zeroJacobian(frame, robot_state);
          },
          py::arg("frame"), py::arg("robot_state"), R"pbdoc(
        Compute zero Jacobian for specified frame.

        @param frame Frame to compute Jacobian for
        @param robot_state Current robot state
        @return Zero Jacobian matrix as numpy array (6, 7)
    )pbdoc");

  // Bind Robot
  py::class_<franka::Robot>(m, "Robot", R"pbdoc(
        Main interface for controlling a Franka robot.

        Provides real-time control capabilities including torque, position,
        and velocity control modes using callback functions.
    )pbdoc")
      .def(py::init<const std::string&, franka::RealtimeConfig, size_t>(),
           py::arg("franka_address"),
           py::arg("realtime_config") = franka::RealtimeConfig::kEnforce,
           py::arg("log_size") = 50, R"pbdoc(
        Connect to a Franka robot.

        @param franka_address IP address or hostname of the robot
        @param realtime_config Real-time scheduling requirements (default: kEnforce)
        @param log_size Number of states to keep for logging (default: 50)
    )pbdoc")
      .def("read_once", &franka::Robot::readOnce, R"pbdoc(
        Read current robot state once.

        @return Current robot state
    )pbdoc")
      .def(
          "read",
          [](franka::Robot& self,
             std::function<bool(const franka::RobotState&)> read_callback) {
            py::gil_scoped_release release;
            self.read(read_callback);
          },
          py::arg("read_callback"), R"pbdoc(
        Start reading robot state in a loop.

        The callback function is called for each robot state update.
        Return False from the callback to stop reading.

        @param read_callback Callback function receiving RobotState, returns bool
    )pbdoc")
      .def(
          "control_torques",
          [](franka::Robot& self,
             std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
                 control_callback,
             bool limit_rate, double cutoff_frequency) {
            py::gil_scoped_release release;
            self.control(control_callback, limit_rate, cutoff_frequency);
          },
          py::arg("control_callback"), py::arg("limit_rate") = true,
          py::arg("cutoff_frequency") = franka::kDefaultCutoffFrequency, R"pbdoc(
        Start torque control loop.

        The callback function computes torque commands for each control cycle (1 kHz).
        Set motion_finished=True in the returned Torques to stop.

        @param control_callback Callback (RobotState, Duration) -> Torques
        @param limit_rate Enable rate limiting (default: True)
        @param cutoff_frequency Low-pass filter cutoff [Hz] (default: 100)
    )pbdoc")
      .def(
          "control_joint_positions",
          [](franka::Robot& self,
             std::function<franka::JointPositions(const franka::RobotState&, franka::Duration)>
                 motion_callback,
             franka::ControllerMode controller_mode, bool limit_rate, double cutoff_frequency) {
            py::gil_scoped_release release;
            self.control(motion_callback, controller_mode, limit_rate, cutoff_frequency);
          },
          py::arg("motion_callback"),
          py::arg("controller_mode") = franka::ControllerMode::kJointImpedance,
          py::arg("limit_rate") = true,
          py::arg("cutoff_frequency") = franka::kDefaultCutoffFrequency, R"pbdoc(
        Start joint position control loop.

        The callback function computes joint position commands for each control cycle (1 kHz).
        Set motion_finished=True in the returned JointPositions to stop.

        @param motion_callback Callback (RobotState, Duration) -> JointPositions
        @param controller_mode Controller mode (default: JointImpedance)
        @param limit_rate Enable rate limiting (default: True)
        @param cutoff_frequency Low-pass filter cutoff [Hz] (default: 100)
    )pbdoc")
      .def(
          "control_joint_velocities",
          [](franka::Robot& self,
             std::function<franka::JointVelocities(const franka::RobotState&, franka::Duration)>
                 motion_callback,
             franka::ControllerMode controller_mode, bool limit_rate, double cutoff_frequency) {
            py::gil_scoped_release release;
            self.control(motion_callback, controller_mode, limit_rate, cutoff_frequency);
          },
          py::arg("motion_callback"),
          py::arg("controller_mode") = franka::ControllerMode::kJointImpedance,
          py::arg("limit_rate") = true,
          py::arg("cutoff_frequency") = franka::kDefaultCutoffFrequency, R"pbdoc(
        Start joint velocity control loop.

        The callback function computes joint velocity commands for each control cycle (1 kHz).
        Set motion_finished=True in the returned JointVelocities to stop.

        @param motion_callback Callback (RobotState, Duration) -> JointVelocities
        @param controller_mode Controller mode (default: JointImpedance)
        @param limit_rate Enable rate limiting (default: True)
        @param cutoff_frequency Low-pass filter cutoff [Hz] (default: 100)
    )pbdoc")
      .def(
          "control_cartesian_pose",
          [](franka::Robot& self,
             std::function<franka::CartesianPose(const franka::RobotState&, franka::Duration)>
                 motion_callback,
             franka::ControllerMode controller_mode, bool limit_rate, double cutoff_frequency) {
            py::gil_scoped_release release;
            self.control(motion_callback, controller_mode, limit_rate, cutoff_frequency);
          },
          py::arg("motion_callback"),
          py::arg("controller_mode") = franka::ControllerMode::kJointImpedance,
          py::arg("limit_rate") = true,
          py::arg("cutoff_frequency") = franka::kDefaultCutoffFrequency, R"pbdoc(
        Start Cartesian pose control loop.

        The callback function computes Cartesian pose commands for each control cycle (1 kHz).
        Set motion_finished=True in the returned CartesianPose to stop.

        @param motion_callback Callback (RobotState, Duration) -> CartesianPose
        @param controller_mode Controller mode (default: JointImpedance)
        @param limit_rate Enable rate limiting (default: True)
        @param cutoff_frequency Low-pass filter cutoff [Hz] (default: 100)
    )pbdoc")
      .def(
          "control_cartesian_velocities",
          [](franka::Robot& self,
             std::function<franka::CartesianVelocities(const franka::RobotState&, franka::Duration)>
                 motion_callback,
             franka::ControllerMode controller_mode, bool limit_rate, double cutoff_frequency) {
            py::gil_scoped_release release;
            self.control(motion_callback, controller_mode, limit_rate, cutoff_frequency);
          },
          py::arg("motion_callback"),
          py::arg("controller_mode") = franka::ControllerMode::kJointImpedance,
          py::arg("limit_rate") = true,
          py::arg("cutoff_frequency") = franka::kDefaultCutoffFrequency, R"pbdoc(
        Start Cartesian velocity control loop.

        The callback function computes Cartesian velocity commands for each control cycle (1 kHz).
        Set motion_finished=True in the returned CartesianVelocities to stop.

        @param motion_callback Callback (RobotState, Duration) -> CartesianVelocities
        @param controller_mode Controller mode (default: JointImpedance)
        @param limit_rate Enable rate limiting (default: True)
        @param cutoff_frequency Low-pass filter cutoff [Hz] (default: 100)
    )pbdoc")
      .def("set_collision_behavior",
           py::overload_cast<const std::array<double, 7>&, const std::array<double, 7>&,
                             const std::array<double, 6>&, const std::array<double, 6>&>(
               &franka::Robot::setCollisionBehavior),
           py::arg("lower_torque_thresholds"), py::arg("upper_torque_thresholds"),
           py::arg("lower_force_thresholds"), py::arg("upper_force_thresholds"),
           R"pbdoc(
        Configure collision detection thresholds.

        @param lower_torque_thresholds Lower torque thresholds [Nm] (7,)
        @param upper_torque_thresholds Upper torque thresholds [Nm] (7,)
        @param lower_force_thresholds Lower Cartesian force thresholds [N, Nm] (6,)
        @param upper_force_thresholds Upper Cartesian force thresholds [N, Nm] (6,)
    )pbdoc")
      .def("set_collision_behavior_full",
           py::overload_cast<const std::array<double, 7>&, const std::array<double, 7>&,
                             const std::array<double, 7>&, const std::array<double, 7>&,
                             const std::array<double, 6>&, const std::array<double, 6>&,
                             const std::array<double, 6>&, const std::array<double, 6>&>(
               &franka::Robot::setCollisionBehavior),
           py::arg("lower_torque_thresholds_acceleration"),
           py::arg("upper_torque_thresholds_acceleration"),
           py::arg("lower_torque_thresholds_nominal"),
           py::arg("upper_torque_thresholds_nominal"),
           py::arg("lower_force_thresholds_acceleration"),
           py::arg("upper_force_thresholds_acceleration"),
           py::arg("lower_force_thresholds_nominal"), py::arg("upper_force_thresholds_nominal"),
           R"pbdoc(
        Configure collision detection thresholds with separate acceleration/nominal phases.

        @param lower_torque_thresholds_acceleration Lower torque thresholds during acceleration [Nm] (7,)
        @param upper_torque_thresholds_acceleration Upper torque thresholds during acceleration [Nm] (7,)
        @param lower_torque_thresholds_nominal Lower torque thresholds during nominal motion [Nm] (7,)
        @param upper_torque_thresholds_nominal Upper torque thresholds during nominal motion [Nm] (7,)
        @param lower_force_thresholds_acceleration Lower force thresholds during acceleration (6,)
        @param upper_force_thresholds_acceleration Upper force thresholds during acceleration (6,)
        @param lower_force_thresholds_nominal Lower force thresholds during nominal motion (6,)
        @param upper_force_thresholds_nominal Upper force thresholds during nominal motion (6,)
    )pbdoc")
      .def("set_joint_impedance", &franka::Robot::setJointImpedance, py::arg("K_theta"), R"pbdoc(
        Set joint impedance for internal controller.

        @param K_theta Joint stiffness values [Nm/rad] (7,), range [0, 14250]
    )pbdoc")
      .def("set_cartesian_impedance", &franka::Robot::setCartesianImpedance, py::arg("K_x"),
           R"pbdoc(
        Set Cartesian impedance for internal controller.

        @param K_x Cartesian stiffness values [N/m, Nm/rad] (6,)
    )pbdoc")
      .def("set_guiding_mode", &franka::Robot::setGuidingMode, py::arg("guiding_mode"),
           py::arg("elbow"), R"pbdoc(
        Set guiding mode movement freedoms.

        @param guiding_mode Unlocked movement in (x, y, z, R, P, Y) (6,)
        @param elbow True if elbow is free in guiding mode
    )pbdoc")
      .def("set_K", &franka::Robot::setK, py::arg("EE_T_K"), R"pbdoc(
        Set stiffness frame K in end effector frame.

        @param EE_T_K Homogeneous transformation matrix (16,) in column-major order
    )pbdoc")
      .def("set_EE", &franka::Robot::setEE, py::arg("NE_T_EE"), R"pbdoc(
        Set end effector frame relative to nominal end effector frame.

        @param NE_T_EE Homogeneous transformation matrix (16,) in column-major order
    )pbdoc")
      .def("set_load", &franka::Robot::setLoad, py::arg("load_mass"), py::arg("F_x_Cload"),
           py::arg("load_inertia"), R"pbdoc(
        Set external load parameters.

        @param load_mass Mass of the external load [kg]
        @param F_x_Cload Center of mass of load in flange frame [m] (3,)
        @param load_inertia Inertia tensor of load [kg*m²] (9,) in column-major order
    )pbdoc")
      .def("automatic_error_recovery", &franka::Robot::automaticErrorRecovery, R"pbdoc(
        Attempt automatic error recovery.

        Tries to recover from robot errors automatically.
    )pbdoc")
      .def("stop", &franka::Robot::stop, R"pbdoc(
        Stop currently running motion.
    )pbdoc")
      .def("load_model", py::overload_cast<>(&franka::Robot::loadModel), R"pbdoc(
        Load robot dynamics model.

        Downloads and loads the robot's dynamics model from the robot.

        @return Model object for computing dynamics quantities
    )pbdoc")
      .def("server_version", &franka::Robot::serverVersion, R"pbdoc(
        Get robot server version.

        @return Server version number
    )pbdoc");

  // Bind GripperState
  py::class_<franka::GripperState>(m, "GripperState", R"pbdoc(
        Current state of the Franka Hand gripper.
    )pbdoc")
      .def_readwrite("width", &franka::GripperState::width, "Current gripper width [m]")
      .def_readwrite("max_width", &franka::GripperState::max_width, "Maximum gripper width [m]")
      .def_readwrite("is_grasped", &franka::GripperState::is_grasped, "True if object is grasped")
      .def_readwrite("temperature", &franka::GripperState::temperature, "Gripper temperature [°C]")
      .def_readwrite("time", &franka::GripperState::time, "Timestamp");

  // Bind Gripper
  py::class_<franka::Gripper>(m, "Gripper", R"pbdoc(
        Interface for controlling a Franka Hand gripper.
    )pbdoc")
      .def(py::init<const std::string&>(), py::arg("franka_address"), R"pbdoc(
        Connect to gripper.

        Establishes connection to the Franka Hand gripper at the specified address.

        @param franka_address IP address or hostname of the robot
    )pbdoc")
      .def("homing", &franka::Gripper::homing, R"pbdoc(
        Perform homing to find maximum width.

        @return True if successful
    )pbdoc")
      .def("grasp", &franka::Gripper::grasp, py::arg("width"), py::arg("speed"), py::arg("force"),
           py::arg("epsilon_inner") = 0.005, py::arg("epsilon_outer") = 0.005, R"pbdoc(
        Grasp an object.

        @param width Target grasp width [m]
        @param speed Closing speed [m/s]
        @param force Grasping force [N]
        @param epsilon_inner Inner tolerance for grasp check [m] (default: 0.005)
        @param epsilon_outer Outer tolerance for grasp check [m] (default: 0.005)
        @return True if grasp successful
    )pbdoc")
      .def("move", &franka::Gripper::move, py::arg("width"), py::arg("speed"), R"pbdoc(
        Move gripper fingers to a specific width.

        @param width Target width [m]
        @param speed Movement speed [m/s]
        @return True if successful
    )pbdoc")
      .def("stop", &franka::Gripper::stop, R"pbdoc(
        Stop current gripper motion.

        @return True if successful
    )pbdoc")
      .def("read_once", &franka::Gripper::readOnce, R"pbdoc(
        Read current gripper state once.

        @return Current gripper state
    )pbdoc")
      .def("server_version", &franka::Gripper::serverVersion, R"pbdoc(
        Get gripper server version.

        @return Server version number
    )pbdoc");

  // Module-level constants
  m.attr("kDefaultCutoffFrequency") = franka::kDefaultCutoffFrequency;
  m.attr("kMaxCutoffFrequency") = franka::kMaxCutoffFrequency;
}

}  // namespace pylibfranka
