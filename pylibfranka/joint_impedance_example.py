#!/usr/bin/env python3

import argparse
import sys
import numpy as np
from pylibfranka import Robot, Torques, Model, Frame

class StateHandler:
    """Tracks the motion progress across callback calls."""
    def __init__(self, targets, stiffness, damping):
        self.targets = targets
        self.stiffness = stiffness
        self.damping = damping
        self.target_index = 0
        self.time_in_step = 0.0
        self.wait_time = 0.5
        self.move_duration = 3.0
        self.current_start_pos = None
        self.motion_finished = False

    def get_target_q(self):
        # Determine current start and end for the segment
        q_start = self.current_start_pos
        q_end = np.array(self.targets[self.target_index])
        
        # Calculate normalized time [0, 1] for minimum jerk
        t = min(self.time_in_step / self.move_duration, 1.0)
        s = 10 * (t**3) - 15 * (t**4) + 6 * (t**5)
        
        return q_start + s * (q_end - q_start)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="172.16.0.2", help="Robot IP address")
    args = parser.parse_args()

    target_joint_positions = [
        [0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0],
        [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0],
        [0.5, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0],
        [-0.5, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0],
        [0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0],
    ]

    stiffness = np.array([50.0] * 7)
    damping = 2.0 * np.sqrt(stiffness)

    try:
        robot = Robot(args.ip)
        if hasattr(robot, "load_model"):
            model = robot.load_model()
        elif hasattr(robot, "get_model"):
            model = robot.get_model()
        else:
            # Debug: print all robot methods to find the model loader
            print("Could not find model loader. Available robot methods:")
            print([m for m in dir(robot) if not m.startswith("_")])
            return -1
        initial_state = robot.read_once()
        handler = StateHandler(target_joint_positions, stiffness, damping)
        handler.current_start_pos = np.array(initial_state.q)

        def torque_callback(robot_state, duration):
            handler.time_in_step += duration.to_sec()
            # print(handler.time_in_step)
            
            # 1. Get targets and robot state
            q_goal = handler.get_target_q()
            q = np.array(robot_state.q)
            dq = np.array(robot_state.dq)
            coriolis = np.array(model.coriolis(robot_state))
            np.array(model.gravity(robot_state))
            np.linalg.inv(np.array(model.mass(robot_state)).reshape((7, 7)))
            np.array(model.pose(Frame.EndEffector, robot_state)).reshape(4, 4)
            # print("gravity :", np.array(model.gravity(robot_state)))
            # print("mass :", np.array(model.mass(robot_state)))
            # print("pose : ", np.array(model.pose(Frame.EndEffector, robot_state)).reshape(4, 4))

            # 2. Compute Impedance Control
            tau_task = -handler.stiffness * (q - q_goal) - handler.damping * dq
            tau_d = tau_task + coriolis

            # 3. Handle Transitions
            step_total_time = handler.move_duration + handler.wait_time
            if handler.time_in_step >= step_total_time:
                handler.target_index += 1
                handler.time_in_step = 0.0
                handler.current_start_pos = q_goal # Update for next segment
                
                # Check if we finished all targets
                if handler.target_index >= len(handler.targets):
                    print("All targets reached. Stopping.")
                    return Torques.finished(tau_d.tolist())

            return Torques(tau_d.tolist())

        print("Starting Joint Impedance Control...")
        # v0.11.0 specific method
        robot.control_torques(torque_callback)

    except Exception as e:
        print(f"Error: {e}")
        return -1

if __name__ == "__main__":
    main()
