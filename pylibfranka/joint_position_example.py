#!/usr/bin/env python3

import argparse
import numpy as np
from pylibfranka import Robot, JointPositions, ControllerMode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="172.16.0.2", help="Robot IP address")
    args = parser.parse_args()

    # The C++ code shows the constructor takes (address, realtime_config, log_size)
    robot = Robot(args.ip)

    # Note: set_collision_behavior wasn't in the snippet but is standard in libfranka
    # If it fails, you can comment it out for now.
    try:
        robot.set_collision_behavior(
            [20.0]*7, [20.0]*7, [20.0]*6, [20.0]*6
        )
    except AttributeError:
        print("set_collision_behavior not found, skipping...")

    print("WARNING: This example will move the robot!")
    input("Press Enter to continue...")

    # State container to track progress within the callback
    state_vars = {
        "initial_q": None,
        "time_elapsed": 0.0
    }

    def motion_callback(robot_state, duration):
        # The C++ Duration binding provides .to_sec()
        state_vars["time_elapsed"] += duration.to_sec()
        
        if state_vars["initial_q"] is None:
            state_vars["initial_q"] = robot_state.q_d

        # Calculate the delta angle
        delta_angle = (np.pi / 8.0) * (1 - np.cos(np.pi / 2.5 * state_vars["time_elapsed"]))

        new_q = list(state_vars["initial_q"])
        new_q[3] += delta_angle
        new_q[4] += delta_angle
        new_q[6] += delta_angle

        # Create the command object
        command = JointPositions(new_q)

        # The C++ code shows JointPositions has a .motion_finished property
        if state_vars["time_elapsed"] >= 5.0:
            print("Motion finished.")
            command.motion_finished = True

        return command

    try:
        # Per your C++ snippet, the method name is 'control_joint_positions'
        # Arguments: (callback, controller_mode, limit_rate, cutoff_frequency)
        robot.control_joint_positions(
            motion_callback, 
            ControllerMode.CartesianImpedance
        )
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
