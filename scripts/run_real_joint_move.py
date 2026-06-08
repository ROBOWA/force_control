"""Runner for Milestone 1 on the real Franka hardware.

Startup order:
    1. connect()               — TCP to robot, collision thresholds, read initial q
    2. start_ft_source()       — ROS subscriber started before control loop
    3. initialize_core_pipeline() — IK solve (synchronous), Enter-waiter thread
    4. user confirmation prompt
    5. run()                   — blocks inside robot.control_torques()

Run from the repo root:

    python -m scripts.run_real_joint_move --config configs/real.yaml
"""

from __future__ import annotations
import argparse
import yaml

from backends.franka_backend import FrankaBackend


def main() -> None:
    parser = argparse.ArgumentParser(description="Real Franka joint-move (Milestone 1)")
    parser.add_argument("--config", default="configs/real.yaml",
                        help="Path to real robot YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    robot_ip = cfg["robot_ip"]
    backend  = FrankaBackend(robot_ip, cfg)

    backend.connect()
    backend.start_ft_source()
    backend.initialize_core_pipeline()

    ans = input("\nAll systems ready. Confirm real-robot execution? (y/N): ").strip().lower()
    if ans != "y":
        print("Cancelled.")
        backend.stop()
        return

    try:
        backend.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        backend.stop()


if __name__ == "__main__":
    main()
