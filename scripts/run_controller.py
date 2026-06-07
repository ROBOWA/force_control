"""Main entry point for force-control experiments.

Run from the repo root (force_control/):

    # MuJoCo simulation (Mac)
    python -m scripts.run_controller --backend sim --config configs/sim.yaml

    # Real robot, dry-run (imports + config check, no motion)
    python -m scripts.run_controller --backend real --config configs/real.yaml --dry-run

    # Real robot, execute (prompts for confirmation)
    python -m scripts.run_controller --backend real --config configs/real.yaml --execute

    # Offline replay from a CSV log
    python -m scripts.run_controller --backend replay --log data/logs/example.csv
"""

from __future__ import annotations
import argparse
import sys
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_sim(config: dict) -> None:
    """Launch the MuJoCo backend (Mac)."""
    from backends.mujoco_backend import MuJoCoBackend
    xml = config.get("mjcf", "franka_emika_panda/scene.xml")
    backend = MuJoCoBackend(xml, config)
    backend.load()
    backend.run()


def run_real(config: dict, dry_run: bool) -> None:
    """Launch the Franka backend (Ubuntu Panda PC)."""
    from backends.franka_backend import FrankaBackend
    robot_ip = config["robot_ip"]
    backend = FrankaBackend(robot_ip, config)
    backend.connect()

    if dry_run:
        print("[DRY RUN] Connection OK. Config loaded. No motion commanded.")
        return

    ans = input("Confirm real-robot execution? (y/N): ").strip().lower()
    if ans != "y":
        print("Cancelled.")
        return

    backend.start_ft_thread()
    backend.start_outer_loop()
    try:
        backend.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        backend.stop()


def run_replay(log_path: str) -> None:
    """Replay a CSV log through the controller core (no robot, no sim)."""
    from force_control.sensors.ft_replay import FTReplaySource
    # TODO: implement replay loop
    # source = FTReplaySource(log_path)
    # source.load()
    # ...
    raise NotImplementedError("Replay backend not yet implemented.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Force-control experiment runner")
    parser.add_argument(
        "--backend", choices=["sim", "real", "replay"], required=True,
        help="Control backend to use",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (required for sim/real)",
    )
    parser.add_argument(
        "--log", type=str, default=None,
        help="Path to CSV log (required for replay backend)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Check imports and config without commanding motion (real only)",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Alias for not --dry-run; prompts before real motion",
    )
    args = parser.parse_args()

    if args.backend == "sim":
        if args.config is None:
            parser.error("--config is required for --backend sim")
        run_sim(load_config(args.config))

    elif args.backend == "real":
        if args.config is None:
            parser.error("--config is required for --backend real")
        dry_run = args.dry_run or (not args.execute)
        run_real(load_config(args.config), dry_run=dry_run)

    elif args.backend == "replay":
        if args.log is None:
            parser.error("--log is required for --backend replay")
        run_replay(args.log)


if __name__ == "__main__":
    main()
