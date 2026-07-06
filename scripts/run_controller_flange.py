"""Entry point for the FLANGE force-control pipeline (Ubuntu Panda PC).

Same experiment as scripts/run_controller.py --backend real, but the contact
wrench comes from the Panda's own flange F_ext estimate read in the 1 kHz loop
(backends.franka_backend_flange.FrankaBackendFlange) — no external FT sensor,
no ROS, no shared memory.

Run from the repo's **parent** directory (one level above force_control/):

    # Dry run (connect + config check, no motion)
    python -m force_control.scripts.run_controller_flange \\
        --config force_control/configs/real_flange.yaml --dry-run

    # Execute (prompts for confirmation before motion)
    python -m force_control.scripts.run_controller_flange \\
        --config force_control/configs/real_flange.yaml --execute

Before trusting the force phases, sanity-check the sign — the response should be
right-handed: push the tip UP -> +Fz, +X -> +Fx, +Y -> +Fy (a freely hanging
tool -> -Fz). +Fz is the compression sign the state machine expects on contact.
libfranka's F_ext global sign varies by setup: if EVERY axis reads opposite to
your push, set ft.sign: -1.0 (fixes the global flip and the contact sign). Use a
per-axis ft.sign like [1,1,-1,1,1,1] only for a single reversed axis.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_real_flange(config: dict, dry_run: bool) -> None:
    """Launch the flange Franka backend."""
    from backends.franka_backend_flange import FrankaBackendFlange

    robot_ip = config["robot_ip"]
    backend = FrankaBackendFlange(robot_ip, config)
    backend.connect()

    if dry_run:
        print("[DRY RUN] Connection OK. Config loaded. No motion commanded.")
        return

    ans = input("Confirm real-robot execution (flange FT)? (y/N): ").strip().lower()
    if ans != "y":
        print("Cancelled.")
        return

    backend.start_ft_source()
    backend.initialize_core_pipeline()
    try:
        backend.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        backend.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Force-control experiment runner (Panda flange FT)"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/real_flange.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Check imports and config without commanding motion",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Alias for not --dry-run; prompts before real motion",
    )
    args = parser.parse_args()

    dry_run = args.dry_run or (not args.execute)
    run_real_flange(load_config(args.config), dry_run=dry_run)


if __name__ == "__main__":
    main()
