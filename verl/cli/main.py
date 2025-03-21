import sys
import os
import argparse
from omegaconf import OmegaConf


def train_ppo(config_path: str, backend: str):
    """Run PPO/GRPO training with the given config file."""
    package_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    if backend == "fsdp":
        base_config_path = os.path.join(
            package_root, "verl/trainer/config/ppo_trainer.yaml"
        )
    elif backend == "megatron":
        base_config_path = os.path.join(
            package_root, "verl/trainer/config/ppo_megatron_trainer.yaml"
        )
    else:
        print(f"Unknown backend: {backend}.")
        sys.exit(1)
    print(f"Using base config: {base_config_path}")
    # Load base config
    config = OmegaConf.load(base_config_path)

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    user_config = OmegaConf.load(config_path)
    config = OmegaConf.merge(config, user_config)

    from verl.trainer.main_ppo import main as main_ppo

    main_ppo(config)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="VERL command line interface")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train PPO subcommand
    train_ppo_parser = subparsers.add_parser(
        "train-ppo", help="Train using PPO/GRPO algorithm"
    )
    train_ppo_parser.add_argument("config_path", help="Path to the configuration file")
    train_ppo_parser.add_argument(
        "--backend",
        choices=["fsdp", "megatron"],
        default="fsdp",
        help="Backend to use (fsdp or megatron)",
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "train-ppo":
        train_ppo(args.config_path, args.backend)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
