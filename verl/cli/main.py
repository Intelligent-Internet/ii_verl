import sys
import os
from omegaconf import OmegaConf
from typing import Optional


def train_ppo(config_path: Optional[str] = None):
    """Run PPO training with the given config file."""
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
        
    config = OmegaConf.load(config_path)
    
    from verl.trainer.main_ppo import main as main_ppo
    main_ppo(config)

def main():
    """Main entry point for the CLI."""
    if len(sys.argv) < 2:
        print("Usage: verl <command> [args]")
        print("\nAvailable commands:")
        print("  train-ppo [config_path] - Train using PPO algorithm")
        sys.exit(1)

    command = sys.argv[1]
    
    if command == "train-ppo":
        config_path = sys.argv[2] if len(sys.argv) > 2 else None
        train_ppo(config_path)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()