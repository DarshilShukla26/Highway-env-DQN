import argparse
import numpy as np
import yaml

from envs.highway_wrapper import make_env
from agents.dqn_agent import DQNAgent
from training.trainer_components import TrainerComponents

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    env = make_env(config["env"], config)

    agent = DQNAgent(env, config)
    trainer = TrainerComponents(env, agent, config)

    returns, components = trainer.train()

    np.save(f"results/{config['name']}.npy", returns)
    np.save(f"results/{config['name']}_components.npy", components)

    print("Training complete:", config["name"])

if __name__ == "__main__":
    main()
