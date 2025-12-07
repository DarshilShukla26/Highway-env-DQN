import argparse
import os
import numpy as np
import yaml

import gym
import highway_env

from envs.highway_wrapper import make_env
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent
from agents.dueling_dqn_agent import DuelingDQNAgent
from agents.per_dqn_agent import PERDQNAgent
from agents.noisy_dqn_agent import NoisyDQNAgent
from training.trainer import Trainer
from evaluations.evaluate import evaluate
from evaluations.plot_rewards import plot_rewards


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_agent(name: str, env, config):
    if name == "DQN":
        return DQNAgent(env, config)
    if name == "DoubleDQN":
        return DoubleDQNAgent(env, config)
    if name == "DuelingDQN":
        return DuelingDQNAgent(env, config)
    if name == "PERDQN":
        return PERDQNAgent(env, config)
    if name == "NoisyDQN":
        return NoisyDQNAgent(env, config)
    raise ValueError(f"Unknown agent type: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dqn.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create environment (MORL-supporting)
    env = make_env(config["env"], config)

    # Build agent
    agent = make_agent(config["agent"], env, config)

    # Train
    trainer = Trainer(env, agent, config)
    returns, components = trainer.train()     # <--- IMPORTANT FIX

    # Determine save name
    # policy_name is used ONLY for MORL configs (fast, safe, lane)
    policy_name = config.get("policy_name", config["agent"])

    # Make directories if needed
    os.makedirs("results/plots", exist_ok=True)

    # Plot training curve
    plot_path = os.path.join("results", "plots", f"{policy_name}_train_curve.png")
    plot_rewards(
        returns,
        plot_path,
        title=f"{policy_name} Training Returns"
    )

    # Save training results
    np.save(f"results/MORL_{policy_name}.npy", returns)
    np.save(f"results/MORL_{policy_name}_components.npy", components)

    # Evaluate (greedy)
    scores = evaluate(env, agent, episodes=args.eval_episodes)
    mean_score = sum(scores) / len(scores)

    print(f"\n===== Evaluation Results ({policy_name}) =====")
    print(f"Mean Score: {mean_score:.3f}")
    print(f"Scores: {scores}\n")


if __name__ == "__main__":
    main()
