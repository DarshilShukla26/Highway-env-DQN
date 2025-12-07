import gymnasium as gym
import highway_env
import numpy as np
import torch
import argparse

from agents.dqn_agent import DQNAgent  # or your MORL agent class
from envs.highway_wrapper import make_env

def load_agent(model_path, env, config):
    agent = DQNAgent(env, config)
    agent.online.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.online.eval()
    return agent

def run_episode(env, agent, render=True):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        if render:
            env.render()

        action = agent.act_greedy(state)  # deterministic action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward

    print("Episode return:", total_reward)
    env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    env = make_env(config["env"], config)

    agent = load_agent(args.model, env, config)
    run_episode(env, agent, render=True)

if __name__ == "__main__":
    main()
