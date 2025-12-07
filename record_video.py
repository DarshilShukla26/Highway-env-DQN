import gymnasium as gym
import highway_env
import torch
import numpy as np
import imageio
import argparse
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent


def load_agent(agent_type, env, model_path, config):
    if agent_type == "DQN":
        agent = DQNAgent(env, config)
    elif agent_type == "DoubleDQN":
        agent = DoubleDQNAgent(env, config)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    agent.online.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.online.eval()
    return agent


def record_agent(model_path, config, agent_type, output_path):
    # Use supported env
    env = gym.make("highway-v0", render_mode="rgb_array")

    agent = load_agent(agent_type, env, model_path, config)

    frames = []
    state, _ = env.reset()

    for _ in range(300):
        action = agent.act_greedy(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        frame = env.render()
        frames.append(frame)

        state = next_state

        if terminated or truncated:
            break

    imageio.mimsave(output_path + ".mp4", frames, fps=30)
    print(f"Saved video → {output_path}.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--agent", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    record_agent(args.model, config, args.agent, args.output)
