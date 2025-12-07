from typing import Dict, List
from tqdm import trange

class TrainerComponents:
    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.cfg = config["train"]

    def train(self):
        episode_returns = []
        episode_components = []

        for ep in range(self.cfg["episodes"]):
            state, _ = self.env.reset()
            done = False

            total_r = 0.0
            comp_sum = {"speed":0, "lane":0, "collision":0, "headway":0}

            while not done:
                action = self.agent.act(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                total_r += reward

                comps = info.get("multi_reward_components", None)
                if comps:
                    for k in comp_sum:
                        comp_sum[k] += comps[k]

            episode_returns.append(total_r)
            episode_components.append(comp_sum)

        return episode_returns, episode_components
