# from typing import Dict, List

# from tqdm import trange


# class Trainer:
#     def __init__(self, env, agent, config: Dict):
#         self.env = env
#         self.agent = agent
#         self.cfg = config["train"]

#     def train(self) -> List[float]:
#         returns = []
#         for ep in trange(self.cfg["episodes"], desc="Training"):
#             state, _ = self.env.reset()
#             done = False
#             total_r = 0.0
#             while not done:
#                 action = self.agent.act(state)
#                 next_state, reward, terminated, truncated, _ = self.env.step(action)
#                 done = terminated or truncated
#                 self.agent.step(state, action, reward, next_state, done)
#                 state = next_state
#                 total_r += reward
#             returns.append(total_r)
#         return returns


# from typing import Dict, List
# from tqdm import trange

# class Trainer:
#     def __init__(self, env, agent, config: Dict):
#         self.env = env
#         self.agent = agent
#         self.cfg = config["train"]

#     def train(self):
#         returns = []
#         components = []   # NEW

#         for ep in trange(self.cfg["episodes"], desc="Training"):
#             state, _ = self.env.reset()
#             done = False
#             total_r = 0.0
#             episode_components = {"speed": 0, "collision": 0, "lane": 0, "headway": 0}

#             while not done:
#                 action = self.agent.act(state)
#                 next_state, reward, terminated, truncated, info = self.env.step(action)
#                 done = terminated or truncated

#                 # accumulate component terms
#                 if "multi_components" in info:
#                     for k, v in info["multi_components"].items():
#                         episode_components[k] += v

#                 self.agent.step(state, action, reward, next_state, done)
#                 state = next_state
#                 total_r += reward

#             returns.append(total_r)
#             components.append(episode_components)

#         return returns, components

# from tqdm import trange

# class Trainer:
#     def __init__(self, env, agent, config):
#         self.env = env
#         self.agent = agent
#         self.cfg = config["train"]

#     def train(self):
#         returns = []

#         for ep in trange(self.cfg["episodes"], desc="Training"):
#             state, _ = self.env.reset()
#             done = False
#             total_r = 0.0

#             while not done:
#                 # Choose action (with epsilon-greedy)
#                 action = self.agent.act(state)

#                 # Step environment
#                 next_state, reward, terminated, truncated, info = self.env.step(action)
#                 done = terminated or truncated

#                 # Update agent
#                 self.agent.step(state, action, reward, next_state, done)

#                 # Move ahead
#                 state = next_state
#                 total_r += reward

#             # Log episode return
#             returns.append(total_r)

#         return returns


# from tqdm import trange

# class Trainer:
#     def __init__(self, env, agent, config):
#         self.env = env
#         self.agent = agent
#         self.cfg = config["train"]

#     def train(self):
#         returns = []
#         components = []   # store multi-objective reward breakdown

#         for ep in trange(self.cfg["episodes"], desc="Training"):
#             state, _ = self.env.reset()
#             done = False

#             total_r = 0.0

#             # Component accumulators for this episode
#             episode_components = {
#                 "speed": 0.0,
#                 "collision": 0.0,
#                 "lane": 0.0,
#                 "headway": 0.0
#             }

#             while not done:
#                 # Epsilon-greedy action
#                 action = self.agent.act(state)

#                 # Step the environment
#                 next_state, reward, terminated, truncated, info = self.env.step(action)
#                 done = terminated or truncated

#                 # Accumulate multi-objective components
#                 if "multi_components" in info:
#                     for key, value in info["multi_components"].items():
#                         episode_components[key] += value

#                 # Update agent (DQN, DoubleDQN, PER, etc.)
#                 self.agent.step(state, action, reward, next_state, done)

#                 # Move to next state
#                 state = next_state
#                 total_r += reward

#             # Store per-episode totals
#             returns.append(total_r)
#             components.append(episode_components)

#         return returns, components

import torch
from tqdm import trange

class TrainerComponents:
    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.cfg = config["train"]

    def train(self):
        episode_returns = []
        episode_components = []

        for ep in trange(self.cfg["episodes"], desc="Training"):
            state, _ = self.env.reset()
            done = False

            total_r = 0.0
            comp_sum = {"speed": 0.0, "lane": 0.0, "collision": 0.0, "headway": 0.0}

            while not done:
                action = self.agent.act(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                total_r += reward

                # MORL component accumulation
                comps = info.get("multi_reward_components", None)
                if comps is not None:
                    for k in comp_sum:
                        comp_sum[k] += comps.get(k, 0.0)

            episode_returns.append(total_r)
            episode_components.append(comp_sum)

            # DEBUG: Save model every 10 episodes
            if ep % 10 == 0:
                path = f"results/models/debug_ep_{ep}.pth"
                torch.save(self.agent.online.state_dict(), path)
                print("Saved intermediate model:", path)

        # FINAL SAVE
        final_path = f"results/models/{self.agent.__class__.__name__}.pth"
        torch.save(self.agent.online.state_dict(), final_path)
        print("FINAL model saved to:", final_path)

        return episode_returns, episode_components
