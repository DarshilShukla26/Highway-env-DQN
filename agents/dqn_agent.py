import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.networks.base_net import BaseDQN
from memory.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, env, config: Dict):
        self.env = env
        self.cfg = config["train"]

        obs_shape = env.observation_space.shape
        self.state_dim = int(np.prod(obs_shape))
        self.action_dim = env.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online + Target Networks
        self.online = BaseDQN(self.state_dim, self.action_dim).to(self.device)
        self.target = BaseDQN(self.state_dim, self.action_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())

        self.optimizer = optim.Adam(self.online.parameters(), lr=self.cfg["lr"])
        self.gamma = self.cfg["gamma"]

        # ===============================
        #    Exploration parameters
        # ===============================
        self.eps_start = self.cfg["eps_start"]
        self.eps_end = self.cfg["eps_end"]
        self.decay_type = self.cfg.get("eps_decay_type", "linear")

        # Linear decay
        if self.decay_type == "linear":
            self.eps_decay = self.cfg.get("eps_decay", 0.995)

        # Exponential decay
        elif self.decay_type == "exp":
            self.eps_decay_rate = self.cfg.get("eps_decay_rate", 0.995)

        else:
            raise ValueError(f"Unknown eps_decay_type: {self.decay_type}")

        self.eps = self.eps_start
        self.step_count = 0

        # Replay buffer
        self.buffer = ReplayBuffer(self.cfg["buffer_size"])
        self.batch_size = self.cfg["batch_size"]
        self.target_update = self.cfg["target_update"]

    # ===============================
    #     Epsilon update rule
    # ===============================
    def update_epsilon(self):
        if self.decay_type == "linear":
            self.eps = max(self.eps_end, self.eps * self.eps_decay)

        elif self.decay_type == "exp":
            self.eps = self.eps_end + (self.eps_start - self.eps_end) * (
                self.eps_decay_rate ** self.step_count
            )

        return self.eps

    # ===============================
    #     Choose action
    # ===============================
    def act(self, state) -> int:
        self.step_count += 1
        eps = self.update_epsilon()  # update epsilon every step

        # Exploration vs exploitation
        if random.random() < eps:
            return self.env.action_space.sample()

        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        state_t = torch.from_numpy(state).to(self.device)

        with torch.no_grad():
            q = self.online(state_t)

        return int(q.argmax(dim=1).item())

    def act_greedy(self, state) -> int:
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        state_t = torch.from_numpy(state).to(self.device)
        with torch.no_grad():
            q = self.online(state_t)
        return int(q.argmax(dim=1).item())

    # ===============================
    #        Add transition
    # ===============================
    def step(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

        # Learn if enough samples
        if len(self.buffer) >= self.batch_size:
            self.learn()

        # Update target network
        if self.step_count % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

    # ===============================
    #          Learning step
    # ===============================
    def learn(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.from_numpy(states.reshape(self.batch_size, -1)).float().to(self.device)
        next_states = torch.from_numpy(next_states.reshape(self.batch_size, -1)).float().to(self.device)
        actions = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        dones = torch.from_numpy(dones.astype(np.float32)).unsqueeze(1).to(self.device)

        # Q(s,a)
        q_values = self.online(states).gather(1, actions)

        # target = r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + self.gamma * (1 - dones) * next_q

        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
