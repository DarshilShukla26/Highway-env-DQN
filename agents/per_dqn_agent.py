import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.dqn_agent import DQNAgent
from memory.per_buffer import PERBuffer


class PERDQNAgent(DQNAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        rcfg = config["replay"]
        self.alpha = rcfg["alpha"]
        self.beta_start = rcfg["beta_start"]
        self.beta_frames = rcfg["beta_frames"]
        self.per_eps = rcfg["eps"]
        self.buffer = PERBuffer(self.cfg["buffer_size"], alpha=self.alpha, eps=self.per_eps)
        self.frame_idx = 0

    def _beta(self) -> float:
        return min(1.0, self.beta_start + self.frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def step(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)
        self.frame_idx += 1
        if len(self.buffer) >= self.batch_size:
            self.learn()
        if self.step_count % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

    def learn(self):
        beta = self._beta()
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights,
        ) = self.buffer.sample(self.batch_size, beta)

        states = torch.from_numpy(states.reshape(self.batch_size, -1)).float().to(self.device)
        next_states = torch.from_numpy(next_states.reshape(self.batch_size, -1)).float().to(self.device)
        actions = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        dones = torch.from_numpy(dones.astype(np.float32)).unsqueeze(1).to(self.device)
        weights_t = torch.from_numpy(weights).float().unsqueeze(1).to(self.device)

        # Q-values
        q_values = self.online(states).gather(1, actions)

        # DoubleDQN style target
        with torch.no_grad():
            next_online = self.online(next_states)
            next_actions = next_online.argmax(dim=1, keepdim=True)
            next_target = self.target(next_states)
            next_q = next_target.gather(1, next_actions)
            target = rewards + self.gamma * (1 - dones) * next_q

        # Ensure TD errors are proper floats
        td_errors = (q_values - target).detach().cpu().numpy().astype(np.float32).squeeze()

        # Update PER priorities
        self.buffer.update_priorities(indices, td_errors)

        # Weighted MSE loss
        loss = (weights_t * (q_values - target) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
