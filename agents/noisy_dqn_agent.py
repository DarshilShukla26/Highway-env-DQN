import numpy as np
import torch
import torch.optim as optim

from agents.dqn_agent import DQNAgent
from agents.networks.noisy_net import NoisyDQN


class NoisyDQNAgent(DQNAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.online = NoisyDQN(self.state_dim, self.action_dim).to(self.device)
        self.target = NoisyDQN(self.state_dim, self.action_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=self.cfg["lr"])

    def _epsilon(self) -> float:
        # Not used for Noisy Nets; exploration is handled by noisy layers
        return 0.0

    def act(self, state) -> int:
        self.step_count += 1
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        state_t = torch.from_numpy(state).to(self.device)
        with torch.no_grad():
            q = self.online(state_t)
        return int(q.argmax(dim=1).item())

    def act_greedy(self, state) -> int:
        return self.act(state)

    def learn(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.from_numpy(states.reshape(self.batch_size, -1)).float().to(self.device)
        next_states = torch.from_numpy(next_states.reshape(self.batch_size, -1)).float().to(self.device)
        actions = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        dones = torch.from_numpy(dones.astype(np.float32)).unsqueeze(1).to(self.device)

        q_values = self.online(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + self.gamma * (1 - dones) * next_q

        loss = (q_values - target).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # reset noise after each update
        self.online.reset_noise()
        self.target.reset_noise()
