import numpy as np
import torch
import torch.optim as optim

from agents.dqn_agent import DQNAgent
from agents.networks.dueling_net import DuelingDQN


class DuelingDQNAgent(DQNAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        # Override networks with dueling architecture
        self.online = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=self.cfg["lr"])
