import torch
import torch.nn as nn
from .noisy_linear import NoisyLinear


class NoisyDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = NoisyLinear(state_dim, 128)
        self.fc2 = NoisyLinear(128, 128)
        self.fc3 = NoisyLinear(128, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()
