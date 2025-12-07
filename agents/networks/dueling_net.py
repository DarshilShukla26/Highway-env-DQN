import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + a - a.mean(dim=1, keepdim=True)
