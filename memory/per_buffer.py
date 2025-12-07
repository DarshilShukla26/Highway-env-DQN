import random
from typing import Tuple
import numpy as np


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def get(self, s: float):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self) -> float:
        return self.tree[0]


class PERBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.eps = eps
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)

        # Prevent corrupt entries
        for item in data:
            if isinstance(item, (int, float, np.ndarray, list, tuple)) is False:
                return

        priority = self.max_priority
        self.tree.add(priority, data)

    def sample(self, batch_size: int, beta: float):
        segment = self.tree.total / batch_size

        states, actions, rewards, next_states, dones = [], [], [], [], []
        indices, priorities = [], []

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)

            # Safety: skip invalid entries
            if not isinstance(data, tuple) or len(data) != 5:
                s = random.uniform(segment * i, segment * (i + 1))
                idx, p, data = self.tree.get(s)

            priorities.append(p)
            indices.append(idx)

            s0, a0, r0, s1, d0 = data
            states.append(s0)
            actions.append(a0)
            rewards.append(r0)
            next_states.append(s1)
            dones.append(d0)

        probs = np.array(priorities, dtype=np.float32) / self.tree.total
        weights = (probs * self.tree.n_entries) ** (-beta)
        weights /= (weights.max() + 1e-8)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights,
        )

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):

            # Ensure proper float values
            err = float(err)

            priority = (abs(err) + float(self.eps)) ** float(self.alpha)

            # Track highest seen priority
            self.max_priority = max(self.max_priority, priority)

            # Update tree
            self.tree.update(idx, priority)

    def __len__(self) -> int:
        return self.tree.n_entries
