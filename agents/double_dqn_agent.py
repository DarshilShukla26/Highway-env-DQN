import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.dqn_agent import DQNAgent


class DoubleDQNAgent(DQNAgent):
    def learn(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.from_numpy(states.reshape(self.batch_size, -1)).float().to(self.device)
        next_states = torch.from_numpy(next_states.reshape(self.batch_size, -1)).float().to(self.device)
        actions = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        dones = torch.from_numpy(dones.astype(np.float32)).unsqueeze(1).to(self.device)

        q_values = self.online(states).gather(1, actions)
        with torch.no_grad():
            next_online = self.online(next_states)
            next_actions = next_online.argmax(dim=1, keepdim=True)
            next_target = self.target(next_states)
            next_q = next_target.gather(1, next_actions)
            target = rewards + self.gamma * (1 - dones) * next_q

        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
