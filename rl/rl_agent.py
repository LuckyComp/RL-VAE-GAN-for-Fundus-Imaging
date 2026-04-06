import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    def __init__(self, latent_dim, n_actions, hidden_dims=(256, 128)):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[1], n_actions),
        )

    def forward(self, x):
        return self.layers(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(torch.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        latent_dim,
        n_actions,
        device=None,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_final=0.01,
        epsilon_decay=10000,
        buffer_capacity=10000,
        target_update=200,
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQNNetwork(latent_dim, n_actions).to(self.device)
        self.target_net = DQNNetwork(latent_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=buffer_capacity)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.n_actions = n_actions
        self.steps_done = 0
        self.target_update = target_update

    def select_action(self, state):
        self.steps_done += 1

        eps_threshold = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
            torch.exp(-1.0 * self.steps_done / self.epsilon_decay)
        self.epsilon = eps_threshold.item() if isinstance(eps_threshold, torch.Tensor) else float(eps_threshold)

        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            tensor_state = state.unsqueeze(0).to(self.device)
            q_values = self.policy_net(tensor_state)
            action = q_values.argmax(dim=1).item()
            return action

    def store_transition(self, state, action, reward, next_state, done):
        state = state.detach().cpu()
        next_state = next_state.detach().cpu()
        self.memory.push(state, torch.tensor([action], dtype=torch.int64), torch.tensor([reward], dtype=torch.float32), next_state, torch.tensor([done], dtype=torch.float32))

    def optimize(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        state_action_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            expected_state_action_values = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * next_state_values

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy_state_dict'])
        self.target_net.load_state_dict(ckpt['target_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
