import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.000   # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, state):
        X = self.fc1(state)
        X = self.fc2(F.relu(X))
        return torch.tanh(self.fc3(F.relu(X)))


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256+action_size, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-1e-20, 1e-20)

    def forward(self, state, action):
        X = self.fc1(state)
        X = self.fc2(torch.cat([F.relu(X), action], -1))
        X = self.fc3(F.relu(X))
        return self.fc4(F.relu(X))


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, buffer_size=BUFFER_SIZE,
                 batch_size=BATCH_SIZE, learning_rate=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate or LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=learning_rate or LR_CRITIC, weight_decay=WEIGHT_DECAY)

        for target_param, local_param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target_param.data.copy_(local_param.data)
        for target_param, local_param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target_param.data.copy_(local_param.data)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state, noise=None):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if noise is not None:
            action = action * (1-noise) + noise * np.random.uniform(-1, 1, action.shape)
        return action

    def reset(self):
        pass

    def learn(self, gamma, det=None, a=0.5, b=0.):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        if len(self.memory) < self.batch_size:
            return
        ids, weights, states, actions, rewards, next_states, dones = self.memory.sample(a, b)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next).detach()
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = (Q_expected - Q_targets).pow(2).mean(1)
        errors = np.abs((Q_expected - Q_targets).cpu().data.numpy())
        # Minimize the loss
        self.critic_local.zero_grad()
        (critic_loss * weights).mean().backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_local.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
        self.memory.update(ids, errors)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.errors = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        max_error = max(self.errors) if len(self.errors) > 0 else 1e-5
        for state, action, reward, next_state, done in \
                zip(state, action, reward, next_state, done):
            self.memory.append(self.experience(state, action, reward, next_state, done))
            self.errors.append(max_error)

    def sample(self, a, b):
        """Randomly sample a batch of experiences from memory."""
        p = np.array(self.errors, dtype=float) ** a
        p /= np.sum(p)

        try:
            ids = np.random.choice(np.arange(len(p)), size=self.batch_size, p=p, replace=True)
        except:
            for k, v in locals().items():
                if 'shape' in dir(v):
                    print(k, v.shape)
                else:
                    print(k, v)
                raise
        experiences = [self.memory[i] for i in ids]
        p = np.array([p[i] for i in ids])

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        weights = torch.from_numpy(np.power(1 / (len(self.errors) * p), b)).float().to(device)

        return (ids, weights, states, actions, rewards, next_states, dones)

    def update(self, ids, errors):
        for i, error in zip(ids, errors):
            self.errors[i] = error[0]

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
