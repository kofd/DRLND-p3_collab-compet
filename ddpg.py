from collections import namedtuple, deque
import random
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Normal(dist.Normal):
    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return torch.normal(self.loc.expand(shape), self.scale.expand(shape))


class Actor(nn.Module):
    def __init__(self, feature_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(feature_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_mean = nn.Linear(128, action_size)
        self.fc_std = nn.Linear(128, action_size)

    def forward(self, state):
        X = F.relu(self.fc1(state))
        #X = F.relu(self.fc2(X))
        A = Normal(0.001 * self.fc_mean(X), torch.exp(self.fc_std(X)))
        return A


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128+action_size, 128)
        self.fc_mean = nn.Linear(128, 1)
        self.fc_std = nn.Linear(128, 1)

    def forward(self, state, action):
        X = F.relu(self.fc1(state))
        X = F.relu(self.fc2(torch.cat([X, action], -1)))
        V = Normal(self.fc_mean(X), torch.exp(self.fc_std(X)))
        return V


class Distill(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)

    def forward(self, state):
        X = F.relu(self.fc1(state))
        #X = F.relu(self.fc2(X))
        return self.fc3(X)


class Agent:
    def __init__(self, state_size, action_size, learning_rate=0.0001, tau=0.001,
                 buffer_size=10000, batch_size=64):
        self.tau = tau
        self.local_distill = Distill(state_size)
        self.target_distill = Distill(state_size)
        self.local_actor = Actor(state_size, action_size)
        self.target_actor = Actor(state_size, action_size)
        self.local_critic = Critic(state_size, action_size)
        self.target_critic = Critic(state_size, action_size)
        self.distill_optimizer = optim.Adam(self.local_distill.parameters(), lr=learning_rate)
        self.actor_optimizer = optim.Adam(self.local_actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.local_critic.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.batch_size = batch_size

        for target_param, local_param in zip(self.target_actor.parameters(), self.local_actor.parameters()):
            target_param.data.copy_(local_param.data)
        for target_param, local_param in zip(self.target_critic.parameters(), self.local_critic.parameters()):
            target_param.data.copy_(local_param.data)

    def act(self, state, det=True):
        state = torch.from_numpy(state).float().to(device)

        self.local_actor.eval()
        i = len(state) // 2
        with torch.no_grad():
            #action_values = []
            #action_dist = self.local_actor.forward(state[:i])
            #action_values.append(action_dist.mean if det else action_dist.sample())
            #action_dist = self.target_actor.forward(state[i:])
            #action_values.append(action_dist.mean if det else action_dist.sample())
            #action_values = torch.tanh(torch.cat(action_values, 0)).cpu().data.numpy()
            action_dist = self.local_actor.forward(state)
            action_values = action_dist.mean if det else action_dist.sample()
            action_values = torch.tanh(action_values).cpu().data.numpy()
        self.local_actor.train()

        return action_values

    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def distill_loss(self, state):
        state = torch.from_numpy(state).float().to(device)

        self.local_actor.eval()
        with torch.no_grad():
            target_mapping = self.target_distill.forward(state).detach()
            mapping = self.local_distill.forward(state)
        self.local_actor.train()
        return (target_mapping - mapping).pow(2).mean(1).cpu().data.numpy()

    def learn(self, gamma, det=True):
        self.update_local(gamma, det)
        self.update_target()

    def update_local(self, gamma, det=True):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        target_mapping = self.target_distill.forward(next_states).detach()
        mapping = self.local_distill.forward(next_states)
        distill_loss = (target_mapping - mapping).pow(2).mean(1, keepdim=True)

        self.local_distill.zero_grad()
        distill_loss.mean().backward()
        self.distill_optimizer.step()

        next_actions = torch.tanh(self.target_actor.forward(states).mean).detach()
        next_values = self.target_critic.forward(states, next_actions).mean.detach()
        q_target = 10 * rewards + ((1. - dones) * gamma * next_values)

        q_local = self.local_critic.forward(states, actions)
        q_local = q_local.mean# if det else q_local.sample()
        critic_loss = F.mse_loss(q_local, q_target)#-q_local.log_prob(q_target).mean()

        self.local_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_critic.parameters(), 1)
        self.critic_optimizer.step()

        local_actions = self.local_actor.forward(states)
        policy_loss = -self.local_critic.forward(states, torch.tanh(local_actions.mean)).mean.mean()
        if not det:
            policy_loss -= self.local_critic.forward(states, torch.tanh(local_actions.sample())).mean.mean()
            policy_loss /= 2

        self.local_actor.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_actor.parameters(), 1)
        self.actor_optimizer.step()

    def update_target(self):
        for target_param, local_param in zip(self.target_actor.parameters(), self.local_actor.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
        for target_param, local_param in zip(self.target_critic.parameters(), self.local_critic.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.batch_size = batch_size
        self.memory = deque(maxlen=int(buffer_size))
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        for state, action, reward, next_state, done in \
                zip(state, action, reward, next_state, done):
            self.memory.append(self.experience(state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
