import random
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer(object):
    def __init__(self):
        self.replay_buffer = deque(max_len=2)


class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = F.relu(self.fc3(s))
        return s


class DQN(object):
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.replay_buffer = deque(maxlen=1000)
        self.batch_size = self.cfg.batch_size
        self.device = self.cfg.device
        self.input_dim = self.cfg.observation_dim
        self.output_dim = self.cfg.action_num
        self.e_greedy = self.cfg.eps
        self.gamma = self.cfg.gamma
        self.q_net = Net(self.input_dim, self.output_dim)
        self.loss_fun = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters())

    def store_transition(self, obs, actor, reward, next_obs, done):
        self.replay_buffer.append((obs, actor, reward, next_obs, done))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        obs_batch = [transition[0] for transition in minibatch]
        actor_batch = [transition[1] for transition in minibatch]
        reward_batch = [transition[2] for transition in minibatch]
        next_obs_batch = [transition[3] for transition in minibatch]
        not_done_batch = [1 - transition[4] for transition in minibatch]
        obs_batch_tensor = torch.FloatTensor(obs_batch)
        reward_batch_tensor = torch.FloatTensor(reward_batch)
        next_obs_batch_tensor = torch.FloatTensor(next_obs_batch)
        actor_batch_tensor = torch.LongTensor(actor_batch)
        not_done_batch_tensor = torch.LongTensor(not_done_batch)
        q_obs_batch = self.q_net(obs_batch_tensor).gather(
            1, actor_batch_tensor.unsqueeze(1)
        )
        q_next = self.q_net(next_obs_batch_tensor).detach()
        q_next_max = q_next.max(1)[0]
        # 在done的情况下没有下一状态
        q_next_max = q_next_max * not_done_batch_tensor
        q_target = reward_batch_tensor + self.gamma * q_next_max
        q_target = q_target.view(len(obs_batch), 1)
        assert q_target.shape == q_obs_batch.shape

        loss = self.loss_fun(q_target, q_obs_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def select_action(self, obs):
        if np.random.random() < self.e_greedy:
            action = np.random.randint(0, self.output_dim)
            return action
        else:
            obs = obs[np.newaxis, :]
            obs_tensor = torch.FloatTensor(obs)
            if self.device == "cuda":
                obs_tensor = obs_tensor.cuda()
            action_value = self.q_net(obs_tensor)
            max_action = torch.argmax(action_value, 1).cpu().numpy()[0]
            return max_action

    @classmethod
    def add_args(cls, parser):
        p = parser
        # super().add_args(p)
        p.add_argument(
            "--eps", default=0.3, type=float, help="train explore probability"
        )
