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
    def __init__(self, batch_size, device, input_dim, output_dim, e_greedy):
        self.replay_buffer = deque(maxlen=1000)
        self.batch_size = batch_size
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.e_greedy = e_greedy
        self.gamma = 0.9
        self.q_net = Net(input_dim, output_dim)
        self.loss_fun = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters())
        pass

    def store_transition(self, obs, actor, reward, next_obs, done):
        self.replay_buffer.append((obs, actor, reward, next_obs, done))

    def learn(self):
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
        p.add_argument("--epoch", default=100, type=int, help="train epoch")
        p.add_argument(
            "--batch_size", default=32, type=int, help="train batch size"
        )
        p.add_argument(
            "--device",
            default="cuda" if torch.cuda.is_available() else "cpu",
            type=str,
            help="train device",
        )
        p.add_argument(
            "--eps", default=0.3, type=float, help="train explore probability"
        )


# epoch = 100000
# max_step = 10000
# e_greedy = 0.1
# batch_size = 32
# device = "cuda" if torch.cuda.is_available() else "cpu"


# def train():
#     env = gym.make("CartPole-v1")
#     input_dim = env.observation_space.shape or env.observation_space.n
#     if not isinstance(input_dim, int):
#         input_dim = input_dim[0]
#     output_dim = env.action_space.shape or env.action_space.n
#     dqn = DQN(batch_size, device, input_dim, output_dim, e_greedy)
#     for episode in range(epoch):
#         obs = env.reset()
#         step_num = 0
#         done = False
#         reward_sum = 0
#         while step_num < max_step and not done:
#             env.render()
#             # action = env.action_space.sample()
#             action = dqn.select_action(obs)
#             next_obs, reward, done, info = env.step(action)
#             reward_sum += reward
#             dqn.store_transition(obs, action, reward, next_obs, done)
#             dqn.learn()
#             obs = next_obs
#         print(reward_sum)
#     env.close()
#     return dqn


if __name__ == "__main__":
    pass
