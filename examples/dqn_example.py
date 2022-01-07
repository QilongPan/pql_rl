import gym
import torch
from pql_rl.actor import Actor
from pql_rl.buffer.base import ReplayBuffer
from pql_rl.infer.base import InferServer
from pql_rl.policy import DQN, DiscreteRandomPolicy
from pql_rl.policy.dqn import Net

epoch = 100000
max_step = 10000
e_greedy = 0.1
batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"


class Config(object):

    pass


def train():
    buffer_size = 100
    env = gym.make("CartPole-v1")
    replay_buffer = ReplayBuffer(buffer_size)
    # discrete_random_policy = DiscreteRandomPolicy(action_num=2)
    input_dim = env.observation_space.shape or env.observation_space.n
    if not isinstance(input_dim, int):
        input_dim = input_dim[0]
    output_dim = env.action_space.shape or env.action_space.n
    net = Net(input_dim, output_dim)
    config = Config()
    policy = DQN(net, config)
    infer_server = InferServer(policy)
    actor = Actor(env, infer_server, replay_buffer)
    actor.run(episode_num=10)
    env2 = gym.make("CartPole-v1")
    actor2 = Actor(env2, infer_server, replay_buffer)
    actor2.run(episode_num=10)
    print(len(replay_buffer.trajectories))
    batch = replay_buffer.sample(300)
    # env = gym.make("CartPole-v1")
    # input_dim = env.observation_space.shape or env.observation_space.n
    # if not isinstance(input_dim, int):
    #     input_dim = input_dim[0]
    # output_dim = env.action_space.shape or env.action_space.n
    # dqn = DQN(batch_size, device, input_dim, output_dim, e_greedy)
    # for episode in range(epoch):
    #     obs = env.reset()
    #     step_num = 0
    #     done = False
    #     reward_sum = 0
    #     while step_num < max_step and not done:
    #         env.render()
    #         # action = env.action_space.sample()
    #         action = dqn.select_action(obs)
    #         next_obs, reward, done, info = env.step(action)
    #         reward_sum += reward
    #         dqn.store_transition(obs, action, reward, next_obs, done)
    #         dqn.learn()
    #         obs = next_obs
    #     print(reward_sum)
    # env.close()
    # return dqn


if __name__ == "__main__":
    pass
