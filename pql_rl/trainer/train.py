import torch
from pql_rl.env.util import create_env
from pql_rl.policy.dqn import DQN


class Trainer(object):
    def __init__(self) -> None:
        super().__init__()

    def train(self):
        """
        1.收集数据
        2.训练
        3.测试
        """
        env = create_env("CartPole-v1")
        input_dim = env.observation_space.shape or env.observation_space.n
        if not isinstance(input_dim, int):
            input_dim = input_dim[0]
        output_dim = env.action_space.shape or env.action_space.n
        dqn = DQN(batch_size, device, input_dim, output_dim, e_greedy)
        for episode in range(epoch):
            obs = env.reset()
            step_num = 0
            done = False
            reward_sum = 0
            while step_num < max_step and not done:
                env.render()
                # action = env.action_space.sample()
                action = dqn.select_action(obs)
                next_obs, reward, done, info = env.step(action)
                reward_sum += reward
                dqn.store_transition(obs, action, reward, next_obs, done)
                dqn.learn()
                obs = next_obs
            print(reward_sum)
        env.close()

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


if __name__ == "__main__":

    import gym
    from pql_rl.actor.base import Actor
    from pql_rl.buffer.base import ReplayBuffer
    from pql_rl.infer.base import InferServer
    from pql_rl.policy.random import DiscreteRandomPolicy

    env_num = 2
    buffer_size = 100
    env = gym.make("CartPole-v1")
    replay_buffer = ReplayBuffer(buffer_size)
    discrete_random_policy = DiscreteRandomPolicy(action_num=2)
    infer_server = InferServer(discrete_random_policy)
    actor = Actor(env, infer_server, replay_buffer)
    actor.run(episode_len=10)
    actor2 = Actor(env, infer_server, replay_buffer)
    actor2.run(episode_len=10)
    print(len(replay_buffer.trajectories))
    batch = replay_buffer.sample(300)
    print(batch["done"])
