import torch
from pql_rl.collector import Collector
from pql_rl.env.util import create_env
from pql_rl.policy.dqn import DQN


class Trainer(object):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

    def train(self):
        """
        1.收集数据
        2.训练
        3.测试
        """
        collector = Collector(self.cfg)

        # env = create_env("CartPole-v1")
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

    @classmethod
    def add_args(cls, parser):
        p = parser
        p.add_argument("--epoch", default=100, type=int, help="train epoch")
        p.add_argument("--env_name", default=32, type=str, help="env name")
        p.add_argument(
            "--policy",
            type=str,
            default=None,
            required=True,
            help="reinforcement algorithm, e.g. random,dqn,ppo",
        )
        p.add_argument(
            "--batch_size", default=32, type=int, help="train batch size"
        )
        p.add_argument(
            "--device",
            default="cuda" if torch.cuda.is_available() else "cpu",
            type=str,
            help="train device",
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False
    )
    Trainer.add_args(parser)
    cfg = parser.parse_known_args()[0]
    trainer = Trainer(cfg)
    trainer.train()
