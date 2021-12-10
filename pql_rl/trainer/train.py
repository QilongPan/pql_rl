class Trainer(object):
    def __init__(self) -> None:
        super().__init__()

    def train(self):
        """
        1.收集数据
        2.训练
        3.测试
        """


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
