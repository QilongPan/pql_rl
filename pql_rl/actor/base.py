import gym
from pql_rl.buffer.base import ReplayBuffer
from pql_rl.infer.base import InferServer


class Actor(object):
    """
    Run the environment to generate episodes 
    Params:
        env:environment
        infer_api:policy infer
        replay_buffer_api: save episodes
        render:env render
        observation:current env observation
    """

    def __init__(
        self,
        env: gym.Env,
        infer_api: InferServer,
        replay_buffer_api: ReplayBuffer,
        render=False,
    ) -> None:
        super().__init__()
        self.env = env
        self.infer_api = infer_api
        self.replay_buffer_api = replay_buffer_api
        self.render = render
        self.observation = None
        self.reset()

    def reset(self):
        self.observation = self.env.reset()
        if self.render:
            self.env.render()

    def run(self, episode_len: int = None):
        current_episode_len = 0
        current_trajectory = []
        while True:
            actions = self.infer_api.get_actions([self.observation])
            next_observation, reward, done, info = self.env.step(actions[0])
            current_trajectory.append(
                (
                    self.observation,
                    actions[0],
                    reward,
                    done,
                    info,
                    next_observation,
                )
            )
            self.observation = next_observation
            if self.render:
                self.env.render()
            if done:
                self.replay_buffer_api.add(current_trajectory)
                self.observation = self.env.reset()
                current_episode_len += 1
                current_trajectory = []
            if current_episode_len >= episode_len:
                break


if __name__ == "__main__":
    import gym
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
    env2 = gym.make("CartPole-v1")
    actor2 = Actor(env2, infer_server, replay_buffer)
    actor2.run(episode_len=10)
    print(len(replay_buffer.trajectories))
    batch = replay_buffer.sample(300)
    print(batch["done"])
