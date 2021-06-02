"""
env 的容器，可以放入多个env进行step
可以将多个环境的action同时传入policy进行推理
"""
import gym


class EnvContainer(gym.Env):
    """
    Attr
    """

    def __init__(self, envs):
        """
        Args:
            envs:
        """
        self.envs = envs
        pass

    def reset(self):
        return [env.reset() for env in self.envs]

    def step(self, actions):
        return [self.envs[i].step(actions[i]) for i in range(len(self.envs))]

    def seed(self, seeds):
        return [self.envs[i].seed(seeds[i]) for i in range(len(self.envs))]

    def render(self):
        pass

    def close(self):
        for i in range(len(self.envs)):
            self.envs[i].close()

    def random_actions(self):
        return [env.action_space.sample() for env in self.envs]
