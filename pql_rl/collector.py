class Collector(object):
    def __init__(self, envs, policy=None, replay_buffer=None):
        """
        Args:
            envs:
        """
        self.envs = envs
        self.policy = policy
        self.replay_buffer = replay_buffer

    def collect(self, epsode_ls):
        for i in range(len(self.envs)):
            for j in range(epsode_ls[i]):
                trajectory = []
                self.envs[i].reset()
                while True:
                    action = self.envs[i].action_space.sample()
                    obs, reward, done, info = self.envs[i].step(action)
                    trajectory.append((obs, reward, done, info))
                    if done:
                        break
        pass


if __name__ == "__main__":
    import gym

    env_num = 2
    envs = [gym.make("CartPole-v1") for i in range(env_num)]
    collector = Collector(envs)
    collector.collect([1 for i in range(env_num)])
    pass
