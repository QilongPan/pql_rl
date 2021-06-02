class Collector(object):
    def __init__(self, env_container, policy=None, replay_buffer=None):
        """
        Args:
            env_container:
        """
        self.env_container = env_container
        self.policy = policy
        self.replay_buffer = replay_buffer

    def collect(self, epsode_ls):
        """
        Args:
            epsode_ls:every env collect epsode
        """
        obs = self.env_container.reset()
        while True:
            if self.policy is None:
                action = self.env_container.random_actions()
            else:
                action = self.policy(obs)
            next_obs, reward, done, info = self.env_container.step(action)

            break
        for i in range(len(self.envs)):
            for j in range(epsode_ls[i]):
                trajectory = []
                obs = self.envs[i].reset()
                while True:
                    action = self.envs[i].action_space.sample()
                    next_obs, reward, done, info = self.envs[i].step(action)
                    trajectory.append((obs, reward, done, next_obs, info))
                    obs = next_obs
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
