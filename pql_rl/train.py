import gym

from pql_rl.collector import Collector
from pql_rl.env_container import EnvContainer

if __name__ == "__main__":
    env_num = 2
    envs = [gym.make("CartPole-v1") for i in range(env_num)]
    env_container = EnvContainer(envs)
    collector = Collector()
    pass
