import random


class DiscreteRandomPolicy(object):
    """
    Discrete env random policy
    Params:
        action_num:action number.
    """

    def __init__(self, action_num=10) -> None:
        super().__init__()
        self.action_num = action_num

    def __call__(self, obs_ls):
        """
        Args:
            obs_ls:observations
        """
        actions = [
            random.randint(0, self.action_num - 1) for i in range(len(obs_ls))
        ]
        return actions


if __name__ == "__main__":
    discrete_random_policy = DiscreteRandomPolicy(10)
    print(discrete_random_policy([12, 1, 1, 1, 1]))
