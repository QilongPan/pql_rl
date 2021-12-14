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

    @classmethod
    def add_args(cls, parser):
        p = parser
        # super().add_args(p)
        # p.add_argument('--experiment_summaries_interval', default=20, type=int, help='How often in seconds we write avg. statistics about the experiment (reward, episode length, extra stats...)')

        # p.add_argument('--adam_eps', default=1e-6, type=float, help='Adam epsilon parameter (1e-8 to 1e-5 seem to reliably work okay, 1e-3 and up does not work)')
        # p.add_argument('--adam_beta1', default=0.9, type=float, help='Adam momentum decay coefficient')
        # p.add_argument('--adam_beta2', default=0.999, type=float, help='Adam second momentum decay coefficient')

        # p.add_argument('--gae_lambda', default=0.95, type=float, help='Generalized Advantage Estimation discounting (only used when V-trace is False')        

if __name__ == "__main__":
    discrete_random_policy = DiscreteRandomPolicy(10)
    print(discrete_random_policy([12, 1, 1, 1, 1，100]))
