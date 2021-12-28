"""
主要负责调度policy进行预测
可以收集一个时间区间的observation，然后policy一批同时预测
"""


class InferServer(object):
    """
    Get obs,return action
    Params:
        policy: infer policy,eg:random policy
    """

    def __init__(self, policy) -> None:
        super().__init__()
        self.policy = policy

    def get_actions(self, obs_ls):
        """
        Get actions use policy.
        Args:
            obs_ls: observation
        """
        return self.policy(obs_ls)


if __name__ == "__main__":

    from pql_rl.policy.random import DiscreteRandomPolicy

    infer_server = InferServer(DiscreteRandomPolicy(action_num=6))
    print(infer_server.get_actions([0, 1, 1, 1, 1, 1, 1]))
