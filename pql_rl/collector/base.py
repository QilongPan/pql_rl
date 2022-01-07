from pql_rl.env.util import create_env


class Collector(object):
    """
    actor container be used to collect data.
    收集数据需要：
    1.env container
    2.推理服接口
    3.actor
    """

    def __init__(self, cfg) -> None:
        """
        cfg参数中传入env name、policy name、env num
        """
        super().__init__()
        self.cfg = cfg
        self.create_env()
        # infer_server = InferServer(discrete_random_policy)

    def create_env(self):
        self.envs = [
            create_env(self.cfg.env_name) for i in range(self.cfg.env_num)
        ]

    def run(self, step_num=0, episode_num=0):
        """
        Params:
            step_num:collect step number.
            episode_num:collect episode number.
        """

        pass
