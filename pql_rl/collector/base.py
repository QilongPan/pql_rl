from pql_rl.env.util import create_env


class Collector(object):
    """
    actor container be used to collect data.
    收集数据需要：
    1.环境
    2.推理服
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

    def run(self):
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False
    )

    # common args
    parser.add_argument(
        "--env_name", type=str, default=None, required=True, help="env name",
    )
    parser.add_argument(
        "--env_num", type=int, default=None, required=True, help="env num",
    )
    cfg = parser.parse_known_args()[0]
    collector = Collector(cfg)
