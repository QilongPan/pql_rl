from pql_rl.collector import Collector

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False
    )

    # common args
    parser.add_argument(
        "--env_name",
        type=str,
        default=None,
        required=True,
        help="env name",
    )
    parser.add_argument(
        "--env_num",
        type=int,
        default=None,
        required=True,
        help="env num",
    )
    cfg = parser.parse_known_args()[0]
    collector = Collector(cfg)
    collector.run()
