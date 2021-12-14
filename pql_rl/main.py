import sys

from pql_rl.arguments import parse_args

# from algorithms.utils.arguments import (
#     get_algo_class,
#     maybe_load_from_checkpoint,
#     parse_args,
# )


def run_algorithm(cfg):
    # cfg = maybe_load_from_checkpoint(cfg)
    # algo = get_algo_class(cfg.algo)(cfg)
    # algo.initialize()
    # status = algo.run()
    # algo.finalize()
    # return status
    pass


def main():
    """Script entry point."""
    cfg = parse_args()
    status = run_algorithm(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
