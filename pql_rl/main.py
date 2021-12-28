import sys

from pql_rl.arguments import get_policy_class, parse_args

# from algorithms.utils.arguments import (
#     get_algo_class,
#     maybe_load_from_checkpoint,
#     parse_args,
# )


def run_algorithm(cfg):
    # cfg = maybe_load_from_checkpoint(cfg)
    policy = get_policy_class(cfg.policy)(cfg)
    # algo.initialize()
    status = policy.train()
    # algo.finalize()
    return status


def main():
    """Script entry point."""
    cfg = parse_args()
    status = run_algorithm(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
