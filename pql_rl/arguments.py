"""
参数模块参照sample-factory
"""
import argparse
import sys

import torch


def get_policy_class(policy):
    policy_class = None

    if policy == "DiscreteRandom":
        from pql_rl.policy import DiscreteRandomPolicy

        policy_class = DiscreteRandomPolicy
    elif policy == "DQN":
        from pql_rl.policy import DQN

        policy_class = DQN
    else:
        print("Policy %s is not supported", policy)

    return policy_class


def arg_parser(argv=None, evaluation=False):
    if argv is None:
        argv = sys.argv[1:]

    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False
    )

    # common args
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        required=True,
        help="reinforcement algorithm, e.g. random,dqn,ppo",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        required=True,
        help="Fully-qualified environment name in the form envfamily_envname, e.g. atari_breakout or doom_battle",
    )
    parser.add_argument("--epoch", default=100, type=int, help="train epoch")
    parser.add_argument(
        "--batch_size", default=32, type=int, help="train batch size"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
        help="train device",
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Print the help message",
        required=False,
    )
    basic_args, _ = parser.parse_known_args(argv)
    policy = basic_args.policy
    # env = basic_args.env

    # # algorithm-specific parameters (e.g. for APPO)
    policy_class = get_policy_class(policy)
    policy_class.add_args(parser)

    # # env-specific parameters (e.g. for Doom env)
    # add_env_args(env, parser)

    # if evaluation:
    #     add_eval_args(parser)

    # # env-specific default values for algo parameters (e.g. model size and convolutional head configuration)
    # env_override_defaults(env, parser)
    return parser


def parse_args(argv=None, evaluation=False, parser=None):
    if argv is None:
        argv = sys.argv[1:]

    if parser is None:
        parser = arg_parser(argv, evaluation)

    # parse all the arguments (algo, env, and optionally evaluation)
    args = parser.parse_args(argv)

    if args.help:
        parser.print_help()
        sys.exit(0)

    # args.command_line = " ".join(argv)

    # # following is the trick to get only the args passed from the command line
    # # We copy the parser and set the default value of None for every argument. Since one cannot pass None
    # # from command line, we can differentiate between args passed from command line and args that got initialized
    # # from their default values. This will allow us later to load missing values from the config file without
    # # overriding anything passed from the command line
    # no_defaults_parser = copy.deepcopy(parser)
    # for arg_name in vars(args).keys():
    #     no_defaults_parser.set_defaults(**{arg_name: None})
    # cli_args = no_defaults_parser.parse_args(argv)

    # for arg_name in list(vars(cli_args).keys()):
    #     if cli_args.__dict__[arg_name] is None:
    #         del cli_args.__dict__[arg_name]

    # args.cli_args = vars(cli_args)
    # args.git_hash = get_git_commit_hash()
    return args
