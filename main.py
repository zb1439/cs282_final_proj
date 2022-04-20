import argparse
import os
import sys
from netease_rank.config import BaseConfig
from netease_rank.pipeline import main_process

sys.path.append(os.getcwd())

try:
    from config import config  # noqa
except ModuleNotFoundError:
    print("Warning: local config not found, using the default BaseConfig")
    config = BaseConfig()


def default_parser():
    parser = argparse.ArgumentParser("Netease Training Framework")
    parser.add_argument("--resume", action="store_true", help="whether to resume training from the latest epoch")
    parser.add_argument("--test-all", action="store_true", help="whether to test all stored checkpoints")
    parser.add_argument("--test-epoch", default=-1, type=int, help="which epoch to test")
    parser.add_argument("--num-workers", default=1, type=int, help="number of threads for data loading")
    return parser


if __name__ == '__main__':
    args = default_parser().parse_args()
    if args.resume:
        config.GLOBAL.RESUME = True
    if args.test_all:
        config.GLOBAL.TRAIN = False
    if args.test_epoch > 0:
        config.GLOBAL.TRAIN = False
        config.GLOBAL.TEST_EPOCH = args.test_epoch
    config.GLOBAL.NUM_WORKERS = args.num_workers
    print(config)
    main_process(config)
