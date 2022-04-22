import random
import numpy as np
import torch

from netease_rank.data import DataSource
from netease_rank.model import MODELS
from .train import Trainer


def main_process(cfg):
    seed = cfg.GLOBAL.RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    data_source = DataSource(cfg)
    model = MODELS.get(cfg.MODEL.NAME)(cfg, data_source.cardinality)
    trainer = Trainer(cfg, data_source, model)
    if cfg.GLOBAL.TRAIN:
        trainer.train()
    else:
        if cfg.GLOBAL.TEST_EPOCH > 0:
            trainer.test(cfg.GLOBAL.TEST_EPOCH)
        else:
            trainer.test()
