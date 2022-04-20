import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from netease_rank.utils import logger, Registry


EVALUATOR = Registry("EVALUATOR")


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bs = cfg.TRAINING.BATCH_SIZE

    def eval(self, model, data_source):
        logger.info(f"Evaluating {self.__class__.__name__}")
        model.eval()
        data_source.test_mode = True
        loader = DataLoader(data_source, batch_size=self.bs, shuffle=False, num_workers=4)

        metrics = []
        for i, data in tqdm(enumerate(loader), len(loader)):
            user_feat, item_feat, scores = data
            pred_scores = model(user_feat, item_feat)
            metrics.append(self.calculate(pred_scores, scores))

        model.train()
        data_source.test_mode = False
        return np.mean(metrics)

    def calculate(self, preds, scores):
        raise NotImplementedError


@EVALUATOR.register()
class HitRate(Evaluator):
    def __init__(self, cfg, top_k):
        super(HitRate, self).__init__(cfg)
        self.top_k = top_k

    def calculate(self, preds, scores):
        preds += torch.randn_like(preds) * 1e-6
        return (torch.argsort(preds, dim=-1, descending=True)[:, 0] < self.top_k).float().mean()


@EVALUATOR.register()
class NDCG(Evaluator):
    def __init__(self, cfg, k):
        super(NDCG, self).__init__(cfg)
        self.k = k

    def calculate(self, preds, scores):
        preds += torch.randn_like(preds) * 1e-6
        rank = torch.argsort(preds, dim=-1, descending=True)[:, 0]
        ndcg = 1 / (torch.log2(rank + 2))
        ndcg[rank >= self.k] = 0
        return ndcg.mean()


if __name__ == "__main__":
    from netease_rank.config import BaseConfig
    from netease_rank.data import DataSource
    from netease_rank.model import MLP
    from torch.utils.data import DataLoader

    cfg = BaseConfig()
    ds = DataSource(cfg)
    model = MLP(cfg, ds.cardinality)
    hr = HitRate(cfg, 10)
    hr.eval(model, ds)
    ndcg = NDCG(cfg, 10)
    ndcg.eval(model, ds)
