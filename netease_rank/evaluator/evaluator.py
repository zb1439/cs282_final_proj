import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bs = cfg.TRAINING.BATCH_SIZE
        self.evaluators = {name: eval(name)(cfg, **kwargs) for name, kwargs in cfg.EVALUATION.EVALUATORS}

    def eval(self, model, data_source, limit=-1):
        model.eval()
        data_source.test_mode = True
        loader = DataLoader(data_source, batch_size=self.bs,
                            shuffle=False, num_workers=self.cfg.GLOBAL.NUM_WORKERS)

        metrics = {name: [] for name in self.evaluators.keys()}
        for i, data in enumerate(loader):
            user_feat, item_feat, scores = data
            if torch.cuda.is_available():
                user_feat = user_feat.cuda()
                item_feat = item_feat.cuda()
                scores = scores.cuda()

            pred_scores = model(user_feat, item_feat)
            for n, ev in self.evaluators.items():
                score = ev.calculate(pred_scores, scores)
                if torch.cuda.is_available():
                    score = score.cpu()
                metrics[n].append(score)
            if limit > 0 and i == limit:
                break
            if (i + 1) % 100 == 0:
                print(f"{i+1} out of {len(loader)} evaluated")

        model.train()
        data_source.test_mode = False
        return {name: np.mean(met) for name, met in metrics.items()}

    def calculate(self, preds, scores):
        raise NotImplementedError


class HitRate:
    def __init__(self, cfg, top_k):
        self.top_k = top_k

    def calculate(self, preds, scores):
        preds += torch.randn_like(preds) * 1e-6
        return (torch.argsort(preds, dim=-1, descending=True)[:, 0] < self.top_k).float().mean()


class NDCG:
    def __init__(self, cfg, k):
        self.k = k

    def calculate(self, preds, scores):
        preds += torch.randn_like(preds) * 1e-6
        rank = torch.argsort(preds, dim=-1, descending=True)[:, 0]
        ndcg = 1 / (torch.log2(rank + 2))
        ndcg[rank >= self.k] = 0
        return ndcg.mean()

class MultiHitRate:
    def __init__(self, cfg, top_k, ninteract_user):
        self.top_k = top_k
        self.ninteract_user = ninteract_user
    
    def calculate(self, preds, scores):
        preds += torch.rand_like(preds) * 1e-6
        hits = torch.argsort(preds, dim=-1, descending=True)[:, :self.ninteract_user] < self.top_k
        return hits.float().mean()

class MultiNDCG:
    def __init__(self, cfg, top_k, ninteract_user):
        self.top_k = top_k
        self.ninteract_user = ninteract_user
    
    def calculate(self, preds, scores):
        discount = 1 / (np.log2(np.arange(self.top_k) + 2))
        preds += torch.randn_like(preds) * 1e-6
        rank_score, index = torch.sort(preds.detach(), dim=-1, descending=True)
        rank_score = torch.zeros(index.shape)
        ranks = rank_score.scatter(1, index, scores)
        idcg = torch.sum((2**scores[:, :self.top_k] - 1) * discount, dim=-1) + 1e-5
        dcg = torch.sum((2**ranks[:, :self.top_k] - 1) * discount, dim=-1)
        return torch.mean(dcg / idcg)


if __name__ == "__main__":
    from netease_rank.config import BaseConfig
    from netease_rank.data import DataSource
    from netease_rank.model import MLP
    from torch.utils.data import DataLoader

    cfg = BaseConfig()
    ds = DataSource(cfg)

    model = MLP(cfg, ds.cardinality)
    # hr = HitRate(cfg, 10)
    # hr.eval(model, ds)
    # ndcg = NDCG(cfg, 10)
    # ndcg.eval(model, ds)
    eval = Evaluator(cfg)
    
    eval.eval(model, ds)
