import logging
import torch
from torch.utils.data import DataLoader

from netease_rank.utils import Registry


EVALUATOR = Registry("EVALUATOR")


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bs = cfg.TRAINING.BATCH_SIZE

    def eval(self, model, data_source):
        model.eval()
        data_source.test_mode = True
        loader = DataLoader(data_source, batch_size=self.bs, shuffle=False, num_workers=4)

        all_preds, all_scores = None, None
        for i, data in enumerate(loader):
            user_feat, item_feat, scores = data
            pred_scores = model(user_feat, item_feat)
            if all_preds is None:
                all_preds = pred_scores
                all_scores = scores
            else:
                all_preds = torch.cat([all_preds, pred_scores], 0)
                all_scores = torch.cat([all_scores, scores], 0)
        self.summarize(all_preds, all_scores)

        model.train()
        data_source.test_mode = False

    def summarize(self, preds, scores):
        raise NotImplementedError
