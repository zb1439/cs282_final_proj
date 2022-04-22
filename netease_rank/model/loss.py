import torch
import torch.nn as nn
import torch.nn.functional as F

from netease_rank.utils import Registry


LOSS = Registry("LOSS")


class Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, logits, scores):
        raise NotImplementedError


@LOSS.register()
class MSE(Loss):
    def forward(self, logits, scores):
        return F.mse_loss(logits, scores)


@LOSS.register()
class RankNet(Loss):
    def forward(self, logits, scores):
        logit_mat = logits[..., None] - logits[:, None]
        label_mat = scores[..., None] - scores[:, None]
        label_mat[label_mat < 0] = -1.
        label_mat[label_mat > 0] = 1.
        return F.binary_cross_entropy_with_logits(logit_mat, 0.5 * (1 + label_mat))


@LOSS.register()
class BCE(Loss):
    def forward(self, logits, scores):
        labels = (scores > 0).float()
        return F.binary_cross_entropy_with_logits(logits, labels)
