import torch
import torch.nn as nn

from .base_model import MODELS, BaseModel
from .layers import *


@MODELS.register()
class DeepAndCross(BaseModel):
    def _init_modules(self):
        hidden_dims = self.cfg.MODEL.DEEP.HIDDEN_DIMS
        self.deep_layers = MLP([self.user_input_dim + self.item_input_dim] + hidden_dims)
        self.cross = CrossNet(self.user_input_dim + self.item_input_dim)
        self.cross_linear = nn.Linear(self.user_input_dim + self.item_input_dim, 1)

    def predict(self, user_emb, item_emb):
        input_feat = torch.cat([user_emb, item_emb], -1)
        logits = self.deep_layers(input_feat).squeeze(-1) + self.cross_linear(self.cross(input_feat)).squeeze(-1)
        return logits
