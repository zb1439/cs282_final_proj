from turtle import Turtle
import torch
import torch.nn as nn
import numpy as np

from netease_rank.model import BaseModel, MODELS
from netease_rank.model.layers import NCF as NCF_layer


@MODELS.register()
class SelfAttentionNCF(BaseModel):
    def _init_modules(self):
        hidden_dims = self.cfg.MODEL.DEEP.HIDDEN_DIMS
        self.ncf = NCF_layer([self.user_input_dim + self.item_input_dim] + hidden_dims, self.cfg.MODEL.NCF.GMF_DIM, self.user_sparse_dim, self.item_sparse_dim)
        self.linear = nn.Linear(hidden_dims[-1] + self.cfg.MODEL.NCF.GMF_DIM, hidden_dims[-1])
        self.after_attn = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_dims[-1], 8, batch_first=True), 3)
        self.predictor = nn.Linear(hidden_dims[-1], 1)

    def predict(self, user_emb, item_emb):
        emb = self.ncf(user_emb, item_emb)
        emb = self.linear(emb)
        emb = self.after_attn(emb)
        logits = self.predictor(emb).squeeze(-1)
        return logits
