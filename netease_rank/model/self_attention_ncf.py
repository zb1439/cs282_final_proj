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
        self.after_attn = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.cfg.TRAINING.ITEM_SIZE, 8, batch_first=False), 3)
        self.predictor = nn.Linear(self.cfg.TRAINING.ITEM_SIZE, self.cfg.TRAINING.ITEM_SIZE)

    def predict(self, user_emb, item_emb):
        emb = self.ncf(user_emb, item_emb)
        emb = self.after_attn(emb)
        logits = self.predictor(emb).squeeze(-1)
        return logits
