import torch
import torch.nn as nn

from .base_model import BaseModel, MODELS
from .layers import MLP as MLP_layer


@MODELS.register()
class MLP(BaseModel):
    def _init_modules(self):
        hidden_dims = self.cfg.MODEL.DEEP.HIDDEN_DIMS
        self.mlp = MLP_layer([self.user_input_dim + self.item_input_dim] + hidden_dims)

    def predict(self, user_emb, item_emb):
        emb = torch.cat([user_emb, item_emb], -1)
        logits = self.mlp(emb).squeeze(-1)
        return logits
