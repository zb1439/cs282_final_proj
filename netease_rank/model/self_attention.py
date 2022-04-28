import torch
import torch.nn as nn

from .base_model import BaseModel, MODELS
from .layers import MLP as MLP_layer


@MODELS.register()
class SelfAttention(BaseModel):
    def _init_modules(self):
        hidden_dims = self.cfg.MODEL.DEEP.HIDDEN_DIMS
        self.mlp = MLP_layer([self.user_input_dim + self.item_input_dim] + hidden_dims, include_final=False)
        self.after_attn = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_dims[-1], 8, batch_first=False), 3)
        self.predictor = nn.Linear(hidden_dims[-1], 1)

    def predict(self, user_emb, item_emb):
        emb = torch.cat([user_emb, item_emb], -1)
        emb = self.mlp(emb)
        emb = self.after_attn(emb)
        logits = self.predictor(emb).squeeze(-1)
        return logits
