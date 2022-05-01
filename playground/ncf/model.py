import torch
import torch.nn as nn

from netease_rank.model import BaseModel, MODELS
from netease_rank.model.layers import MLP


@MODELS.register()
class SelfAttentionNCF2(BaseModel):
    def _init_modules(self):
        hidden_dims = self.cfg.MODEL.DEEP.HIDDEN_DIMS
        gmf_dim = self.cfg.MODEL.NCF.GMF_DIM
        self.deep_layers = MLP([self.user_input_dim + self.item_input_dim] + hidden_dims, include_final=True)
        self.user_gmf = nn.Linear(self.user_sparse_dim, gmf_dim)
        self.item_gmf = nn.Linear(self.item_sparse_dim, gmf_dim)
        self.item_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(gmf_dim, 8, batch_first=True), 3)
        self.predictor = nn.Linear(gmf_dim, 1)

    def predict(self, user_emb, item_emb):
        user_gmf = self.user_gmf(user_emb[..., :self.user_sparse_dim])
        item_gmf = self.item_gmf(item_emb[..., :self.item_sparse_dim])
        gmf_feat = user_gmf * item_gmf
        logits = self.predictor(self.item_transformer(gmf_feat))
        logits += self.deep_layers(torch.cat([user_emb, item_emb], -1))
        logits = logits.squeeze(-1)
        return logits
