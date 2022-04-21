import torch
import torch.nn as nn

from .base_model import BaseModel, MODELS
from .layers import MLP


@MODELS.register()
class NCF(BaseModel):
    def _init_modules(self):
        hidden_dims = self.cfg.MODEL.DEEP.HIDDEN_DIMS
        gmf_dim = self.cfg.MODEL.NCF.GMF_DIM
        self.deep_layers = MLP([self.user_input_dim + self.item_input_dim] + hidden_dims, include_final=False)
        self.user_gmf = nn.Linear(self.user_sparse_dim, gmf_dim)
        self.item_gmf = nn.Linear(self.item_sparse_dim, gmf_dim)
        self.predictor = nn.Linear(gmf_dim + hidden_dims[-1], 1)

    def predict(self, user_emb, item_emb):
        dnn_feat = self.deep_layers(torch.cat([user_emb, item_emb], -1))
        user_gmf = self.user_gmf(user_emb[..., :self.user_sparse_dim])
        item_gmf = self.item_gmf(item_emb[..., :self.item_sparse_dim])
        gmf_feat = user_gmf * item_gmf
        return self.predictor(torch.cat([gmf_feat, dnn_feat], -1)).squeeze(-1)
