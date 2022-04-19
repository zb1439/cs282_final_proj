import torch
import torch.nn as nn

from .base_model import BaseModel, MODELS
from .layers import MLP, FM


@MODELS.register()
class DeepFM(BaseModel):
    def _init_modules(self):
        assert self.small_dim != "auto" and self.small_dim == self.large_dim, \
            "DeepFM requires all fields to have same length embeddings!"
        hidden_dims = self.cfg.MODEL.DEEP.HIDDEN_DIMS
        self.deep_layers = MLP(hidden_dims)
        self.fm = FM()

    def predict(self, user_feat, item_feat):
        sparse_user_feat = user_feat[..., :self.user_sparse_dim]
        sparse_item_feat = item_feat[..., :self.item_sparse_dim]
        sparse_feat = torch.cat([sparse_user_feat, sparse_item_feat], -1)
        sparse_feat = sparse_feat.view(sparse_feat.size(0), sparse_feat.size(1),
                                       sparse_feat.size(2) // self.small_dim, self.small_dim)
        logits = self.fm(sparse_feat)
        logits += self.deep_layers(torch.cat([user_feat, item_feat], -1))
        return logits
