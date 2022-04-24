import torch
import torch.nn as nn

from .base_model import BaseModel, MODELS
from .layers import MLP


@MODELS.register()
class Transformer(BaseModel):
    def _init_modules(self):
        assert self.small_dim != "auto" and self.small_dim == self.large_dim, \
            "Transformer requires all fields to have same length embeddings!"
        hidden_dims = self.cfg.MODEL.DEEP.HIDDEN_DIMS
        self.deep_layers = MLP([self.user_input_dim + self.item_input_dim] + hidden_dims)
        self.transformer = nn.Transformer(self.small_dim, num_encoder_layers=3, num_decoder_layers=1, batch_first=True)
        self.query = nn.Parameter(torch.Tensor(1, self.small_dim))
        nn.init.xavier_normal_(self.query)
        self.tfm_linear = nn.Linear(self.small_dim, 1)

    def predict(self, user_feat, item_feat):
        sparse_user_feat = user_feat[..., :self.user_sparse_dim]
        sparse_item_feat = item_feat[..., :self.item_sparse_dim]
        sparse_feat = torch.cat([sparse_user_feat, sparse_item_feat], -1)
        sparse_feat = sparse_feat.view(-1, sparse_feat.size(2) // self.small_dim, self.small_dim)
        out = self.transformer(sparse_feat, self.query.data.unsqueeze(0).tile(sparse_feat.size(0), 1, 1))
        out = out.squeeze(-2).view(user_feat.size(0), -1, self.small_dim)
        logits = self.tfm_linear(out).squeeze(-1)
        logits += self.deep_layers(torch.cat([user_feat, item_feat], -1)).squeeze(-1)
        return logits
