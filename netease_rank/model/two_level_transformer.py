import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel, MODELS
from .layers import MLP


@MODELS.register()
class TwoLevelTransformer(BaseModel):
    def _init_modules(self):
        assert self.small_dim != "auto" and self.small_dim == self.large_dim, \
            "Transformer requires all fields to have same length embeddings!"
        hidden_dims = self.cfg.MODEL.DEEP.HIDDEN_DIMS
        self.deep_layers = MLP([self.user_input_dim + self.item_input_dim] + hidden_dims, include_final=False)
        last_dim = self.item_sparse_dim + self.user_sparse_dim + hidden_dims[-1]
        self.field_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.small_dim, 8, batch_first=True), 3)
        self.feat_merger = nn.Linear(last_dim, self.small_dim * 8)
        self.items_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.small_dim * 8, 8, batch_first=True), 3)
        self.tfm_linear = nn.Linear(self.small_dim * 8, 1)

    def predict(self, user_feat, item_feat):
        sparse_user_feat = user_feat[..., :self.user_sparse_dim]
        sparse_item_feat = item_feat[..., :self.item_sparse_dim]
        sparse_feat = torch.cat([sparse_user_feat, sparse_item_feat], -1)
        sparse_feat = sparse_feat.view(-1, sparse_feat.size(2) // self.small_dim, self.small_dim)

        mlp_feat = self.deep_layers(torch.cat([user_feat, item_feat], -1)).flatten(0, 1)
        field_attn_feat = self.field_transformer(sparse_feat).flatten(-2)
        feat = torch.cat([mlp_feat, field_attn_feat], dim=-1)
        feat = F.relu(self.feat_merger(feat), inplace=True).view(item_feat.size(0), item_feat.size(1), -1)

        out = self.items_transformer(feat)
        logits = self.tfm_linear(out).squeeze(-1)
        return logits
