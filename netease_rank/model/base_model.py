import numpy as np
import torch
import torch.nn as nn

from netease_rank.utils import Registry
from .layers import FieldwiseLinear


MODELS = Registry("MODELS")


@MODELS.register()
class BaseModel(nn.Module):
    def __init__(self, cfg, cardinality):
        self.cfg = cfg
        self.user_sparse_feature_names = cfg.FEATURE_ENG.DISCRETE_COLS.USER
        self.item_sparse_feature_names = cfg.FEATURE_ENG.DISCRETE_COLS.ITEM
        self.small_dim = cfg.MODEL.SMALL_EMB_DIM
        self.large_dim = cfg.MODEL.LARGE_EMB_DIM
        self.cardinality = cardinality
        self.enable_linear = cfg.MODEL.FIELDWISE_LINEAR

        self._init_embededdings()
        self._init_modules()
        if self.enable_linear:
            self.user_linear_branch = FieldwiseLinear(
                self.user_sparse_feature_names, self.user_input_dim - self.user_sparse_dim, self.cardinality)
            self.item_linear_branch = FieldwiseLinear(
                self.item_sparse_feature_names, self.item_input_dim - self.item_sparse_dim, self.cardinality)

    def _init_embededdings(self):
        emb_dict = {}
        self.user_input_dim = len(self.cfg.FEATURE_ENG.CONTINUOUS_DIM.USER)
        self.item_input_dim = len(self.cfg.FEATURE_ENG.CONTINUOUS_DIM.ITEM)
        for name in self.user_sparse_feature_names:
            if name == "userIdx":
                emb_dict[name] = nn.Embedding(self.cardinality[name], self.large_dim)
                self.user_input_dim += self.large_dim
            else:
                dim = self.small_dim if self.small_dim != "auto" else 6 * int(self.cardinality[name] ** 0.25)
                emb_dict[name] = nn.Embedding(self.cardinality[name], dim)
                self.user_input_dim += dim
        for name in self.item_sparse_feature_names:
            if name == "mlogindex":
                emb_dict[name] = nn.Embedding(self.cardinality[name], self.large_dim)
                self.item_input_dim += self.large_dim
            else:
                dim = self.small_dim if self.small_dim != "auto" else 6 * int(self.cardinality[name] ** 0.25)
                emb_dict[name] = nn.Embedding(self.cardinality[name], dim)
                self.item_input_dim += dim
        self.embeddings = nn.ModuleDict(emb_dict)
        self.user_sparse_dim = self.user_input_dim - len(self.cfg.FEATURE_ENG.CONTINUOUS_DIM.USER)
        self.item_sparse_dim = self.item_input_dim - len(self.cfg.FEATURE_ENG.CONTINUOUS_DIM.ITEM)

    def _init_modules(self):
        raise NotImplementedError

    def embed(self, user_feat, item_feat):
        user_sparse_feat = user_feat[:len(self.user_sparse_feature_names)].long()
        user_emb_feat = user_feat[len(self.user_dense_feature_names):]
        item_sparse_feat = item_feat[:len(self.item_sparse_feature_names)].long()
        item_emb_feat = item_feat[len(self.item_sparse_feature_names):]

        for i, name in enumerate(self.user_sparse_feature_names):
            user_emb_feat = torch.cat([user_emb_feat, self.embeddings[name](user_sparse_feat[:, i])], -1)
        for i, name in enumerate(self.item_sparse_feature_names):
            item_emb_feat = torch.cat([item_emb_feat, self.embeddings[name](item_sparse_feat[:, :, i])], -1)
        assert user_emb_feat.size(-1) == self.user_input_dim
        assert item_emb_feat.size(-1) == self.item_input_dim

        user_emb_feat = torch.tile(user_emb_feat[:, None], (1, item_emb_feat.size(1), 1))
        return user_emb_feat, item_emb_feat

    def linear_predict(self, user_feat, item_feat):
        user_logits = self.user_linear_branch(user_feat)
        item_logits = self.item_linear_branch(item_feat)
        return user_logits[:, None] + item_logits

    def predict(self, user_emb, item_emb):
        raise NotImplementedError

    def forward(self, user_feat, item_feat):
        user_emb_feat, item_emb_feat = self.embed(user_feat, item_feat)
        logit_matrix = self.predict(user_emb_feat, item_emb_feat)
        if self.enable_linear:
            logit_matrix += self.linear_predict(user_feat, item_feat)
        return logit_matrix
