from netease_rank.model import BaseModel, MODELS

import torch
import torch.nn as nn

@MODELS.register()
class Identity(BaseModel):
    def __init__(self, cfg, cardinality):
        super().__init__(cfg, cardinality)

    def predict(self, user_emb, item_emb):
        return torch.zeros_like(user_emb)
