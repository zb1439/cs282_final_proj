import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, hidden_dims, include_final=True):
        super().__init__()
        layers = [nn.Linear(self.user_input_dim + self.item_input_dim, hidden_dims[0])]
        for in_dim, out_dim in hidden_dims[1:-1], hidden_dims[2:]:
            layers.append(nn.Linear(in_dim, out_dim))
        if include_final:
            layers.append(nn.Linear(hidden_dims[-1], 1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        return self.layers(x)


class FM(nn.Module):
    def forward(self, fm_input):
        """
        :param fm_input: tensor of shape [*, field_size, dim]
        :return: tensor of size [*]
        """
        square_of_sum = torch.pow(torch.sum(fm_input, dim=-2, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=-2, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=-1, keepdim=False)
        return cross_term.squeeze(-1)


class FieldwiseLinear(nn.Module):
    def __init__(self, sparse_feat_names, dense_dims, cardinality):
        """
        Linear layer which maps each sparse field and dense entry to the final output logit
        :param sparse_feat_names: sparse feature names
        :param dense_dims: number of dense entries
        :param cardinality: dictionary where keys are sparse feature names and values are their cardinality
        """
        super().__init__()
        emb_dict = {name: nn.Embedding(cardinality[name], 1) for name in sparse_feat_names}
        self.sparse_feat_names = sparse_feat_names
        self.sparse_weight = nn.ModuleDict(emb_dict)
        self.dense_fc = nn.Linear(dense_dims, 1, bias=False)

    def forward(self, raw_feat):
        """
        :param raw_feat: Raw input tensor before embedding layers in base model of shape [*, dim],
                         where the first len(self.sparse_feat_names) are expected to be the sparse non-encoded entries.
        :return: Logits of shape [*]
        """
        sparse_feat = raw_feat[..., :len(self.sparse_feat_names)].long()
        dense_feat = raw_feat[..., len(self.sparse_feat_names):]
        logits = torch.zeros((len(raw_feat), 1), device=raw_feat.device)
        for i, name in enumerate(self.sparse_feat_names):
            logits += self.sparse_weight[name](sparse_feat[..., i])
        logits += self.dense_fc(dense_feat)
        return logits.squeeze(-1)
