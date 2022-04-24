import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, hidden_dims, include_final=True):
        super().__init__()
        layers = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
        if include_final:
            layers.append(nn.Linear(hidden_dims[-1], 1))
        self.mlp_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp_layers(x)


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


class CrossNet(nn.Module):
    def __init__(self, in_features, layers=2):
        super().__init__()
        self.layers = layers
        self.kernels = nn.Parameter(torch.Tensor(self.layers, in_features, 1))
        self.biases = nn.Parameter(torch.Tensor(self.layers, in_features, 1))
        for i in range(self.kernels.shape[0]):
            nn.init.xavier_normal_(self.kernels[i])
        for i in range(self.biases.shape[0]):
            nn.init.zeros_(self.biases[i])

    def forward(self, inputs):
        """
        :param inputs: [*, dim]
        :return: [*, dim]
        """
        leading_dims = inputs.size()[:-1]
        inputs = inputs.flatten(0, -2)
        x_0 = inputs.unsqueeze(-1)
        x_l = x_0
        for i in range(self.layers):
            xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
            dot_ = torch.matmul(x_0, xl_w)
            x_l = dot_ + self.biases[i] + x_l
        x_l = x_l.squeeze(-1)
        x_l = x_l.view(*[dim for dim in leading_dims], -1)
        return x_l


class CIN(nn.Module):
    def __init__(self, field_size, layer_size=(128, 128)):
        super().__init__()
        self.layer_size = layer_size
        self.field_nums = [field_size]
        self.conv1ds = nn.ModuleList()
        for i, size in enumerate(self.layer_size):
            self.conv1ds.append(
                nn.Conv1d(self.field_nums[-1] * self.field_nums[0], size, 1))

            if i != len(self.layer_size) - 1 and size % 2 > 0:
                raise ValueError(
                    "layer_size must be even number except for the last layer when split_half=True")
            self.field_nums.append(size // 2)
        self.out_dim = sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]

    def forward(self, cin_inputs):
        """
        :param cin_inputs: [*, field_size, dim]
        :return: [*, out_dim], out_dim = sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]
        """
        leading_dims = cin_inputs.size()[:-2]
        cin_inputs = cin_inputs.flatten(0, -3)  # [b, f, d]
        bs = cin_inputs.size(0)
        dim = cin_inputs.size(-1)
        hidden_outputs = [cin_inputs]
        final_results = []
        for i, size in enumerate(self.layer_size):
            x = torch.einsum('bhd,bmd->bhmd', hidden_outputs[-1], hidden_outputs[0])
            x = x.reshape(bs, hidden_outputs[-1].size(1) * hidden_outputs[0].size(1), dim)
            x = self.conv1ds[i](x)
            x = F.relu(x, inplace=True)
            if i != len(self.layer_size) - 1:
                next_hidden, direct_connect = torch.split(x, 2 * [size // 2], 1)
            else:
                direct_connect = x
                next_hidden = 0
            final_results.append(direct_connect)
            hidden_outputs.append(next_hidden)

        result = torch.cat(final_results, dim=1).sum(-1)
        result = result.view(*[d for d in leading_dims], -1)
        return result


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
        logits = None
        for i, name in enumerate(self.sparse_feat_names):
            if logits is None:
                logits = self.sparse_weight[name](sparse_feat[..., i])
            else:
                logits += self.sparse_weight[name](sparse_feat[..., i])
        logits += self.dense_fc(dense_feat)
        return logits.squeeze(-1)
