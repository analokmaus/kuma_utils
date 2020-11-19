import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLinear(nn.Module):
    def __init__(self, 
                 in_features, out_features,
                 bias=True, activation=nn.ReLU(), 
                 batch_norm=False, p=0.5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.relu = activation
        self.dobn = batch_norm
        self.bn = nn.BatchNorm1d(in_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        if self.dobn:
            x = self.bn(x)
        x = self.drop(x)
        return x


class TabularNet(nn.Module):
    '''
    Smart dense neural network for tabular data.
    # Input
    categorical_features: list of int  = indice of categorical features
    emb_dims: unique value count, output dimension), ...]
    '''

    def __init__(self, 
                 in_features ,out_features=2,
                 categorical_features=[], emb_dims=[],
                 hidden_dims=[256, 128, 64, 32], 
                 dropout_ratios=[0.5, 0.5, 0.5], 
                 batch_norms=[0, 0, 0],
                 emb_dropout=0.5):
        super().__init__()
        
        # Categorical features: embedding
        assert len(categorical_features) == len(emb_dims)
        self.cat_idx = categorical_features
        self.lin_idx = [x for x in range(in_features) if x not in self.cat_idx]
        self.emb_layers = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in emb_dims])
        emb_len = sum([y for x, y in emb_dims]) # calc total dims
        self.emb_len = emb_len
        self.lin_len = len(self.lin_idx)
        self.emb_dropout_layer = nn.Dropout(emb_dropout)

        # Continuous features: custom linear
        assert len(hidden_dims) == len(dropout_ratios) + 1
        assert len(dropout_ratios) == len(batch_norms)
        self.first_layer = CustomLinear(
            self.emb_len + self.lin_len, hidden_dims[0], p=0.0)

        self.lin_layers = nn.ModuleList([
            CustomLinear(hidden_dims[i], hidden_dims[i+1], 
                         batch_norm=batch_norms[i], 
                         p=dropout_ratios[i]) for i in range(len(hidden_dims)-1)])

        self.output_layer = nn.Linear(hidden_dims[-1], out_features)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

    def forward(self, X):
        if self.emb_len > 0:
            x = [emb_layer(X[:, self.cat_idx[i]].long())
                 for i, emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, 1)
            x = self.emb_dropout_layer(x)
        else:
            x = torch.Tensor().to(X.device)

        if self.lin_len > 0:
            x = torch.cat([x, X[:, self.lin_idx]], 1)
        
        x = self.first_layer(x)
        for lin_layer in self.lin_layers:
            x = lin_layer(x)
        x = self.output_layer(x)

        return x
