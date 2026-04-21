import torch
import torch.nn as nn


class RawFFN(nn.Module):
    """
    Baseline simple : FFN sur résidus bruts
    """

    def __init__(self, lookback=30, n_assets=40, hidden_units=[30, 16, 8, 4], dropout=0.25):
        super().__init__()

        self.lookback = lookback
        self.n_assets = n_assets

        layers = []
        prev_dim = lookback

        for h in hidden_units:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch * n_assets, lookback)
        return self.net(x).squeeze()