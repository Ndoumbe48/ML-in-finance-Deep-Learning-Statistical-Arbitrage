import torch
import torch.nn as nn


class OUFFN(nn.Module):
    """
    Baseline : FFN sur paramètres Ornstein-Uhlenbeck
    (kappa, mu, sigma, X_L, R²)
    """

    def __init__(self, n_assets=40, hidden_units=[5, 4, 4, 4], dropout=0.25):
        super().__init__()

        self.n_assets = n_assets

        layers = []
        prev_dim = 5  # 5 paramètres OU (kappa, mu, sigma, X_L, R²)

        for h in hidden_units:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Sigmoid())  # OU utilise Sigmoid (valeurs entre 0 et 1)
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch * n_assets, 5)
        return self.net(x).squeeze()