import torch
import torch.nn as nn


class FourierFFN(nn.Module):
    """
    Feedforward Network pour signaux transformés de Fourier
    (modèle benchmark plus simple que CNN/LSTM)
    """

    def __init__(self,
                 lookback=30,
                 n_assets=40,
                 hidden_units=[30, 16, 8, 4],
                 dropout=0.25):

        super(FourierFFN, self).__init__()

        self.lookback = lookback
        self.n_assets = n_assets
        self.hidden_units = hidden_units

        # Couches cachées
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_units) - 1):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_units[i], hidden_units[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))

        # Couche finale
        self.final_layer = nn.Linear(hidden_units[-1], 1)
        self.tanh = nn.Tanh()  # Pour contraindre les poids entre -1 et 1

    def forward(self, x):
        # x: (batch, lookback * n_assets) ou (batch, lookback)
        batch_size = x.shape[0]

        for layer in self.hidden_layers:
            x = layer(x)

        x = self.final_layer(x)
        x = self.tanh(x)  # Poids entre -1 et 1

        return x.squeeze()