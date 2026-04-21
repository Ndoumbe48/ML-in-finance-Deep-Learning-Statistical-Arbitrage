import torch
import torch.nn as nn


class CNN_Block(nn.Module):
    """
    Bloc convolutionnel avec 2 couches et connection résiduelle
    """

    def __init__(self, in_filters=1, out_filters=8, filter_size=2):
        super(CNN_Block, self).__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.conv1 = nn.Conv1d(in_channels=in_filters, out_channels=out_filters,
                               kernel_size=filter_size, stride=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=out_filters, out_channels=out_filters,
                               kernel_size=filter_size, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.left_zero_padding = nn.ConstantPad1d((filter_size - 1, 0), 0)
        self.normalization = nn.InstanceNorm1d(out_filters)

    def forward(self, x):
        # x: (N, C, T)
        out = self.left_zero_padding(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.normalization(out)

        out = self.left_zero_padding(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.normalization(out)

        # Connection résiduelle (ajuster les dimensions si nécessaire)
        if self.in_filters != self.out_filters:
            x = x.repeat(1, self.out_filters // self.in_filters, 1)
        out = out + x
        return out


class CNNArbitrage(nn.Module):
    """
    CNN simplifié pour l'arbitrage statistique sur cryptos
    """

    def __init__(self,
                 lookback=30,
                 n_assets=40,
                 filter_numbers=[1, 8, 16],
                 filter_size=2,
                 hidden_dim=32,
                 dropout=0.25):

        super(CNNArbitrage, self).__init__()

        self.lookback = lookback
        self.n_assets = n_assets
        self.filter_numbers = filter_numbers

        # Blocs convolutifs
        self.conv_blocks = nn.ModuleList()
        for i in range(len(filter_numbers) - 1):
            self.conv_blocks.append(
                CNN_Block(filter_numbers[i], filter_numbers[i + 1], filter_size)
            )

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # FFN finale
        self.ffn = nn.Sequential(
            nn.Linear(filter_numbers[-1], hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # x: (batch, lookback, n_assets)
        batch_size, lookback, n_assets = x.shape

        # Reshaper pour traiter chaque actif indépendamment
        # (batch * n_assets, 1, lookback)
        x = x.permute(0, 2, 1)  # (batch, n_assets, lookback)
        x = x.reshape(batch_size * n_assets, 1, lookback)  # (batch*n_assets, 1, lookback)

        # CNN
        for conv_block in self.conv_blocks:
            x = conv_block(x)  # (batch*n_assets, C, lookback)

        # Global Average Pooling
        x = self.global_avg_pool(x)  # (batch*n_assets, C, 1)
        x = x.squeeze(-1)  # (batch*n_assets, C)

        # FFN finale
        weights = self.ffn(x)  # (batch*n_assets, 1)
        weights = weights.reshape(batch_size, n_assets)  # (batch, n_assets)

        return weights


# Version encore plus simple (recommandée pour commencer)
class SimpleCNNArbitrage(nn.Module):
    """
    CNN très simple pour l'arbitrage (2 couches convolutives)
    """

    def __init__(self, lookback=30, n_assets=40, cnn_filters=16, cnn_kernel=3):
        super().__init__()

        self.lookback = lookback
        self.n_assets = n_assets

        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_filters, kernel_size=cnn_kernel, padding=1),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size=cnn_kernel, padding=1),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(cnn_filters, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size, lookback, n_assets = x.shape

        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size * n_assets, 1, lookback)

        x = self.cnn(x)
        x = x.squeeze(-1)

        weights = self.fc(x)
        weights = weights.reshape(batch_size, n_assets)

        return weights