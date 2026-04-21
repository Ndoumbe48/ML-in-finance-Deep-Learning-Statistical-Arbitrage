import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float)


def train(model,
          data_train,
          data_dev=None,
          num_epochs=100,
          batchsize=200,
          lr=0.001,
          trans_cost=0,
          hold_cost=0,
          lookback=30,
          objective="sharpe",
          output_path=None,
          model_tag='',
          device='cpu'):
    if output_path is None:
        output_path = getattr(model, 'logdir', "results")

    T, N = data_train.shape
    logging.info(f"train(): data_train.shape {data_train.shape}")

    # Préparer les séquences
    windows, idxs_selected = prepare_sequences(data_train, lookback)
    logging.info(f"train(): windows.shape {windows.shape}")

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    begin_time = time.time()

    for epoch in range(num_epochs):
        all_returns = []

        # Réinitialiser prev_weights pour chaque epoch
        prev_weights = None

        # Entraînement par batches
        for i in range(0, T - lookback, batchsize):
            batch_start = i
            batch_end = min(i + batchsize, T - lookback)

            # Préparer les données du batch
            batch_windows = windows[batch_start:batch_end]
            batch_returns = data_train[lookback + batch_start:lookback + batch_end]

            # Forward pass
            weights = model(torch.tensor(batch_windows, device=device))

            # Normalisation L1
            abs_sum = torch.sum(torch.abs(weights), dim=1, keepdim=True)
            weights = weights / (abs_sum + 1e-8)

            # Rendements du portefeuille
            portfolio_returns = torch.sum(weights * torch.tensor(batch_returns, device=device), dim=1)

            # Coûts de transaction
            if trans_cost > 0 and prev_weights is not None and weights.shape == prev_weights.shape:
                turnover = torch.sum(torch.abs(weights - prev_weights), dim=1)
                portfolio_returns = portfolio_returns - trans_cost * turnover

            # Coûts de holding (positions courtes)
            if hold_cost > 0:
                short_exposure = torch.sum(torch.abs(torch.min(weights, torch.zeros_like(weights))), dim=1)
                portfolio_returns = portfolio_returns - hold_cost * short_exposure

            # Loss function
            mean_ret = torch.mean(portfolio_returns)
            std_ret = torch.std(portfolio_returns)

            if objective == "sharpe":
                loss = -mean_ret / (std_ret + 1e-8)
            elif objective == "sortino":
                downside = portfolio_returns[portfolio_returns < 0]
                downside_std = torch.std(downside) if len(downside) > 0 else std_ret
                loss = -mean_ret / (downside_std + 1e-8)
            else:
                loss = -mean_ret

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Mettre à jour prev_weights pour le prochain batch
            prev_weights = weights.detach().clone()

            all_returns.extend(portfolio_returns.detach().cpu().numpy())

        if epoch % 10 == 0:
            mean_return = np.mean(all_returns) * 252
            std_return = np.std(all_returns) * np.sqrt(252)
            sharpe = mean_return / (std_return + 1e-8)
            logging.info(f"Epoch {epoch}: Sharpe = {sharpe:.2f}, Return = {mean_return:.2%}")

    return model


def get_returns(model,
                data_test,
                lookback=30,
                trans_cost=0,
                hold_cost=0,
                device='cpu'):
    T, N = data_test.shape
    windows, idxs_selected = prepare_sequences(data_test, lookback)

    model.eval()

    with torch.no_grad():
        weights = model(torch.tensor(windows, device=device))

        # Normalisation L1
        abs_sum = torch.sum(torch.abs(weights), dim=1, keepdim=True)
        weights = weights / (abs_sum + 1e-8)

        # Rendements
        returns = torch.sum(weights * torch.tensor(data_test[lookback:], device=device), dim=1)

        # Coûts
        turnover = torch.cat((torch.zeros(1, device=device),
                              torch.sum(torch.abs(weights[1:] - weights[:-1]), dim=1)))
        short_exposure = torch.sum(torch.abs(torch.min(weights, torch.zeros_like(weights))), dim=1)

        returns = returns - trans_cost * turnover - hold_cost * short_exposure

    return returns.cpu().numpy(), weights.cpu().numpy()


def prepare_sequences(data, lookback=30):
    """
    Prépare les séquences pour le modèle LSTM/CNN
    data: (T, N) matrice des résidus
    """
    T, N = data.shape
    cumulative = np.cumsum(data, axis=0)

    windows = []
    idxs_selected = []

    for i in range(lookback, T):
        seq = cumulative[i - lookback:i, :]
        windows.append(seq)
        # Vérifier qu'il n'y a pas de nan
        valid = ~np.any(np.isnan(seq), axis=0)
        idxs_selected.append(valid)

    windows = np.array(windows, dtype=np.float32)
    idxs_selected = np.array(idxs_selected)

    return windows, idxs_selected


def test_strategy(data,
                  daily_dates,
                  model_class,
                  model_config,
                  lookback=30,
                  train_window=500,
                  retrain_freq=250,
                  trans_cost=0.0005,
                  hold_cost=0.0001,
                  objective='sharpe',
                  output_path='results'):
    T, N = data.shape
    all_returns = []
    all_weights = []

    for start_idx in range(0, T - train_window - lookback, retrain_freq):
        train_end = start_idx + train_window
        test_end = min(train_end + retrain_freq, T - lookback)

        if test_end <= train_end:
            break

        logging.info(f"Training: {start_idx} to {train_end}, Testing: {train_end} to {test_end}")

        # Entraînement
        train_data = data[start_idx:train_end]
        model = model_class(**model_config)
        model = train(model, train_data, lookback=lookback,
                      trans_cost=trans_cost, hold_cost=hold_cost,
                      objective=objective, num_epochs=20)

        # Test
        test_data = data[train_end:test_end]
        returns, weights = get_returns(model, test_data, lookback,
                                       trans_cost=trans_cost, hold_cost=hold_cost)

        all_returns.extend(returns)
        all_weights.extend(weights)

    # Métriques finales
    all_returns = np.array(all_returns)
    sharpe = all_returns.mean() / (all_returns.std() + 1e-8) * np.sqrt(252)
    sortino = all_returns.mean() / (all_returns[all_returns < 0].std() + 1e-8) * np.sqrt(252)

    logging.info(f"Final Sharpe: {sharpe:.2f}, Sortino: {sortino:.2f}")

    return all_returns, all_weights