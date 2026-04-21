import argparse
import datetime
import logging
import os
import numpy as np
import pandas as pd
import torch

from train_test import test_strategy
from models.CNN import CNNArbitrage
from factor_model.pca import CryptoPCA


def setup_logging(logdir="logs"):
    os.makedirs(logdir, exist_ok=True)
    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f"{logdir}/crypto_arbitrage_{start_time}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return start_time


def load_crypto_residuals(tickers, start_date='2018-01-01', end_date='2024-12-31',
                          factor_list=[5], logdir='residuals'):

    pca = CryptoPCA(logdir=logdir)

    # Charger les données
    returns, dates = pca.load_crypto_data(tickers, start_date, end_date)

    # Calculer les résidus
    residuals_dict = pca.OOSRollingWindowCrypto(
        save=True,
        printOnConsole=True,
        initialOOSYear=int(start_date[:4]) + 2,
        sizeWindow=60,
        sizeCovarianceWindow=252,
        factorList=factor_list
    )

    return residuals_dict, dates, pca.tickers


def run_crypto_arbitrage(tickers,
                         factor=5,
                         objective='sharpe',
                         lookback=30,
                         train_window=500,
                         retrain_freq=250,
                         trans_cost=0.0005,
                         hold_cost=0.0001,
                         model_hidden_dim=32,
                         num_epochs=20,
                         device='cpu'):

    # Lance la stratégie d'arbitrage sur les cryptos

    setup_logging()
    logging.info("=" * 60)
    logging.info("CRYPTO ARBITRAGE STRATEGY")
    logging.info("=" * 60)

    # Charger les résidus PCA
    logging.info("Loading PCA residuals for crypto...")
    residuals_dict, dates, tickers_list = load_crypto_residuals(
        tickers=tickers,
        factor_list=[factor],
        start_date='2018-01-01',
        end_date='2024-12-31'
    )

    residuals = residuals_dict[factor]
    logging.info(f"Residuals shape: {residuals.shape}")

    # Convertir en DataFrame
    residuals_df = pd.DataFrame(
        residuals,
        index=dates[dates.year >= 2020],
        columns=tickers_list
    )
    residuals_df = residuals_df.dropna(axis=1, how='all')
    residuals_df = residuals_df.fillna(0)

    logging.info(f"Final residuals shape: {residuals_df.shape}")

    # Définir le modèle
    model_config = {
        'lookback': lookback,
        'n_assets': len(residuals_df.columns),
        'hidden_dim': model_hidden_dim
    }

    # 4. Lancer le backtest
    returns, weights = test_strategy(
        data=residuals_df.values,
        daily_dates=residuals_df.index,
        model_class=CNNArbitrage,
        model_config=model_config,
        lookback=lookback,
        train_window=train_window,
        retrain_freq=retrain_freq,
        trans_cost=trans_cost,
        hold_cost=hold_cost,
        objective=objective,
        output_path='results/crypto'
    )

    # Afficher les résultats
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
    downside = returns[returns < 0].std()
    sortino = returns.mean() / (downside + 1e-8) * np.sqrt(252) if downside > 0 else 0

    logging.info("=" * 60)
    logging.info("FINAL RESULTS")
    logging.info("=" * 60)
    logging.info(f"Sharpe ratio: {sharpe:.2f}")
    logging.info(f"Sortino ratio: {sortino:.2f}")
    logging.info(f"Mean return (ann.): {returns.mean() * 252:.2%}")
    logging.info(f"Volatility (ann.): {returns.std() * np.sqrt(252):.2%}")

    return returns, weights, {'sharpe': sharpe, 'sortino': sortino}


def main():
    parser = argparse.ArgumentParser(description="Crypto Arbitrage Strategy")
    parser.add_argument("--tickers", "-t", nargs="+",
                        default=[
                            'BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'DOGE-USD',
                            'ADA-USD', 'BNB-USD', 'DOT-USD', 'LINK-USD', 'XLM-USD',
                            'BCH-USD', 'ETC-USD', 'TRX-USD', 'EOS-USD', 'XMR-USD',
                            'DASH-USD', 'ZEC-USD', 'NEO-USD', 'QTUM-USD', 'ZIL-USD',
                            'VET-USD', 'XTZ-USD', 'ATOM-USD', 'ALGO-USD', 'KSM-USD',
                            'WAVES-USD', 'ICX-USD', 'LSK-USD', 'DCR-USD', 'REP-USD',
                            'ANT-USD', 'LOOM-USD', 'BAT-USD', 'KNC-USD', 'GNT-USD',
                            'ZRX-USD', 'CHZ-USD', 'ENJ-USD', 'MANA-USD', 'SAND-USD',
                            'MATIC-USD', 'SOL-USD', 'AVAX-USD', 'NEAR-USD', 'FIL-USD',
                            'AAVE-USD', 'UNI-USD', 'MKR-USD', 'SNX-USD', 'COMP-USD'
                        ],
                        help="List of crypto tickers")
    parser.add_argument("--factor", "-f", type=int, default=5,
                        help="Number of PCA factors")
    parser.add_argument("--objective", "-o", default='sharpe',
                        choices=['sharpe', 'sortino'],
                        help="Objective function")
    parser.add_argument("--trans-cost", type=float, default=0.0005,
                        help="Transaction cost (bps)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")

    args = parser.parse_args()

    returns, weights, metrics = run_crypto_arbitrage(
        tickers=args.tickers,
        factor=args.factor,
        objective=args.objective,
        trans_cost=args.trans_cost,
        num_epochs=args.epochs
    )

    print(f"\n Sharpe: {metrics['sharpe']:.2f}")
    print(f" Sortino: {metrics['sortino']:.2f}")


if __name__ == "__main__":
    main()