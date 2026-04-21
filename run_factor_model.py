import argparse
import logging
import os

import numpy as np
import pandas as pd

from factor_model.pca import CryptoPCA
from utils import initialize_logging


def run_pca_crypto():

    pca = CryptoPCA(logdir=os.path.join('residuals', 'crypto_pca'))

    tickers = [
        # Top capitalisations (2013-2017)
        'BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'DOGE-USD',
        'ADA-USD', 'BNB-USD', 'DOT-USD', 'LINK-USD', 'XLM-USD',
        'BCH-USD', 'ETC-USD', 'TRX-USD', 'EOS-USD', 'XMR-USD',
        'DASH-USD', 'ZEC-USD', 'NEO-USD', 'QTUM-USD', 'ZIL-USD',

        # Moyennes capitalisations (2017-2018)
        'VET-USD', 'XTZ-USD', 'ATOM-USD', 'ALGO-USD', 'KSM-USD',
        'WAVES-USD', 'ICX-USD', 'LSK-USD', 'DCR-USD', 'REP-USD',
        'ANT-USD', 'LOOM-USD', 'BAT-USD', 'KNC-USD', 'GNT-USD',
        'ZRX-USD', 'CHZ-USD', 'ENJ-USD', 'MANA-USD', 'SAND-USD',

        # DéFi et L1 (2019-2020)
        'MATIC-USD', 'SOL-USD', 'AVAX-USD', 'NEAR-USD', 'FIL-USD',
        'AAVE-USD', 'UNI-USD', 'MKR-USD', 'SNX-USD', 'COMP-USD'
    ]

    returns, dates = pca.load_crypto_data(
        tickers=tickers,
        start_date='2018-01-01',
        end_date='2024-12-31'
    )

    print("STATISTIQUES DES DONNÉES")
    print(f"Nombre de jours: {returns.shape[0]}")
    print(f"Nombre de cryptos: {returns.shape[1]}")
    print(f"Nombre total d'observations: {returns.shape[0] * returns.shape[1]:,}")
    print(f"Période: {dates[0]} à {dates[-1]}")
    print(f"Années: {(dates[-1] - dates[0]).days / 365:.1f}")

    residuals = pca.OOSRollingWindowCrypto(
        save=True,
        printOnConsole=True,
        initialOOSYear=2021,
        sizeWindow=60,
        sizeCovarianceWindow=252,  # Garde 30% des plus liquides
        factorList=[0, 1, 3, 5, 8, 10, 15]
    )

    logging.info("PCA crypto completed!")
    return residuals


def init_argparse():
    parser = argparse.ArgumentParser(
        description="Run PCA factor model on crypto data"
    )
    parser.add_argument("--model", "-m",
                        help="factor model (only 'pca' works for crypto)")
    return parser


def main():
    parser = init_argparse()
    args = parser.parse_args()

    print("PCA CRYPTO - Running")
    run_pca_crypto()

if __name__ == "__main__":
    main()