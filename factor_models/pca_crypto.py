import os
import numpy as np
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import time


class CryptoPCABinance:

    def __init__(self, logdir=os.getcwd()):
        """Initialise avec Binance Client (gratuit, pas de clé API requise)"""
        self._logdir = logdir
        self.client = Client()  # Pas besoin de clé pour public data
        self.crypto_returns = None
        self.crypto_volumes = None
        self.hourly_dates = None
        self.tickers = None

    def load_crypto_data(self, symbols, start_date='2020-06-22', end_date='2024-12-31',
                         interval=Client.KLINE_INTERVAL_4HOUR):  # ← CHANGÉ: 4HOUR

        print(f" Chargement depuis Binance")
        print(f"   Symboles: {len(symbols)}")
        print(f"   Plage: {start_date} à {end_date}")
        print(f"   Interval: 4 HOURS")

        prices = pd.DataFrame()
        volumes = pd.DataFrame()
        failed = []

        for symbol in symbols:
            try:
                print(f"   {symbol}...", end=' ', flush=True)

                klines = self.client.get_historical_klines(
                    symbol,
                    interval,
                    start_str=start_date,
                    end_str=end_date
                )

                if not klines:
                    failed.append(symbol)
                    print(f"✗ Pas de data")
                    continue

                # Convertir en DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])

                # Nettoyer les types
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                prices[symbol] = df['close']
                volumes[symbol] = df['volume']
                print(f" {len(df)} candles")

                # Rate limit: 1200 req/min
                time.sleep(0.1)

            except Exception as e:
                failed.append(symbol)
                print(f" Erreur: {str(e)[:50]}")
                continue

        if len(prices) == 0:
            raise ValueError(f" Aucune donnée chargée de Binance! Vérifier les symboles.")

        print(f"\n Nettoyage des données...")

        # Supprimer les colonnes avec trop de nan (< 70% data)
        prices = prices.dropna(axis=1, thresh=len(prices) * 0.7)
        volumes = volumes.dropna(axis=1, thresh=len(volumes) * 0.7)

        # Calculer les returns
        returns = prices.pct_change().dropna()

        # Aligner les index
        common_dates = returns.index.intersection(volumes.index)
        returns = returns.loc[common_dates]
        volumes = volumes.loc[common_dates]

        self.crypto_returns = returns.values
        self.crypto_volumes = volumes.values
        self.hourly_dates = returns.index
        self.tickers = returns.columns.tolist()

        print(f" Données chargées: {len(returns)} périodes, {len(self.tickers)} symboles")
        if len(self.tickers) > 0:
            print(f"   Période: {self.hourly_dates[0]} à {self.hourly_dates[-1]}")
        if failed:
            print(f" Symboles échoués ({len(failed)}): {failed[:10]}...")

        return self.crypto_returns, self.hourly_dates

    def OOSRollingWindowCrypto(self, save=True, printOnConsole=True,
                               initialOOSYear=2023,  # ← AJUSTÉ pour 4h
                               sizeWindow=24,  # 24 périodes de 4h = 4 jours
                               sizeCovarianceWindow=168,  # 168 périodes = 28 jours
                               factorList=[5, 10]):

        """
        PCA Factor model - Out-of-sample rolling window

        Pour intervalle 4h:
        - sizeWindow = 24 → 4 jours de données
        - sizeCovarianceWindow = 168 → 28 jours (4 semaines)
        """

        if self.crypto_returns is None:
            raise ValueError("Chargez d'abord les données avec load_crypto_data()")

        if len(self.crypto_returns) == 0:
            raise ValueError("Pas de données à analyser!")

        Rdaily = self.crypto_returns.copy()
        T, N = Rdaily.shape

        # Index du début OOS
        firstOOSDailyIdx = np.argmax(self.hourly_dates.year >= initialOOSYear)

        if firstOOSDailyIdx == 0:
            print(f"  Attention: aucune donnée après {initialOOSYear}")
            firstOOSDailyIdx = max(1, T // 2)

        assetsToConsider = np.ones(N, dtype=bool)
        Ntilde = np.sum(assetsToConsider)
        print(f'\n Période OOS: {self.hourly_dates[firstOOSDailyIdx]} onwards')
        print(f' N actifs totaux: {N}, N actifs retenus: {Ntilde} (pas de filtrage)')

        residuals_dict = {}

        for factor in factorList:
            print(f"\n Traitement factor={factor}...")
            residualsOOS = np.zeros((T - firstOOSDailyIdx, N), dtype=float)
            valid_count = 0

            for t in range(T - firstOOSDailyIdx):
                # Fenêtre pour covariance
                window_start = max(0, t + firstOOSDailyIdx - sizeCovarianceWindow + 1)
                window_end = t + firstOOSDailyIdx + 1

                # Identifier assets sans NaN dans la fenêtre
                idxsNotMissing = ~np.any(np.isnan(Rdaily[window_start:window_end, :]), axis=0)
                idxsSelected = idxsNotMissing & assetsToConsider

                # Vérifier si assez d'assets
                if np.sum(idxsSelected) < max(factor + 5, 10):
                    continue

                valid_count += 1
                current_t_idx = t + firstOOSDailyIdx

                if factor == 0:
                    # Cas trivial: returns directes sans factorisation
                    residualsOOS[t:t + 1, idxsSelected] = Rdaily[current_t_idx:current_t_idx + 1, idxsSelected]
                else:
                    # PCA sur la fenêtre de covariance
                    res_cov_window = Rdaily[window_start:window_end, idxsSelected].copy()
                    res_cov_window = np.nan_to_num(res_cov_window)

                    # Standardiser
                    res_mean = np.mean(res_cov_window, axis=0, keepdims=True)
                    res_vol = np.std(res_cov_window, axis=0, keepdims=True)
                    res_vol[res_vol == 0] = 1
                    res_normalized = (res_cov_window - res_mean) / res_vol

                    # Matrice de corrélation
                    Corr = np.dot(res_normalized.T, res_normalized) / res_normalized.shape[0]

                    # Eigendecomposition
                    eigenValues, eigenVectors = np.linalg.eigh(Corr)
                    loadings = eigenVectors[:, -factor:].real  # Top 'factor' components

                    # Estimer factors sur la dernière sizeWindow
                    factors = np.dot(res_cov_window[-sizeWindow:, :] / res_vol, loadings)

                    # Régression: factors → returns
                    regr = LinearRegression(fit_intercept=False, n_jobs=-1)
                    regr.fit(factors, res_cov_window[-sizeWindow:, :])
                    estimated_loadings = regr.coef_

                    # Appliquer au jour courant
                    current_returns = Rdaily[current_t_idx:current_t_idx + 1, idxsSelected]
                    current_factors = np.dot(current_returns / res_vol, loadings)
                    fitted_returns = np.dot(current_factors, estimated_loadings.T)
                    residuals = current_returns - fitted_returns

                    residualsOOS[t:t + 1, idxsSelected] = residuals

                # Progress
                if t % 500 == 0 and printOnConsole and t > 0:
                    print(f"   t={t}/{T - firstOOSDailyIdx}, Actifs: {np.sum(idxsSelected)}")

            # Nettoyer les NaN
            residualsOOS = np.nan_to_num(residualsOOS)
            residuals_dict[factor] = residualsOOS

            print(f" Factor {factor} terminé - {valid_count} dates valides sur {T - firstOOSDailyIdx}")

            if save:
                save_path = os.path.join(
                    self._logdir,
                    f"Crypto_PCA_residuals_{factor}factors_{initialOOSYear}start_binance_4h.npy"
                )
                np.save(save_path, residualsOOS)
                print(f"    Sauvegardé: {save_path}")

        return residuals_dict


# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

def main():
    """Charger données Binance (intervalle 4h) et lancer PCA"""

    print("=" * 60)
    print(" PCA CRYPTO - BINANCE 4H INTERVAL")
    print("=" * 60)

    # Initialiser
    pca = CryptoPCABinance(logdir=os.path.join('../residuals', 'binance_4h'))

    # Symboles Binance (format: BTCUSDT, ETHUSDT, etc.)
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT',  # ← DOGEUSDT (pas DOGUSDT)
        'XRPUSDT', 'AVAXUSDT', 'MATICUSDT', 'DOTUSDT', 'LINKUSDT',
        'UNIUSDT', 'ATOMUSDT', 'AAVEUSDT', 'FTMUSDT', 'NEARUSDT',
        'MANAUSDT', 'XLMUSDT', 'VETUSDT', 'FILUSDT', 'SANDUSDT',
        'LTCUSDT', 'ETCUSDT', 'BCHUSDT', 'EOSUSDT', 'XMRUSDT'
    ]

    print(f" Symboles à charger: {len(symbols)}")

    try:
        # Charger données (intervalle 4H)
        returns, dates = pca.load_crypto_data(
            symbols=symbols,
            start_date='2020-01-01',  # Plus tôt pour plus de données
            end_date='2024-12-31',
            interval=pca.client.KLINE_INTERVAL_4HOUR  # ← 4 HOURS
        )

        print(f"\n Statistiques des données:")
        print(f"   Shape des returns: {returns.shape}")
        print(f"   Première période: {dates[0]}")
        print(f"   Dernière période: {dates[-1]}")
        print(f"   Total observations: {returns.shape[0] * returns.shape[1]:,}")

        # Lancer PCA
        residuals = pca.OOSRollingWindowCrypto(
            save=True,
            printOnConsole=True,
            initialOOSYear=2022,  # Commence en 2022
            sizeWindow=360,  # 24 périodes de 4h = 4 jours
            sizeCovarianceWindow=1512,  # 168 périodes = 28 jours
            factorList=[5, 8, 10]  # 5, 8 et 10 facteurs
        )

        print("\n" + "=" * 60)
        print(" Analyse complète!")
        print("=" * 60)
        for factor, res in residuals.items():
            print(f"  Factor {factor}: {res.shape}")

        print(f"\n Résidus sauvegardés dans: residuals/binance_4h/")

    except Exception as e:
        print(f" Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()