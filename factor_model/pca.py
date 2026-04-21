import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression


class CryptoPCA:

    def __init__(self, logdir=os.getcwd()):
        self._logdir = logdir
        self.crypto_returns = None
        self.crypto_volumes = None
        self.daily_dates = None
        self.tickers = None

    def load_crypto_data(self, tickers, start_date='2018-01-01', end_date='2024-12-31'):
        print(f" Chargement des données crypto pour {len(tickers)} actifs")

        data = yf.download(tickers, start=start_date, end=end_date,
                           group_by='ticker', auto_adjust=True, progress=False)

        prices = pd.DataFrame()
        volumes = pd.DataFrame()
        failed = []

        for ticker in tickers:
            if ticker in data:
                prices[ticker] = data[ticker]['Close']
                volumes[ticker] = data[ticker]['Volume']
            else:
                failed.append(ticker)
                print(f" {ticker} non trouvé")

        # Supprimer les colonnes avec trop de nan
        prices = prices.dropna(axis=1, thresh=len(prices) * 0.7)
        volumes = volumes.dropna(axis=1, thresh=len(volumes) * 0.7)

        returns = prices.pct_change().dropna()

        common_dates = returns.index.intersection(volumes.index)
        returns = returns.loc[common_dates]
        volumes = volumes.loc[common_dates]

        self.crypto_returns = returns.values
        self.crypto_volumes = volumes.values
        self.daily_dates = returns.index
        self.tickers = returns.columns.tolist()

        print(f" Données chargées: {len(returns)} jours, {len(self.tickers)} cryptos")
        print(f"   Période: {self.daily_dates[0]} à {self.daily_dates[-1]}")
        if failed:
            print(f" Cryptos ignorées: {failed}")

        return self.crypto_returns, self.daily_dates

    def OOSRollingWindowCrypto(self, save=True, printOnConsole=True,
                               initialOOSYear=2019,
                               sizeWindow=60,
                               sizeCovarianceWindow=252,
                               factorList=[5]):

        if self.crypto_returns is None:
            raise ValueError("Chargez d'abord les données avec load_crypto_data()")

        Rdaily = self.crypto_returns.copy()
        T, N = Rdaily.shape
        firstOOSDailyIdx = np.argmax(self.daily_dates.year >= initialOOSYear)

        assetsToConsider = np.ones(N, dtype=bool)
        Ntilde = np.sum(assetsToConsider)
        print(f'N actifs totaux: {N}, N actifs retenus: {Ntilde} (pas de filtrage)')

        residuals_dict = {}

        for factor in factorList:
            residualsOOS = np.zeros((T - firstOOSDailyIdx, N), dtype=float)
            valid_count = 0

            for t in range(T - firstOOSDailyIdx):
                window_start = max(0, t + firstOOSDailyIdx - sizeCovarianceWindow + 1)
                window_end = t + firstOOSDailyIdx + 1

                idxsNotMissing = ~np.any(np.isnan(Rdaily[window_start:window_end, :]), axis=0)
                idxsSelected = idxsNotMissing & assetsToConsider

                if np.sum(idxsSelected) < max(factor + 5, 10):
                    continue

                valid_count += 1

                if factor == 0:
                    residualsOOS[t:t + 1, idxsSelected] = Rdaily[t + firstOOSDailyIdx:t + firstOOSDailyIdx + 1, idxsSelected]
                else:
                    res_cov_window = Rdaily[window_start:window_end, idxsSelected]
                    res_cov_window = np.nan_to_num(res_cov_window)

                    res_mean = np.mean(res_cov_window, axis=0, keepdims=True)
                    res_vol = np.std(res_cov_window, axis=0, keepdims=True)
                    res_vol[res_vol == 0] = 1
                    res_normalized = (res_cov_window - res_mean) / res_vol

                    Corr = np.dot(res_normalized.T, res_normalized) / res_normalized.shape[0]
                    eigenValues, eigenVectors = np.linalg.eigh(Corr)
                    loadings = eigenVectors[:, -factor:].real

                    factors = np.dot(res_cov_window[-sizeWindow:, :] / res_vol, loadings)

                    regr = LinearRegression(fit_intercept=False, n_jobs=-1)
                    regr.fit(factors, res_cov_window[-sizeWindow:, :])
                    estimated_loadings = regr.coef_

                    current_factors = np.dot(Rdaily[t + firstOOSDailyIdx:t + firstOOSDailyIdx + 1, idxsSelected] / res_vol, loadings)
                    residuals = Rdaily[t + firstOOSDailyIdx:t + firstOOSDailyIdx + 1, idxsSelected] - np.dot(current_factors, estimated_loadings.T)
                    residualsOOS[t:t + 1, idxsSelected] = residuals

                if t % 100 == 0 and printOnConsole and t > 0:
                    print(f"   Date {t}/{T - firstOOSDailyIdx}, Actifs sélectionnés: {np.sum(idxsSelected)}")

            residualsOOS = np.nan_to_num(residualsOOS)
            residuals_dict[factor] = residualsOOS

            print(f" Factor {factor} terminé - {valid_count} dates valides sur {T - firstOOSDailyIdx}")

            if save:
                save_path = os.path.join(self._logdir, f"Crypto_PCA_residuals_{factor}factors_{initialOOSYear}start.npy")
                np.save(save_path, residualsOOS)
                print(f"   Sauvegardé dans {save_path}")

        return residuals_dict