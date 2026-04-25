import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class PCA_Crypto:
    """
    Class to perform principal component analysis on cryptocurrency prices.
    
    - Select the top 50 cryptocurrencies by market cap
    - The top 40 change at each timestamp (4h)
    """
    
    def __init__(self, prices_csv, top40_csv, logdir=os.getcwd()):
        """
        Initialize the class with price and top 40 data.
        
        Args:
            prices_csv (str): Path to crypto_prices_4h_2022_2026.csv
            top40_csv (str): Path to crypto_top40_by_marketcap_2022_2026.csv
            logdir (str): Output directory for results
        """
        
        # Load price data (4h)
        self.prices_df = pd.read_csv(prices_csv, parse_dates=['date'])
        # Dimensions: (T_timestamps, N_cryptos+1) where first column is date

        
        # Extract dates and prices
        self.dates = self.prices_df['date'].values
        # Dimensions: (T_timestamps,) in numpy datetime64
        
        self.daily_prices = self.prices_df.drop('date', axis=1).values.astype(float)
        # Dimensions: (T_timestamps, N_cryptos)
        # Prices as float64
        
        self.crypto_names = self.prices_df.drop('date', axis=1).columns.tolist()
        # List of cryptocurrency names (to map indices to symbols)

        
        # Load top 40 by market cap data
        self.top40_df = pd.read_csv(top40_csv, parse_dates=['date'])
        # Dimensions: (T_timestamps, 41) where first column is date, then rank_1 to rank_40
        
        # Align both DataFrames on date
        # Important: both files must have exactly the same timestamps
        assert len(self.prices_df) == len(self.top40_df), \
            f"Mismatch: {len(self.prices_df)} prices vs {len(self.top40_df)} top40"
        
        assert (self.prices_df['date'] == self.top40_df['date']).all(), \
            "Dates in both files do not match"
        
        # Create a boolean matrix indicating if a crypto is in the top 40 at each timestamp
        self.top40_mask = self._create_top40_mask()
        # Dimensions: (T_timestamps, N_cryptos) with True/False
        
        self._logdir = logdir
        
    
    def _create_top40_mask(self):
        """
        Create a boolean matrix: True if the crypto is in the top 50 at this timestamp.
        
        Returns:
            np.ndarray: Shape (T_timestamps, N_cryptos) with True/False
        """
        T, N = self.daily_prices.shape
        top40_mask = np.zeros((T, N), dtype=bool)
        # Initialize a matrix full of False
        
        # Columns of top40_df: ['date', 'rank_1', 'rank_2', ..., 'rank_40']
        rank_columns = [col for col in self.top40_df.columns if col.startswith('rank_')]
        # Get the 40 rank columns
        
        for t in range(T):
            # For each timestamp
            top40_cryptos_at_t = self.top40_df.loc[t, rank_columns].tolist()
            # List of the 40 top cryptos at this timestamp (in rank order)
            
            for crypto in top40_cryptos_at_t:
                # For each crypto in the top 40
                if crypto in self.crypto_names:
                    # Verify that the crypto exists in the price list
                    idx = self.crypto_names.index(crypto)
                    # Get the index of the crypto in the price matrix
                    
                    top40_mask[t, idx] = True
                    # Set the mask to True for this crypto at this timestamp
        
        return top40_mask
    
    
    def _get_returns(self, prices):
        """
        Calculate returns from prices.
        
        Returns = log(price_t / price_t-1)
        
        Args:
            prices (np.ndarray): Shape (T, N) with prices
            
        Returns:
            np.ndarray: Shape (T-1, N) with returns
        """
        # Avoid division by zero and log of NaN
        returns = np.diff(np.log(prices), axis=0)
        # diff(..., axis=0) computes consecutive differences
        # log() is applied before, so it's log(p_t) - log(p_t-1) = log(p_t / p_t-1)
        
        return returns
    
    
    
    def OOSRollingWindowCryptosVectorized(self, save=True, printOnConsole=True, initialOOSYear=2023,
                                          sizeWindow=60, sizeCovarianceWindow=252, factorList=range(0, 16)):
        """
        Compute all factors in a single loop over timestamps.
        
        """
        
        # Convert prices to returns
        if printOnConsole:
            print("Converting prices to returns...")
        
        returns = self._get_returns(self.daily_prices)
        # Dimensions: (T-1, N_cryptos)
        
        dates = self.dates[1:]
        top40_mask = self.top40_mask[1:, :]
        
        T, N = returns.shape
        
        # Find the first OOS index
        firstOOSIdx = np.argmax(pd.DatetimeIndex(dates).year >= initialOOSYear)
        
        if printOnConsole:
            print(f"Total cryptos: {N}, OOS start index: {firstOOSIdx} ({dates[firstOOSIdx]})")
        
        # Filter cryptocurrencies
        assetsToConsider = (np.count_nonzero(~np.isnan(returns[firstOOSIdx:, :]), axis=0) >= 10) \
                         & (np.sum(top40_mask[firstOOSIdx:, :], axis=0) >= 1)
        
        Ntilde = np.sum(assetsToConsider)
        print(f"Cryptos to consider: {Ntilde}/{N}")
        
        # Initialize a single array for all factors
        residualsOOS_all = np.zeros((len(factorList), T - firstOOSIdx, N), dtype=float)
        # Dimensions: (n_factors, T_OOS, N_cryptos)
        
        if printOnConsole:
            print("Computing residuals (vectorized)...")
        
        # Loop over timestamps (outer)
        for t in range(T - firstOOSIdx):
            abs_t = t + firstOOSIdx
            
            # Select top 40 cryptos with complete data
            window_start = max(0, abs_t - sizeCovarianceWindow)
            window_end = abs_t + 1
            
            idxsNotMissing = ~np.any(np.isnan(returns[window_start:window_end, :]), axis=0)
            idxsInTop40 = top40_mask[abs_t, :]
            idxsSelected = idxsNotMissing & idxsInTop40
            
            if t % 60 == 0 and printOnConsole:
                print(f"  t={t}/{T - firstOOSIdx}, "
                      f"Top40: {np.sum(idxsInTop40)}, "
                      f"Selected: {np.sum(idxsSelected)}")
            
            if np.sum(idxsSelected) < 3:
                continue
            
            # Covariance window
            res_cov_window = returns[window_start:window_end, idxsSelected]
            
            # Normalization
            res_mean = np.mean(res_cov_window, axis=0, keepdims=True)
            res_vol = np.sqrt(np.mean((res_cov_window - res_mean)**2, axis=0, keepdims=True))
            res_normalized = (res_cov_window - res_mean) / (res_vol + 1e-8)
            
            # Diagonalization
            Corr = np.dot(res_normalized.T, res_normalized)
            
            try:
                eigenValues, eigenVectors = np.linalg.eigh(Corr)
            except np.linalg.LinAlgError:
                continue
            
            # Loop over factors (inner)
            for i, factor in enumerate(factorList):
                if factor == 0:
                    residualsOOS_all[i, t, idxsSelected] = returns[abs_t, idxsSelected]
                else:
                    factor_to_use = min(factor, len(eigenValues))
                    loadings = eigenVectors[:, -factor_to_use:].real
                    
                    factors = np.dot(res_cov_window[-sizeWindow:, :] / (res_vol + 1e-8), loadings)
                    
                    if len(factors) > 0:
                        regr = LinearRegression(fit_intercept=False, n_jobs=-1).fit(
                            factors, res_cov_window[-len(factors):, :]
                        )
                        loadings = regr.coef_
                        
                        current_factors = np.dot(returns[abs_t, idxsSelected] / (res_vol.squeeze() + 1e-8),
                                               loadings)
                        residuals = returns[abs_t, idxsSelected] - current_factors.dot(loadings.T)
                        
                        residualsOOS_all[i, t, idxsSelected] = residuals
        
        # Replace NaN with 0
        np.nan_to_num(residualsOOS_all, copy=False)
        
        # Save results
        if save:
            for i, factor in enumerate(factorList):
                filename = os.path.join(
                    self._logdir,
                    f"Crypto_PCAresiduals_{factor}_factors_{initialOOSYear}_initialOOSYear_"
                    f"{sizeWindow}_rollingWindow_0.01_Cap.npy"
                )
                np.save(filename, residualsOOS_all[i, :, :])
                
                if printOnConsole:
                    print(f"Saved: {filename}")
        
        return residualsOOS_all


# Usage 

if __name__ == "__main__":
    # Paths 
    prices_file = "./crypto_output/crypto_prices_4h_2022_2026.csv"
    top40_file = "./crypto_output/crypto_top40_by_marketcap_2022_2026.csv"
    output_dir = "./residuals/crypto_pca_4h"
    
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the class
    pca = PCA_Crypto(prices_file, top40_file, logdir=output_dir)
    
    # Launch the computation
    pca.OOSRollingWindowCryptosVectorized(
        save=True,
        printOnConsole=True,
        initialOOSYear=2022,
        sizeWindow=360,           
        sizeCovarianceWindow=1080, 
        factorList={0,1,3,5,8,10}
    )
    
    print("\nEnd")