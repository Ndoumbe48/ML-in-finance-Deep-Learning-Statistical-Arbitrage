import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class PCA_Crypto:
    """
    Classe pour effectuer une analyse en composantes principales sur des prix de cryptomonnaies.
    
    À la différence de la version stocks:
    - Au lieu de filtrer par capitalisation minimale, on sélectionne les top 40 cryptos par market cap
    - Les top 40 changent à chaque timestamp (4h)
    - Pas d'aggégation temporelle complexe (les prix sont déjà en 4h)
    """
    
    def __init__(self, prices_csv, top40_csv, logdir=os.getcwd()):
        """
        Initialise la classe avec les données de prix et de top 40.
        
        Args:
            prices_csv (str): Chemin vers crypto_prices_4h_2022_2026.csv
            top40_csv (str): Chemin vers crypto_top40_by_marketcap_2022_2026.csv
            logdir (str): Répertoire de sortie pour les résultats
        """
        
        # Charge les données de prix (4h)
        self.prices_df = pd.read_csv(prices_csv, parse_dates=['date'])
        # Dimensions: (T_timestamps, N_cryptos+1) où la première colonne est la date
        # Exemple: (9440, 96) avec date + 95 cryptos
        
        # Extrait les dates et les prix
        self.dates = self.prices_df['date'].values
        # Dimensions: (T_timestamps,) en numpy datetime64
        
        self.daily_prices = self.prices_df.drop('date', axis=1).values.astype(float)
        # Dimensions: (T_timestamps, N_cryptos)
        # Les prix en float64
        
        self.crypto_names = self.prices_df.drop('date', axis=1).columns.tolist()
        # Liste des noms de cryptos (pour mapper les indices aux symboles)
        # Exemple: ['1INCH', 'AAVE', 'AGLD', ...]
        
        # Charge les données de top 40 par market cap
        self.top40_df = pd.read_csv(top40_csv, parse_dates=['date'])
        # Dimensions: (T_timestamps, 41) où la première colonne est la date, puis rank_1 à rank_40
        
        # Aligne les deux DataFrames sur la date
        # Important: les deux fichiers doivent avoir exactement les mêmes timestamps
        assert len(self.prices_df) == len(self.top40_df), \
            f"Mismatch: {len(self.prices_df)} prix vs {len(self.top40_df)} top40"
        
        assert (self.prices_df['date'] == self.top40_df['date']).all(), \
            "Les dates des deux fichiers ne correspondent pas"
        
        # Crée une matrice booléenne indiquant si un crypto est dans le top 40 à chaque timestamp
        self.top40_mask = self._create_top40_mask()
        # Dimensions: (T_timestamps, N_cryptos) avec True/False
        
        self._logdir = logdir
        
    
    def _create_top40_mask(self):
        """
        Crée une matrice booléenne: True si le crypto est dans le top 40 à ce timestamp.
        
        Returns:
            np.ndarray: Shape (T_timestamps, N_cryptos) avec True/False
        """
        T, N = self.daily_prices.shape
        top40_mask = np.zeros((T, N), dtype=bool)
        # Initialise une matrice pleine de False
        
        # Colones du top40_df: ['date', 'rank_1', 'rank_2', ..., 'rank_40']
        rank_columns = [col for col in self.top40_df.columns if col.startswith('rank_')]
        # Récupère les 40 colonnes de ranks
        
        for t in range(T):
            # Pour chaque timestamp
            top40_cryptos_at_t = self.top40_df.loc[t, rank_columns].tolist()
            # Liste des 40 cryptos au top à ce timestamp (par ordre de rank)
            
            for crypto in top40_cryptos_at_t:
                # Pour chaque crypto dans le top 40
                if crypto in self.crypto_names:
                    # Vérifie que le crypto existe dans la liste des prix
                    idx = self.crypto_names.index(crypto)
                    # Récupère l'index du crypto dans la matrice de prix
                    
                    top40_mask[t, idx] = True
                    # Met le masque à True pour ce crypto à ce timestamp
        
        return top40_mask
    
    
    def _get_returns(self, prices):
        """
        Calcule les rendements à partir des prix.
        
        Rendements = log(prix_t / prix_t-1)
        
        Args:
            prices (np.ndarray): Shape (T, N) avec les prix
            
        Returns:
            np.ndarray: Shape (T-1, N) avec les rendements
        """
        # Évite les divisions par zéro et les log de NaN
        returns = np.diff(np.log(prices), axis=0)
        # diff(..., axis=0) calcule les différences consécutives
        # log() s'applique avant, donc c'est log(p_t) - log(p_t-1) = log(p_t / p_t-1)
        
        return returns
    
    
    def OOSRollingWindowCryptos(self, save=True, printOnConsole=True, initialOOSYear=2023,
                                sizeWindow=60, sizeCovarianceWindow=252, factorList=range(0, 16)):
        """
        Calcule les résidus PCA out-of-sample avec une fenêtre roulante pour les cryptos.
        
        Args:
            save (bool): Sauvegarde les résultats en fichiers .npy
            printOnConsole (bool): Affiche les messages de progression
            initialOOSYear (int): Année à partir de laquelle OOS commence (2023)
            sizeWindow (int): Nombre de timestamps (4h) pour estimer les facteurs (60 ≈ 10 jours)
            sizeCovarianceWindow (int): Nombre de timestamps pour la fenêtre de covariance (252 ≈ 42 jours)
            factorList (range): Nombres de composantes PCA à tester
        """
        
        # Convertir les prix en rendements
        if printOnConsole:
            print("Converting prices to returns...")
        
        returns = self._get_returns(self.daily_prices)
        # Dimensions: (T-1, N_cryptos)
        # Note: on perd une ligne (la première) car on calcule des rendements
        
        # Ajuste les dates (perte du premier timestamp)
        dates = self.dates[1:]
        # Dimensions: (T-1,)
        
        # Ajuste le masque top40 (perte du premier timestamp)
        top40_mask = self.top40_mask[1:, :]
        # Dimensions: (T-1, N_cryptos)
        
        T, N = returns.shape
        # T = nombre de rendements (= nombre de timestamps - 1)
        # N = nombre de cryptos
        
        # Trouve le premier index OOS
        firstOOSIdx = np.argmax(pd.DatetimeIndex(dates).year >= initialOOSYear)
        # Recherche le premier timestamp >= 2023
        
        if printOnConsole:
            print(f"Total cryptos: {N}, OOS start index: {firstOOSIdx} ({dates[firstOOSIdx]})")
            print(f"OOS period: {dates[firstOOSIdx]} to {dates[-1]}")
        
        # Crée un masque pour les cryptos qui ont suffisamment de données OOS
        # et qui sont dans le top40 au moins une fois OOS
        assetsToConsider = (np.count_nonzero(~np.isnan(returns[firstOOSIdx:, :]), axis=0) >= 10) \
                         & (np.sum(top40_mask[firstOOSIdx:, :], axis=0) >= 1)
        # Conditions:
        # - Au moins 10 timestamps avec données non-NaN OOS
        # - Apparaît au moins une fois dans le top 40 OOS
        
        Ntilde = np.sum(assetsToConsider)
        # Nombre de cryptos conservés
        
        print(f"Cryptos to consider: {Ntilde}/{N}")
        
        if printOnConsole:
            print("Computing residuals...")
        
        # Boucle sur chaque nombre de facteurs PCA
        for factor in factorList:
            if printOnConsole:
                print(f"\n--- Processing factor={factor} ---")
            
            # Initialise les arrays de résultats
            residualsOOS = np.zeros((T - firstOOSIdx, N), dtype=float)
            # Dimensions: (T_OOS, N_cryptos)
            
            residualsMatricesOOS = np.zeros((T - firstOOSIdx, Ntilde, Ntilde), dtype=np.float32)
            # Dimensions: (T_OOS, Ntilde, Ntilde) pour les matrices de projection
            
            # Boucle sur chaque timestamp OOS
            for t in range(T - firstOOSIdx):
                # t=0 est le premier timestamp OOS
                
                abs_t = t + firstOOSIdx
                # Index absolu dans le array original
                
                # Sélectionne les cryptos dans le top 40 ET avec données non-NaN dans la fenêtre
                # 1) Vérifie qu'on a des données non-NaN dans la fenêtre de covariance
                window_start = max(0, abs_t - sizeCovarianceWindow)
                window_end = abs_t + 1
                
                idxsNotMissing = ~np.any(np.isnan(returns[window_start:window_end, :]), axis=0)
                # True si le crypto a des données complètes sur toute la fenêtre
                
                # 2) Vérifie que le crypto est dans le top 40 au timestamp courant
                idxsInTop40 = top40_mask[abs_t, :]
                # True si le crypto est dans le top 40 à ce timestamp
                
                # 3) Combine les deux conditions
                idxsSelected = idxsNotMissing & idxsInTop40
                # True si dans top 40 ET pas de NaN
                
                if t % 60 == 0 and printOnConsole:
                    print(f"  t={t}/{T - firstOOSIdx}, "
                          f"Top40 cryptos: {np.sum(idxsInTop40)}, "
                          f"Selected (top40 + no NaN): {np.sum(idxsSelected)}")
                
                if np.sum(idxsSelected) < 3:
                    # Pas assez de cryptos sélectionnés pour faire une PCA
                    if printOnConsole and t % 60 == 0:
                        print(f"    WARNING: Only {np.sum(idxsSelected)} cryptos selected, skipping")
                    continue
                
                # Extrait la fenêtre de covariance
                res_cov_window = returns[window_start:window_end, idxsSelected]
                # Dimensions: (sizeCovarianceWindow ou moins, n_selected_cryptos)
                
                # Cas spécial: factor = 0
                if factor == 0:
                    # Pas de réduction dimensionnelle, juste les rendements bruts
                    residualsOOS[t, idxsSelected] = returns[abs_t, idxsSelected]
                
                else:
                    # Cas général: factor >= 1
                    
                    # 1) Normalise les rendements de la fenêtre de covariance
                    res_mean = np.mean(res_cov_window, axis=0, keepdims=True)
                    # Dimension: (1, n_selected_cryptos)
                    
                    res_vol = np.sqrt(np.mean((res_cov_window - res_mean)**2, axis=0, keepdims=True))
                    # Volatilité (écart-type) de chaque crypto
                    
                    res_normalized = (res_cov_window - res_mean) / (res_vol + 1e-8)
                    # Normalisation: (X - E[X]) / std(X)
                    # +1e-8 pour éviter les divisions par zéro
                    
                    # 2) Calcule la matrice de corrélation
                    Corr = np.dot(res_normalized.T, res_normalized)
                    # Dimensions: (n_selected_cryptos, n_selected_cryptos)
                    
                    # 3) Diagonalise
                    try:
                        eigenValues, eigenVectors = np.linalg.eigh(Corr)
                        # eigh pour matrices symétriques (plus rapide et stable)
                        # Les valeurs propres sont en ordre croissant
                    except np.linalg.LinAlgError:
                        if printOnConsole:
                            print(f"    ERROR: Eigendecomposition failed at t={t}, skipping")
                        continue
                    
                    # 4) Sélectionne les facteurs avec les plus grandes valeurs propres
                    if factor > len(eigenValues):
                        factor_to_use = len(eigenValues)
                    else:
                        factor_to_use = factor
                    
                    loadings = eigenVectors[:, -factor_to_use:].real
                    # Prend les factor derniers vecteurs propres (= les plus grandes valeurs)
                    # Dimensions: (n_selected_cryptos, factor)
                    
                    # 5) Projette les rendements sur les loadings
                    # Utilise la fenêtre de covariance complète pour estimer les facteurs
                    factors = np.dot(res_cov_window[-sizeWindow:, :] / (res_vol + 1e-8), loadings)
                    # Dimensions: (min(sizeWindow, len(res_cov_window)), factor)
                    
                    old_loadings = loadings
                    # Sauvegarde les loadings originaux
                    
                    # 6) Ajuste les loadings par régression linéaire
                    if len(factors) > 0:
                        regr = LinearRegression(fit_intercept=False, n_jobs=-1).fit(
                            factors, res_cov_window[-len(factors):, :]
                        )
                        # Régression: rendements = facteurs @ coefficients
                        
                        loadings = regr.coef_
                        # Nouveaux loadings (coefficients de régression)
                        
                        # 7) Calcule les résidus du timestamp courant
                        current_factors = np.dot(returns[abs_t, idxsSelected] / (res_vol.squeeze() + 1e-8), 
                                               loadings)
                        # Exposition du crypto courant aux facteurs
                        
                        residuals = returns[abs_t, idxsSelected] - current_factors.dot(loadings.T)
                        # Résidus = rendements - (exposition @ loadings)
                        
                        residualsOOS[t, idxsSelected] = residuals
                    else:
                        if printOnConsole and t % 60 == 0:
                            print(f"    WARNING: Empty factors at t={t}")
            
            # Remplace les NaN par 0
            np.nan_to_num(residualsOOS, copy=False)
            np.nan_to_num(residualsMatricesOOS, copy=False)
            
            if printOnConsole:
                logging.info(f"Finished factor: {factor}")
            
            # Sauvegarde les résultats
            if save:
                filename_residuals = os.path.join(
                    self._logdir,
                    f"CryptoPCA_OOSresiduals_factor{factor}_initialOOSYear{initialOOSYear}_"
                    f"window{sizeWindow}_cov{sizeCovarianceWindow}.npy"
                )
                np.save(filename_residuals, residualsOOS)
                
                if printOnConsole:
                    print(f"Saved: {filename_residuals}")
        
        return
    
    
    def OOSRollingWindowCryptosVectorized(self, save=True, printOnConsole=True, initialOOSYear=2023,
                                          sizeWindow=60, sizeCovarianceWindow=252, factorList=range(0, 16)):
        """
        Version vectorisée: calcule tous les facteurs en une seule boucle sur les timestamps.
        
        Plus efficace que la version non-vectorisée.
        """
        
        # Convertir les prix en rendements
        if printOnConsole:
            print("Converting prices to returns...")
        
        returns = self._get_returns(self.daily_prices)
        # Dimensions: (T-1, N_cryptos)
        
        dates = self.dates[1:]
        top40_mask = self.top40_mask[1:, :]
        
        T, N = returns.shape
        
        # Trouve le premier index OOS
        firstOOSIdx = np.argmax(pd.DatetimeIndex(dates).year >= initialOOSYear)
        
        if printOnConsole:
            print(f"Total cryptos: {N}, OOS start index: {firstOOSIdx} ({dates[firstOOSIdx]})")
        
        # Filtre les cryptos
        assetsToConsider = (np.count_nonzero(~np.isnan(returns[firstOOSIdx:, :]), axis=0) >= 10) \
                         & (np.sum(top40_mask[firstOOSIdx:, :], axis=0) >= 1)
        
        Ntilde = np.sum(assetsToConsider)
        print(f"Cryptos to consider: {Ntilde}/{N}")
        
        # Initialise un array unique pour tous les facteurs
        residualsOOS_all = np.zeros((len(factorList), T - firstOOSIdx, N), dtype=float)
        # Dimensions: (n_factors, T_OOS, N_cryptos)
        
        if printOnConsole:
            print("Computing residuals (vectorized)...")
        
        # Boucle sur les timestamps (externe)
        for t in range(T - firstOOSIdx):
            abs_t = t + firstOOSIdx
            
            # Sélectionne les cryptos du top 40 avec données complètes
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
            
            # Fenêtre de covariance
            res_cov_window = returns[window_start:window_end, idxsSelected]
            
            # Normalisation
            res_mean = np.mean(res_cov_window, axis=0, keepdims=True)
            res_vol = np.sqrt(np.mean((res_cov_window - res_mean)**2, axis=0, keepdims=True))
            res_normalized = (res_cov_window - res_mean) / (res_vol + 1e-8)
            
            # Diagonalisation
            Corr = np.dot(res_normalized.T, res_normalized)
            
            try:
                eigenValues, eigenVectors = np.linalg.eigh(Corr)
            except np.linalg.LinAlgError:
                continue
            
            # Boucle sur les facteurs (interne)
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
        
        # Remplace les NaN par 0
        np.nan_to_num(residualsOOS_all, copy=False)
        
        # Sauvegarde les résultats
        if save:
            for i, factor in enumerate(factorList):
                filename = os.path.join(
                    self._logdir,
                    f"CryptoPCA_OOSresiduals_factor{factor}_initialOOSYear{initialOOSYear}_"
                    f"window{sizeWindow}_cov{sizeCovarianceWindow}.npy"
                )
                np.save(filename, residualsOOS_all[i, :, :])
                
                if printOnConsole:
                    print(f"Saved: {filename}")
        
        return residualsOOS_all


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    # Chemins vers tes fichiers
    prices_file = "./crypto_output/crypto_prices_4h_2022_2026.csv"
    top40_file = "./crypto_output/crypto_top40_by_marketcap_2022_2026.csv"
    output_dir = "./crypto_pca_results"
    
    # Crée le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialise la classe
    pca = PCA_Crypto(prices_file, top40_file, logdir=output_dir)
    
    # Lance le calcul (vectorisé = plus rapide)
    pca.OOSRollingWindowCryptosVectorized(
        save=True,
        printOnConsole=True,
        initialOOSYear=2022,
        sizeWindow=360,           
        sizeCovarianceWindow=1512, 
        factorList=range(0, 16)
    )
    
    print("\nDone!")