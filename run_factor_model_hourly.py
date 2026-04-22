import argparse
import logging
import os
import numpy as np
import pandas as pd

from factor_model.pca_crypto import CryptoPCABinance
from utils import initialize_logging


def compute_explained_variance(returns, n_factors_max=15, sizeCovarianceWindow=180):
    """
    Calcule la variance expliquée par chaque facteur PCA
    pour déterminer le nombre optimal (80% de variance)
    """
    # Prendre une fenêtre récente de données
    recent_returns = returns[-sizeCovarianceWindow:]

    # Standardiser
    recent_returns = recent_returns - recent_returns.mean(axis=0)
    recent_returns = recent_returns / (recent_returns.std(axis=0) + 1e-8)

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(n_factors_max, recent_returns.shape[1]))
    pca.fit(recent_returns)

    # Variance expliquée cumulée
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)

    # Trouver le nombre de facteurs pour 80% de variance
    n_factors_80 = np.argmax(cumsum_var >= 0.8) + 1

    return pca.explained_variance_ratio_, cumsum_var, n_factors_80


def run_pca_crypto_4h():
    pca = CryptoPCABinance(logdir=os.path.join('residuals', 'crypto_pca_4h'))

    symbols = [
        # TIER 1: Les plus grosses et anciennes (indispensables)
        'BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'LTCUSDT', 'DOGEUSDT',
        'ADAUSDT', 'BNBUSDT', 'DOTUSDT', 'LINKUSDT', 'XLMUSDT',
        'BCHUSDT', 'ETCUSDT', 'TRXUSDT', 'EOSUSDT', 'XMRUSDT',

        # TIER 2: Anciennes et liquides (2017-2018)
        'DASHUSDT', 'ZECUSDT', 'NEOUSDT', 'QTUMUSDT', 'ZILUSDT',
        'VETUSDT', 'XTZUSDT', 'ATOMUSDT', 'ALGOUSDT', 'KSMUSDT',
        'WAVESUSDT', 'BATUSDT', 'ZRXUSDT', 'ENJUSDT', 'MANAUSDT',

        # TIER 3: Remplaçants (cryptos pures avec grosse capitalisation)
        'HBARUSDT', 'THETAUSDT', 'FTMUSDT', 'ONEUSDT', 'CROUSDT',
        'HNTUSDT', 'IOTXUSDT', 'KAVAUSDT', 'RUNEUSDT', 'FLOWUSDT'
    ]

    print("\n" + "=" * 60)
    print("TEST 1: VÉRIFICATION DES SYMBOLES (4H INTERVAL)")
    print("=" * 60)
    print(f"Nombre de symboles demandés: {len(symbols)}")

    # =========================================================
    # CHARGEMENT DES DONNÉES (intervalle 4H)
    # =========================================================
    print("\n Chargement des données Binance (intervalle 4H)...")

    returns, dates = pca.load_crypto_data(
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2024-12-31',
        interval=pca.client.KLINE_INTERVAL_4HOUR
    )

    # =========================================================
    # TEST 2: Vérifier les données chargées
    # =========================================================
    print("\n" + "=" * 60)
    print("TEST 2: VÉRIFICATION DES DONNÉES CHARGÉES")
    print("=" * 60)
    print(f" Périodes chargées (4h): {returns.shape[0]}")
    print(f" Cryptos chargées: {returns.shape[1]}")
    print(f" Total observations: {returns.shape[0] * returns.shape[1]:,}")
    print(f" Période: {dates[0]} à {dates[-1]}")

    days_count = (dates[-1] - dates[0]).days
    print(f" Années: {days_count / 365:.1f}")

    loaded_symbols = pca.tickers if hasattr(pca, 'tickers') else []
    missing_symbols = set(symbols) - set(loaded_symbols)
    if missing_symbols:
        print(f"️ Symboles manquants ({len(missing_symbols)}): {list(missing_symbols)[:10]}")
    else:
        print(f" Tous les {len(symbols)} symboles ont été chargés")

    # =========================================================
    # TEST 3: Premières observations
    # =========================================================
    print("\n" + "=" * 60)
    print("TEST 3: PREMIÈRES OBSERVATIONS (rendements 4h)")
    print("=" * 60)

    df_returns = pd.DataFrame(
        returns[:5, :10],
        index=dates[:5],
        columns=loaded_symbols[:10] if loaded_symbols else [f"Crypto_{i}" for i in range(10)]
    )
    print("Premiers rendements (5 périodes × 10 cryptos):")
    print(df_returns.round(6))

    print("\nStatistiques des rendements 4h:")
    print(f"  Moyenne: {returns.mean():.6f}")
    print(f"  Écart-type: {returns.std():.6f}")
    print(f"  Min: {returns.min():.6f}")
    print(f"  Max: {returns.max():.6f}")
    print(f"  % de zéros: {(returns == 0).sum() / returns.size * 100:.2f}%")

    # =========================================================
    # TEST 4: Volumes
    # =========================================================
    if hasattr(pca, 'crypto_volumes') and pca.crypto_volumes is not None:
        print("\n" + "=" * 60)
        print("TEST 4: STATISTIQUES DES VOLUMES (liquidité)")
        print("=" * 60)
        volumes = pca.crypto_volumes
        print(f"Volume moyen par crypto: {volumes.mean(axis=0).mean():,.0f}")
        print(f"Volume médian par crypto: {np.median(volumes, axis=0).mean():,.0f}")

        avg_volumes = volumes.mean(axis=0)
        top5_idx = np.argsort(avg_volumes)[-5:][::-1]
        top5_symbols = [loaded_symbols[i] for i in top5_idx if i < len(loaded_symbols)]
        print(f"Top 5 cryptos par volume: {top5_symbols}")

    # =========================================================
    # TEST 5: NaN check
    # =========================================================
    print("\n" + "=" * 60)
    print("TEST 5: VÉRIFICATION DES NaN")
    print("=" * 60)

    nan_count = np.isnan(returns).sum()
    print(f"NaN dans les rendements: {nan_count} ({nan_count / returns.size * 100:.2f}%)")

    if nan_count / returns.size > 0.1:
        print(" Attention: Plus de 10% de NaN")
    else:
        print(" Taux de NaN acceptable")

    # =========================================================
    # TEST 6: Calcul de la variance expliquée
    # =========================================================
    print("\n" + "=" * 60)
    print("TEST 6: VARIANCE EXPLIQUÉE PAR FACTEUR PCA")
    print("=" * 60)

    var_ratios, cumsum_var, optimal_n_factors = compute_explained_variance(
        returns, n_factors_max=15, sizeCovarianceWindow=180
    )

    print("\nVariance expliquée par facteur:")
    for i, var in enumerate(var_ratios[:10], 1):
        print(f"  Facteur {i}: {var:.2%}")

    print(f"\nVariance expliquée cumulée:")
    for i, cum in enumerate(cumsum_var[:10], 1):
        print(f"  {i} facteur(s): {cum:.2%}")

    print(f"\n Nombre optimal de facteurs pour 80% de variance: {optimal_n_factors}")

    # Liste des facteurs à calculer (basée sur l'optimal)
    factor_list = [0, 1, 3, 5, optimal_n_factors, 8, 10, 15]
    factor_list = sorted(set(factor_list))  # Enlever les doublons et trier

    print(f"Facteurs qui seront calculés: {factor_list}")

    # =========================================================
    # TEST 7: Lancer le PCA
    # =========================================================
    print("\n" + "=" * 60)
    print("TEST 7: LANCEMENT DU PCA ROLLING WINDOW (4H)")
    print("=" * 60)

    residuals = pca.OOSRollingWindowCrypto(
        save=True,
        printOnConsole=True,
        initialOOSYear=2022,
        sizeWindow=360,
        sizeCovarianceWindow=1512,
        factorList=factor_list
    )

    # =========================================================
    # TEST 8: Vérification des résidus
    # =========================================================
    print("\n" + "=" * 60)
    print("TEST 8: VÉRIFICATION DES RÉSIDUS")
    print("=" * 60)

    for factor in factor_list:
        if factor in residuals:
            res = residuals[factor]
            print(f"\n Factor {factor}:")
            print(f"   Shape: {res.shape}")
            print(f"   Moyenne: {res.mean():.6f}")
            print(f"   Écart-type: {res.std():.6f}")
            print(f"   Min: {res.min():.6f}")
            print(f"   Max: {res.max():.6f}")

    # =========================================================
    # VALIDATION FINALE
    # =========================================================
    print("\n" + "=" * 60)
    print("VALIDATION FINALE")
    print("=" * 60)

    # Utiliser le facteur optimal recommandé
    if optimal_n_factors in residuals:
        res_opt = residuals[optimal_n_factors]
        mean_res = res_opt.mean()
        std_res = res_opt.std()
        print(f" Résidus factor={optimal_n_factors} (optimal 80% variance)")
        print(f"   Moyenne: {mean_res:.6f}")
        print(f"   Écart-type: {std_res:.6f}")

        if abs(mean_res) < 0.001:
            print(" Les résidus sont bien centrés autour de 0")
        else:
            print(f" Les résidus sont légèrement décentrés ({mean_res:.6f})")

    # =========================================================
    # TEST 9: Fichiers sauvegardés
    # =========================================================
    print("\n" + "=" * 60)
    print("TEST 9: FICHIERS SAUVEGARDÉS")
    print("=" * 60)

    residual_dir = os.path.join('residuals', 'crypto_pca_4h')
    if os.path.exists(residual_dir):
        saved_files = [f for f in os.listdir(residual_dir) if f.endswith('.npy')]
        print(f" {len(saved_files)} fichiers .npy trouvés dans {residual_dir}")
        for f in sorted(saved_files)[:10]:
            print(f"   - {f}")
        if len(saved_files) > 10:
            print(f"   ... et {len(saved_files) - 10} autres")
    else:
        print(f" Dossier {residual_dir} non trouvé")

    logging.info("PCA crypto 4h completed!")

    # Afficher le résumé final
    print("\n" + "=" * 60)
    print(" RÉSUMÉ FINAL")
    print("=" * 60)
    print(f" Residuals saved in residuals/crypto_pca_4h/")
    print(f" Optimal number of factors (80% variance): {optimal_n_factors}")
    print(f" Interval: 4 hours")
    print(f" Total observations: {returns.shape[0] * returns.shape[1]:,}")
    print("=" * 60)

    return residuals, optimal_n_factors


def init_argparse():
    parser = argparse.ArgumentParser(
        description="Run PCA factor model on crypto data (4h interval)"
    )
    parser.add_argument("--model", "-m",
                        help="factor model (only 'pca' works for crypto)")
    return parser


def main():
    parser = init_argparse()
    args = parser.parse_args()

    print("=" * 60)
    print(" PCA CRYPTO - 4H INTERVAL (Binance)")
    print("=" * 60)

    residuals, optimal_n_factors = run_pca_crypto_4h()

    print("\n" + "=" * 60)
    print(" PCA COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f" Residuals saved in residuals/crypto_pca_4h/")
    print(f" Recommended number of factors: {optimal_n_factors}")
    print(f" Interval: 4 hours")
    print("=" * 60)


if __name__ == "__main__":
    main()