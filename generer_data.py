import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path

START_DATE = '2021-01-01'
END_DATE = '2026-04-23'

OUTPUT_DIR = Path('crypto_output')
OUTPUT_DIR.mkdir(exist_ok=True)

SYMBOLS_120 = """1INCH	AAVE	AGLD	ALCX	ALGO	ALICE	ALPHA	AMPL	APE	ASD	ATLAS	ATOM	AUDIO	AVAX	AXS	BADGER	BAL	BAND	BAT	BCH	BIT	BNB	BNT	BOBA	BRZ	BTC	BTT	C98	CEL	CHR	CHZ	CLV	COMP	CREAM	CRO	CRV	CVC	CVX	DAWN	DENT	DODO	DOGE	DOT	DYDX	EDEN	ENJ	ENS	ETH	ETHW	FIDA	FTM	FTT	FXS	GAL	GALA	GMT	GRT	GST	HNT	HOLY	HT	IMX	KBTT	KNC	KSHIB	KSOS	LDO	LEO	LINA	LINK	LOOKS	LRC	LTC	MANA	MAPS	MATIC	MEDIA	MKR	MNGO	MOB	MTL	NEAR	OKB	OMG	OXY	PAXG	PEOPLE	PERP	POLIS	PROM	PUNDIX	RAY	REEF	REN	RNDR	RSR	SAND	SECO	SHIB	SKL	SLP	SNX	SOL	SOS	SPELL	SRM	STEP	STG	STMX	STORJ	SUSHI	SXP	TLM	TOMO	TONCOIN	TRU	TRX	TRYB	UNI	WAVES	XAUT	XRP	YFI	YFII	ZRX""".split('\t')

EXCLUDE = {'BRZ', 'PAXG', 'XAUT', 'BIT'}
SYMBOLS_CLEAN = [s for s in SYMBOLS_120 if s not in EXCLUDE]

print(f"Nombre de cryptos: {len(SYMBOLS_120)}")
print(f"Après nettoyage: {len(SYMBOLS_CLEAN)} cryptos\n")

SYMBOL_TO_COINGECKO = {
    'BTC': 'bitcoin', 'ETH': 'ethereum', 'BNB': 'binancecoin', 'SOL': 'solana',
    'DOGE': 'dogecoin', 'XRP': 'ripple', 'DOT': 'polkadot', 'LINK': 'chainlink',
    'AVAX': 'avalanche-2', 'MATIC': 'matic-network', 'LTC': 'litecoin', 'BCH': 'bitcoin-cash',
    'ATOM': 'cosmos', 'NEAR': 'near', 'AAVE': 'aave', 'UNI': 'uniswap', 'CRV': 'curve-dao-token',
    'CVX': 'convex-finance', 'MKR': 'maker', 'SNX': 'synthetix-network-token', 'SUSHI': 'sushi',
    'COMP': 'compound-governance-token', 'GRT': 'the-graph', 'LDO': 'lido-dao', 'ENS': 'ethereum-name-service',
    'FTM': 'fantom', 'GALA': 'gala', 'AXS': 'axie-infinity', 'SAND': 'the-sandbox', 'MANA': 'decentraland',
    'IMX': 'immutable-x', 'CHZ': 'chiliz', 'BAND': 'band-protocol', '1INCH': '1inch', 'SHIB': 'shiba-inu',
    'DYDX': 'dydx', 'ENJ': 'enjin-coin', 'BAT': 'basic-attention-token', 'LINA': 'linear-finance',
    'LOOKS': 'looksrare', 'LRC': 'loopring', 'PERP': 'perpetual-protocol', 'SKL': 'skale',
    'SLP': 'smooth-love-potion', 'TRX': 'tron', 'VET': 'vechain', 'WAVES': 'waves', 'ZRX': '0x',
    'AGLD': 'adventure-gold', 'ALCX': 'alchemix', 'ALICE': 'my-neighbor-alice', 'ALGO': 'algorand',
    'ALPHA': 'alpha-finance-lab', 'AMPL': 'ampleforth', 'APE': 'apecoin', 'ATLAS': 'star-atlas',
    'AUDIO': 'audio', 'BADGER': 'badger-dao', 'BAL': 'balancer', 'BOBA': 'boba-network',
    'BTT': 'bittorrent', 'BNT': 'bancor', 'C98': 'coin98', 'CEL': 'celsius-degree-token',
    'CHR': 'chromaway', 'CLV': 'clover-finance', 'CREAM': 'cream-2', 'CRO': 'crypto-com-chain',
    'CVC': 'civic', 'DENT': 'dent', 'DODO': 'dodo', 'EDEN': 'eden', 'ETHW': 'ethereum-pow-iou',
    'FIDA': 'bonfida', 'FTT': 'ftx-token', 'FXS': 'frax-share', 'GAL': 'project-galaxy',
    'GMT': 'stepn', 'GST': 'green-satoshi-token', 'HNT': 'helium', 'HOLY': 'holy-trinity',
    'HT': 'huobi-token', 'KBTT': 'kbtt', 'KNC': 'kyber-network-crystal', 'KSHIB': 'kshib',
    'KSOS': 'ksos', 'LEO': 'leo-token', 'MAPS': 'star-atlas-maps', 'MEDIA': 'media-network',
    'MNGO': 'mango-markets', 'MOB': 'mobilecoin', 'MTL': 'metal', 'OKB': 'okb', 'OMG': 'omisego',
    'OXY': 'oxygen', 'PAXG': 'pax-gold', 'PEOPLE': 'constitutiondao', 'POLIS': 'polis',
    'PROM': 'prometheus', 'PUNDIX': 'pundi-x', 'RAY': 'raydium', 'REEF': 'reef-finance',
    'REN': 'republic-protocol', 'RNDR': 'render-token', 'RSR': 'reserve-rights-token',
    'SECO': 'seco', 'SOS': 'sos', 'SPELL': 'spell-token', 'SRM': 'serum', 'STEP': 'step-app',
    'STG': 'stargate-finance', 'STMX': 'storm-x', 'STORJ': 'storj', 'SXP': 'swipe',
    'TLM': 'alien-worlds', 'TOMO': 'tomochain', 'TONCOIN': 'ton', 'TRU': 'truefi', 'TRYB': 'tryb',
    'XAUT': 'tether-gold', 'YFI': 'yearn-finance', 'YFII': 'yfii-finance',
}

def fetch_prices_binance_4h(symbol, retry_count=3):
    for attempt in range(retry_count):
        try:
            binance_pair = f"{symbol}USDT"
            all_klines = []
            
            start_ts = int(pd.Timestamp(START_DATE, tz='UTC').timestamp() * 1000)
            end_ts = int(pd.Timestamp(END_DATE, tz='UTC').timestamp() * 1000)
            current_ts = start_ts
            
            request_count = 0
            print(f"    {binance_pair}...", end='', flush=True)
            
            while current_ts < end_ts:
                time.sleep(0.1)
                request_count += 1
                
                url = "https://api.binance.com/api/v3/klines"
                params = {
                    'symbol': binance_pair,
                    'interval': '4h',
                    'startTime': current_ts,
                    'endTime': end_ts,
                    'limit': 1000
                }
                
                resp = requests.get(url, params=params, timeout=15)
                
                if resp.status_code == 429:
                    print(f"\n    ⏳ Rate limited, waiting...")
                    time.sleep(5)
                    continue
                elif resp.status_code == 400:
                    print(f"\n  ❌ {symbol}: Not found on Binance")
                    return None
                
                resp.raise_for_status()
                klines = resp.json()
                
                if not klines or len(klines) == 0:
                    break
                
                all_klines.extend(klines)
                last_time = klines[-1][0]
                current_ts = last_time + 1
                
                print(f".", end='', flush=True)
            
            print()
            
            if not all_klines:
                print(f"  ❌ {symbol}: No data")
                return None
            
            df_data = []
            for kline in all_klines:
                timestamp = pd.Timestamp(kline[0], unit='ms', tz='UTC')
                close_price = float(kline[4])
                df_data.append({'date': timestamp, 'price': close_price})
            
            df = pd.DataFrame(df_data)
            df = df.set_index('date')
            
            if len(df) == 0:
                print(f"  ❌ {symbol}: No data in range")
                return None
            
            print(f"  ✓ {symbol}: {len(df)} candles [{request_count} requests]")
            return df
            
        except Exception as e:
            if attempt < retry_count - 1:
                print(f"\n  ⏳ {symbol}: Retry...")
                continue
            else:
                print(f"\n  ❌ {symbol}: {str(e)[:100]}")
                return None
    
    return None

def load_all_prices(symbols):
    print("\n" + "="*80)
    print(f"TÉLÉCHARGEMENT ({START_DATE} → {END_DATE})")
    print("="*80 + "\n")
    
    all_prices = {}
    
    for i, symbol in enumerate(symbols):
        print(f"[{i+1}/{len(symbols)}] {symbol}...", end=' ', flush=True)
        df = fetch_prices_binance_4h(symbol)
        if df is not None:
            all_prices[symbol] = df['price']
    
    if not all_prices:
        print("\n❌ Aucun crypto chargé!")
        return None
    
    prices_df = pd.concat(all_prices, axis=1)
    prices_df = prices_df.sort_index()
    
    print(f"\n✓ Chargé {len(prices_df.columns)} cryptos")
    print(f"  Bougies: {len(prices_df)}")
    
    return prices_df

def fetch_circulating_supply(symbol, retry_count=3):
    coingecko_id = SYMBOL_TO_COINGECKO.get(symbol, symbol.lower())
    
    for attempt in range(retry_count):
        try:
            wait_time = 6.0 + (attempt * 5)
            time.sleep(wait_time)
            
            url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}"
            resp = requests.get(url, timeout=15)
            
            if resp.status_code == 429:
                if attempt < retry_count - 1:
                    print(f"  ⏳ {symbol}: Rate limited, waiting 30s...")
                    time.sleep(30)
                    continue
                else:
                    print(f"  ⚠️  {symbol}: Rate limited (skip)")
                    return np.nan
            elif resp.status_code == 404:
                print(f"  ❌ {symbol}: Not found on CoinGecko")
                return np.nan
            
            resp.raise_for_status()
            data = resp.json()
            
            supply = data.get('market_data', {}).get('circulating_supply')
            
            if supply is None or supply == 0:
                print(f"  ⚠️  {symbol}: Supply is None/0")
                return np.nan
            
            print(f"  ✓ {symbol}: {supply:.2e}")
            return supply
            
        except Exception as e:
            if attempt < retry_count - 1:
                print(f"  ⏳ {symbol}: Error {str(e)[:50]}, retry...")
                continue
            else:
                print(f"  ❌ {symbol}: {str(e)[:100]}")
                return np.nan
    
    return np.nan

def load_all_supplies(symbols):
    print("\n" + "="*80)
    print("RÉCUPÉRATION SUPPLIES (CoinGecko)")
    print("="*80 + "\n")
    
    supplies = {}
    successful = 0
    
    for symbol in symbols:
        supply = fetch_circulating_supply(symbol)
        supplies[symbol] = supply
        if not np.isnan(supply):
            successful += 1
    
    print(f"\n✓ {successful} / {len(symbols)} supplies\n")
    
    return supplies

def calculate_marketcap_and_filter_top40(prices_df, supplies_dict, n=40):
    print("\n" + "="*80)
    print(f"FILTRAGE TOP {n}")
    print("="*80 + "\n")
    
    top_n_history = []
    
    for idx, row in prices_df.iterrows():
        marketcaps = {}
        
        for symbol in prices_df.columns:
            price = row[symbol]
            supply = supplies_dict.get(symbol, np.nan)
            
            if pd.isna(price) or np.isnan(supply):
                continue
            
            mcap = price * supply
            marketcaps[symbol] = mcap
        
        if len(marketcaps) >= n:
            top_symbols = sorted(marketcaps.items(), key=lambda x: x[1], reverse=True)[:n]
            top_symbols = [s[0] for s in top_symbols]
        else:
            top_symbols = sorted(marketcaps.keys(), key=lambda s: marketcaps[s], reverse=True)
        
        top_n_history.append({
            'date': idx,
            'symbols': top_symbols,
            'count': len(top_symbols)
        })
    
    data_dict = {symbol: [] for symbol in prices_df.columns}
    
    for entry in top_n_history:
        for symbol in prices_df.columns:
            if symbol in entry['symbols']:
                data_dict[symbol].append(1)
            else:
                data_dict[symbol].append(np.nan)
    
    top_n_df = pd.DataFrame(data_dict, index=prices_df.index)
    
    print(f"✓ Filtrage done")
    
    return top_n_df, top_n_history

def main():
    print("\n" + "="*80)
    print("CRYPTO PIPELINE - 4H BINANCE + TOP 40")
    print("="*80)
    
    prices_df = load_all_prices(SYMBOLS_CLEAN)
    
    if prices_df is None or len(prices_df) == 0:
        print("\n❌ Error!")
        return
    
    supplies_dict = load_all_supplies(SYMBOLS_CLEAN)
    
    top40_df, top40_history = calculate_marketcap_and_filter_top40(
        prices_df, 
        supplies_dict, 
        n=40
    )
    
    output_file_1 = OUTPUT_DIR / 'crypto_prices_4h_2022_2026.csv'
    prices_df.to_csv(output_file_1)
    
    print(f"\n✓ Fichier 1: {output_file_1}")
    print(f"  Shape: {prices_df.shape}")
    
    top40_export = []
    for entry in top40_history:
        row = {'date': entry['date']}
        for i, symbol in enumerate(entry['symbols']):
            row[f'rank_{i+1}'] = symbol
        top40_export.append(row)
    
    top40_export_df = pd.DataFrame(top40_export)
    
    output_file_2 = OUTPUT_DIR / 'crypto_top40_by_marketcap_2022_2026.csv'
    top40_export_df.to_csv(output_file_2, index=False)
    
    print(f"\n✓ Fichier 2: {output_file_2}")
    print(f"  Shape: {top40_export_df.shape}")
    
    print("\n✓ Done!")

if __name__ == '__main__':
    main()