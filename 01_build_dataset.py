"""
01_build_dataset.py - Data Preparation
=======================================
Downloads Fama-French 10 Industry Portfolios (value-weighted monthly returns)
and macro variables (risk-free rate, CPI inflation, VIX) from Kenneth French's
Data Library and FRED, then merges them into a single clean dataset.

Inputs:  Online data via pandas-datareader and fredapi.
Outputs: data/data_clean.csv
"""

import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
from fredapi import Fred
import warnings
import os

warnings.filterwarnings('ignore')

# ===== CONFIGURATION =====
from config import FRED_API_KEY, INDUSTRIES, START_DATE, END_DATE

start_date = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
end_date = datetime.datetime.strptime(END_DATE, '%Y-%m-%d')

print("=" * 70)
print("BUILDING COMPLETE THESIS DATASET")
print("DeMiguel et al. (2009) Replication")
print(f"Requesting data: {START_DATE} to {END_DATE}")
print("=" * 70)

# ===== STEP 1: Download Fama-French 10 Industry Portfolios =====
print("\n[1/4] Downloading Fama-French 10 Industry Portfolios...")
try:
    ff_industries = web.DataReader('10_Industry_Portfolios', 'famafrench', start_date, end_date)[0]
    ff_industries = ff_industries / 100.0  # Convert % to decimal
    ff_industries.index = ff_industries.index.to_timestamp()

    print(f"   ✓ {len(ff_industries)} months × {len(ff_industries.columns)} industries")
    print(f"   Date range: {ff_industries.index[0].strftime('%Y-%m')} to {ff_industries.index[-1].strftime('%Y-%m')}")
    print(f"   >>> LAST AVAILABLE MONTH: {ff_industries.index[-1].strftime('%Y-%m')} <<<")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit()

# ===== STEP 2: Download Risk-Free Rate from Ken French =====
print("\n[2/4] Downloading Risk-Free Rate (1-Month T-Bill) from Ken French...")
try:
    ff_factors = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start_date, end_date)[0]

    # Ken French's RF is ALREADY a MONTHLY percentage.
    # Convert to decimal by dividing by 100 ONLY. Do NOT divide by 12.
    rf_data = ff_factors[['RF']].copy()
    rf_data['RF_Monthly'] = rf_data['RF'] / 100.0

    rf_data.index = rf_data.index.to_timestamp()

    print(f"   ✓ {len(rf_data)} months of risk-free rate")
    print(f"   Date range: {rf_data.index[0].strftime('%Y-%m')} to {rf_data.index[-1].strftime('%Y-%m')}")
    print(f"   Mean RF (annual): {rf_data['RF_Monthly'].mean() * 12 * 100:.2f}%")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit()

# ===== STEP 3: Download CPI from FRED =====
print("\n[3/4] Downloading CPI from FRED...")
try:
    fred = Fred(api_key=FRED_API_KEY)
    cpi_series = fred.get_series('CPIAUCSL',
                                 observation_start=START_DATE,
                                 observation_end=END_DATE)

    cpi_data = pd.DataFrame(cpi_series, columns=['CPI'])
    cpi_data['Inflation'] = cpi_data['CPI'].pct_change()
    cpi_data.index = pd.to_datetime(cpi_data.index).to_period('M').to_timestamp()

    print(f"   ✓ {len(cpi_data)} months of CPI data")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit()

# ===== STEP 4: Merge All Data =====
print("\n[4/4] Merging datasets...")

merged = ff_industries.copy()
merged = pd.merge(merged, rf_data[['RF_Monthly']], left_index=True, right_index=True, how='inner')
merged = pd.merge(merged, cpi_data[['Inflation']], left_index=True, right_index=True, how='inner')

# Calculate Real Interest Rate (for regime classification)
merged['Real_Rate'] = merged['RF_Monthly'] - merged['Inflation']

# Download VIX from FRED (for regression controls)
print("\n--- VIX ---")
try:
    vix_raw = fred.get_series('VIXCLS', observation_start='1990-01-01')
    vix_monthly = vix_raw.resample('MS').last().rename('VIX')
    vix_monthly.index = vix_monthly.index.to_period('M').to_timestamp()
    merged = pd.merge(merged, vix_monthly, left_index=True, right_index=True, how='left')
    print(f"VIX: {vix_monthly.dropna().index[0].strftime('%Y-%m')} to {vix_monthly.dropna().index[-1].strftime('%Y-%m')}")
except Exception as e:
    print(f"Warning: VIX download failed ({e}) — column will be NaN")
    merged['VIX'] = np.nan

# Keep Mkt-RF locally so downstream scripts don't need network calls
mkt_rf = (ff_factors['Mkt-RF'] / 100.0).copy()
mkt_rf.index = mkt_rf.index.to_timestamp() if hasattr(mkt_rf.index, 'to_timestamp') else mkt_rf.index
merged = pd.merge(merged, mkt_rf.rename('Mkt_RF'), left_index=True, right_index=True, how='left')

# CRITICAL FIX: Keep NOMINAL returns in dataset
# Do NOT calculate excess returns here (avoid double subtraction)
final_data = merged[INDUSTRIES + ['Real_Rate', 'RF_Monthly', 'Inflation', 'VIX', 'Mkt_RF']].copy()

# Verify and Save
if final_data.index.duplicated().any():
    final_data = final_data[~final_data.index.duplicated(keep='first')]

missing_core = final_data[INDUSTRIES + ['Real_Rate', 'RF_Monthly', 'Inflation']].isnull().sum().sum()
if missing_core > 0:
    final_data = final_data.dropna(subset=INDUSTRIES + ['Real_Rate', 'RF_Monthly', 'Inflation'])

output_path = './data/data_clean.csv'
final_data.to_csv(output_path)

print(f"\n✓ Dataset saved: {output_path}")
print(f"   Shape: {final_data.shape[0]} months × {final_data.shape[1]} columns")

# Statistics
print("\n" + "=" * 70)
print("DATASET STATISTICS")
print("=" * 70)

first_date = final_data.index[0].strftime('%Y-%m')
last_date = final_data.index[-1].strftime('%Y-%m')
print(f"\n>>> ACTUAL DATE RANGE: {first_date} to {last_date} <<<")
print(f">>> TOTAL MONTHS: {len(final_data)} <<<")

print("\n--- REAL INTEREST RATE REGIMES ---")
print(f"Mean Real Rate: {final_data['Real_Rate'].mean() * 12 * 100:.2f}% (annualized)")
print(f"Mean Nominal RF: {final_data['RF_Monthly'].mean() * 12 * 100:.2f}% (annualized)")

neg = (final_data['Real_Rate'] < 0).sum()
print(f"Negative Real Rate: {neg} months ({neg / len(final_data) * 100:.1f}%)")

pre_2008 = final_data[final_data.index < '2009-01-01']
post_2008 = final_data[final_data.index >= '2009-01-01']

print(f"\nPre-2009: {len(pre_2008)} months")
print(f"  Mean Nominal RF: {pre_2008['RF_Monthly'].mean() * 12 * 100:.2f}%")
print(f"  Mean Real Rate: {pre_2008['Real_Rate'].mean() * 12 * 100:.2f}%")

print(f"\nPost-2009: {len(post_2008)} months")
print(f"  Mean Nominal RF: {post_2008['RF_Monthly'].mean() * 12 * 100:.2f}%")
print(f"  Mean Real Rate: {post_2008['Real_Rate'].mean() * 12 * 100:.2f}%")

# Nominal return statistics
print("\n--- NOMINAL RETURN STATISTICS (Annualized %) ---")
summary = final_data[INDUSTRIES].describe().loc[['mean', 'std']].T
summary['mean'] = summary['mean'] * 12 * 100
summary['std'] = summary['std'] * (12 ** 0.5) * 100
summary.columns = ['Mean Return', 'Volatility']
print(summary.round(2))

print("\n" + "=" * 70)
print("DATASET COMPLETE")
print("=" * 70)
