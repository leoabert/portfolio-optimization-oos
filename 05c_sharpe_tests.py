#!/usr/bin/env python3
"""
05c_sharpe_tests.py - Sharpe Ratio Equality Tests
===================================================
Jobson-Korkie test with Memmel (2003) finite-sample correction.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import OUTPUT_DIR

print("=" * 60)
print("SHARPE RATIO EQUALITY TESTS")
print("=" * 60)

# =====================================================================
# 1. Load data
# =====================================================================
returns = pd.read_csv(f'{OUTPUT_DIR}strategy_returns.csv',
                      index_col=0, parse_dates=True)
data = pd.read_csv('./data/data_clean.csv',
                    index_col=0, parse_dates=True)
rf = data['RF_Monthly'].reindex(returns.index)

excess = returns.subtract(rf, axis=0)
T = len(excess)

print(f"  OOS observations: {T} months")

# =====================================================================
# 2. Annualised Sharpe ratios
# =====================================================================
sr_ann = (excess.mean() / excess.std()) * np.sqrt(12)
print(f"\n  Annualised Sharpe ratios:")
for col in excess.columns:
    print(f"    {col}: {sr_ann[col]:.3f}")


# =====================================================================
# 3. Jobson-Korkie / Memmel (2003) test
# =====================================================================
def jk_memmel_test(exc_i, exc_n):
    """
    Jobson-Korkie z-test with Memmel (2003) finite-sample correction.

    Tests H0: SR_i = SR_n (two-sided).
    Returns (z_stat, p_value).
    """
    mu_i = exc_i.mean()
    mu_n = exc_n.mean()
    sig_i = exc_i.std(ddof=1)
    sig_n = exc_n.std(ddof=1)
    sig_in = np.cov(exc_i, exc_n, ddof=1)[0, 1]
    # T_OOS = T - M (DeMiguel et al. 2009, fn. 16)
    n = len(exc_i)

    numerator = sig_n * mu_i - sig_i * mu_n

    theta = (1.0 / n) * (
        2 * sig_i**2 * sig_n**2
        - 2 * sig_i * sig_n * sig_in
        + 0.5 * mu_i**2 * sig_n**2
        + 0.5 * mu_n**2 * sig_i**2
        - (mu_i * mu_n / (sig_i * sig_n)) * sig_in**2
    )

    z = numerator / np.sqrt(theta)
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p


# =====================================================================
# 4. Run tests
# =====================================================================
pairs = [
    ('1/N', 'Mean-Variance', 'MV'),
    ('1/N', 'Minimum Variance', 'MinVar'),
]

results = []

print(f"\n{'='*60}")
print(f"  {'Pair':<20s} | {'SR_1':>6s} | {'SR_2':>6s} | {'DSR':>7s} | "
      f"{'JK z':>7s} | {'JK p':>6s}")
print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-"
      f"{'-'*7}-+-{'-'*6}")

for col_i, col_n, short in pairs:
    exc_i = excess[col_i].values
    exc_n = excess[col_n].values

    sr_i = sr_ann[col_i]
    sr_n = sr_ann[col_n]
    dsr = sr_i - sr_n

    # Jobson-Korkie / Memmel
    jk_z, jk_p = jk_memmel_test(exc_i, exc_n)

    results.append({
        'Pair': f'1/N vs {short}',
        'SR_1': round(sr_i, 3),
        'SR_2': round(sr_n, 3),
        'DeltaSR': round(dsr, 3),
        'JK_z': round(jk_z, 3),
        'JK_p': round(jk_p, 3),
    })

    print(f"  {'1/N vs '+short:<20s} | {sr_i:6.3f} | {sr_n:6.3f} | {dsr:+7.3f} | "
          f"{jk_z:+7.3f} | {jk_p:6.3f}")

# =====================================================================
# 5. Export CSV
# =====================================================================
df = pd.DataFrame(results)
csv_path = f'{OUTPUT_DIR}sharpe_test_results.csv'
df.to_csv(csv_path, index=False)
print(f"\n  CSV saved: {csv_path}")

# =====================================================================
# 6. Export footnote text
# =====================================================================
r_mv = results[0]
r_minvar = results[1]

footnote = (
    r"The \citet{memmelPerformanceHypothesisTesting2003}-corrected"
    "\n"
    r"Jobson--Korkie test for equal Sharpe ratios yields"
    "\n"
    f"$z = {r_mv['JK_z']:.2f}$ ($p = {r_mv['JK_p']:.3f}$) for the"
    "\n"
    r"\oneN{} versus mean-variance comparison and"
    "\n"
    f"$z = {r_minvar['JK_z']:.2f}$ ($p = {r_minvar['JK_p']:.3f}$) for"
    "\n"
    r"\oneN{} versus minimum-variance."
)

fn_path = f'{OUTPUT_DIR}sharpe_test_footnote.txt'
with open(fn_path, 'w', encoding='utf-8') as f:
    f.write(footnote + '\n')
print(f"  Footnote saved: {fn_path}")

# =====================================================================
# 7. Summary
# =====================================================================
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(df.to_string(index=False))

print("\n" + "=" * 60)
print("SHARPE RATIO TESTS COMPLETE")
print("=" * 60)
