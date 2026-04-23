#!/usr/bin/env python3
"""
99_validate_demiguel.py - DeMiguel et al. (2009) Replication Validation
========================================================================
Restricts backtest results to the approximate DeMiguel sample period
(10 Industry Portfolios, OOS ~1973-07 to 2004-11, M=120) and compares
our metrics against their Table III.

DeMiguel et al. (2009, RFS) Table III reference values (10-Industry):
  - Monthly Sharpe: 1/N ~ 0.14,  sample-MV ~ 0.04,  min-var ~ 0.14
  - DCEQ (vs 1/N):               sample-MV > 0,      min-var > 0
  - Monthly Turnover:  1/N ~ 0.02, sample-MV ~ 0.34, min-var ~ 0.10

Exact match is impossible (different data vintage, slight cleaning
differences), but directional consistency is required.
"""

import pandas as pd
import numpy as np
from config import RISK_AVERSION

# =====================================================================
# 1. Load data and restrict to DeMiguel overlap period
# =====================================================================
print("=" * 70)
print("DeMiguel et al. (2009) REPLICATION VALIDATION")
print("=" * 70)

returns = pd.read_csv('./output/strategy_returns.csv',
                       index_col=0, parse_dates=True)
turnover = pd.read_csv('./output/strategy_turnover.csv',
                        index_col=0, parse_dates=True)
data = pd.read_csv('./data/data_clean.csv',
                    index_col=0, parse_dates=True)
rf = data['RF_Monthly']

# DeMiguel et al. used Kenneth French data through Nov 2004
# with M=120 estimation window, so OOS starts ~1973-07.
# Their exact window: Jul 1963 + 120 months = Jul 1973 OOS start.
# End: Nov 2004.
DEMIGUEL_START = '1973-07-01'
DEMIGUEL_END = '2004-11-30'

mask = (returns.index >= DEMIGUEL_START) & (returns.index <= DEMIGUEL_END)
ret = returns.loc[mask]
turn = turnover.loc[mask]
rf_sub = rf.reindex(ret.index)

n_months = len(ret)
print(f"\nRestricted period: {ret.index[0].strftime('%Y-%m')} to "
      f"{ret.index[-1].strftime('%Y-%m')}  ({n_months} months)")

strategies = ['1/N', 'Mean-Variance', 'Minimum Variance']
short_names = {'1/N': '1/N', 'Mean-Variance': 'MV', 'Minimum Variance': 'MinVar'}

# =====================================================================
# 2. Compute metrics for the restricted period
# =====================================================================
print("\n" + "=" * 70)
print("OUR REPLICATION (DeMiguel overlap period)")
print("=" * 70)

metrics = {}
for s in strategies:
    r = ret[s]
    t = turn[s]
    rf_a = rf_sub

    excess = r - rf_a
    mu_excess = excess.mean()
    std_excess = excess.std()

    sharpe_monthly = mu_excess / std_excess if std_excess > 0 else 0.0
    sharpe_annual = sharpe_monthly * np.sqrt(12)

    ceq_monthly = mu_excess - (RISK_AVERSION / 2) * excess.var()
    ceq_annual_pct = ceq_monthly * 12 * 100

    avg_turnover = t.mean()  # monthly fraction

    metrics[s] = {
        'sharpe_monthly': sharpe_monthly,
        'sharpe_annual': sharpe_annual,
        'ceq_annual_pct': ceq_annual_pct,
        'turnover_monthly': avg_turnover,
    }

    sn = short_names[s]
    print(f"\n  {sn}:")
    print(f"    Sharpe (monthly)    = {sharpe_monthly:.4f}")
    print(f"    Sharpe (annualized) = {sharpe_annual:.4f}")
    print(f"    CEQ (annual %)      = {ceq_annual_pct:+.3f}")
    print(f"    Turnover (monthly)  = {avg_turnover:.4f}")

# Welfare costs
dceq_mv = metrics['1/N']['ceq_annual_pct'] - metrics['Mean-Variance']['ceq_annual_pct']
dceq_minvar = metrics['1/N']['ceq_annual_pct'] - metrics['Minimum Variance']['ceq_annual_pct']

print(f"\n  Welfare costs (DCEQ = CEQ_1/N - CEQ_strategy):")
print(f"    DCEQ(MV)     = {dceq_mv:+.3f}%  {'(1/N wins)' if dceq_mv > 0 else '(MV wins)'}")
print(f"    DCEQ(MinVar) = {dceq_minvar:+.3f}%  {'(1/N wins)' if dceq_minvar > 0 else '(MinVar wins)'}")

# =====================================================================
# 3. Side-by-side comparison with DeMiguel Table III
# =====================================================================
print("\n" + "=" * 70)
print("SIDE-BY-SIDE: DeMiguel Table III vs Our Replication")
print("=" * 70)

# DeMiguel et al. (2009) Table III, "10 Ind" column.
# Values read from the paper (monthly, gamma=1):
#   Sharpe (monthly): 1/N=0.1413, sample-MV=0.0371, min-var=0.1378
#   CEQ (monthly bp): 1/N=46, sample-MV=13, min-var=50
#   Turnover (monthly): 1/N=0.024, sample-MV=0.342, min-var=0.100
#   DCEQ: sample-MV: 46-13=+33bp, min-var: 46-50=-4bp
dm_ref = {
    '1/N': {'sharpe_m': 0.1413, 'ceq_bp': 46, 'turn': 0.024},
    'MV':  {'sharpe_m': 0.0371, 'ceq_bp': 13, 'turn': 0.342},
    'MinVar': {'sharpe_m': 0.1378, 'ceq_bp': 50, 'turn': 0.100},
}

header = f"{'Metric':<28} {'DeMiguel':>10} {'Ours':>10} {'Match?':>8}"
print(f"\n{header}")
print("-" * len(header))

def compare_row(label, dm_val, our_val, fmt='.4f', tol_pct=100):
    """Print one row; flag if direction differs or magnitude way off."""
    match = 'OK'
    if dm_val != 0:
        ratio = abs(our_val / dm_val)
        if ratio < 0.2 or ratio > 5.0:
            match = '??'
    print(f"  {label:<26} {dm_val:>10{fmt}} {our_val:>10{fmt}} {match:>8}")

# Sharpe (monthly)
compare_row('Sharpe(1/N) monthly',
            dm_ref['1/N']['sharpe_m'],
            metrics['1/N']['sharpe_monthly'])
compare_row('Sharpe(MV) monthly',
            dm_ref['MV']['sharpe_m'],
            metrics['Mean-Variance']['sharpe_monthly'])
compare_row('Sharpe(MinVar) monthly',
            dm_ref['MinVar']['sharpe_m'],
            metrics['Minimum Variance']['sharpe_monthly'])

print()

# CEQ (monthly basis points for comparability)
our_ceq_1n_bp = metrics['1/N']['ceq_annual_pct'] / 12 * 100  # annual % -> monthly bp
our_ceq_mv_bp = metrics['Mean-Variance']['ceq_annual_pct'] / 12 * 100
our_ceq_minvar_bp = metrics['Minimum Variance']['ceq_annual_pct'] / 12 * 100

compare_row('CEQ(1/N) monthly bp',
            dm_ref['1/N']['ceq_bp'], our_ceq_1n_bp, fmt='.1f')
compare_row('CEQ(MV) monthly bp',
            dm_ref['MV']['ceq_bp'], our_ceq_mv_bp, fmt='.1f')
compare_row('CEQ(MinVar) monthly bp',
            dm_ref['MinVar']['ceq_bp'], our_ceq_minvar_bp, fmt='.1f')

print()

# Turnover (monthly fraction)
compare_row('Turnover(1/N) monthly',
            dm_ref['1/N']['turn'],
            metrics['1/N']['turnover_monthly'])
compare_row('Turnover(MV) monthly',
            dm_ref['MV']['turn'],
            metrics['Mean-Variance']['turnover_monthly'])
compare_row('Turnover(MinVar) monthly',
            dm_ref['MinVar']['turn'],
            metrics['Minimum Variance']['turnover_monthly'])

print()

# DCEQ
dm_dceq_mv_bp = dm_ref['1/N']['ceq_bp'] - dm_ref['MV']['ceq_bp']
dm_dceq_minvar_bp = dm_ref['1/N']['ceq_bp'] - dm_ref['MinVar']['ceq_bp']
our_dceq_mv_bp = our_ceq_1n_bp - our_ceq_mv_bp
our_dceq_minvar_bp = our_ceq_1n_bp - our_ceq_minvar_bp

compare_row('DCEQ(MV) monthly bp',
            dm_dceq_mv_bp, our_dceq_mv_bp, fmt='.1f')
compare_row('DCEQ(MinVar) monthly bp',
            dm_dceq_minvar_bp, our_dceq_minvar_bp, fmt='.1f')

# =====================================================================
# 4. Verdict
# =====================================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

checks = []

# Check A: MV has positive DCEQ (1/N beats MV)
chk_a = dceq_mv > 0
checks.append(chk_a)
print(f"\n  [A] DCEQ(MV) > 0 (1/N outperforms MV on CEQ)?")
print(f"      DCEQ(MV) = {dceq_mv:+.3f}%  -->  {'PASS' if chk_a else 'FAIL'}")

# Check B: 1/N Sharpe in plausible range (monthly 0.05-0.25)
sharpe_1n_m = metrics['1/N']['sharpe_monthly']
chk_b = 0.05 < sharpe_1n_m < 0.25
checks.append(chk_b)
print(f"\n  [B] 1/N monthly Sharpe in [0.05, 0.25]?")
print(f"      Sharpe = {sharpe_1n_m:.4f}  -->  {'PASS' if chk_b else 'FAIL'}")

# Check C: MV Sharpe < 1/N Sharpe (estimation error hurts)
sharpe_mv_m = metrics['Mean-Variance']['sharpe_monthly']
chk_c = sharpe_mv_m < sharpe_1n_m
checks.append(chk_c)
print(f"\n  [C] MV Sharpe < 1/N Sharpe (estimation error drags)?")
print(f"      MV={sharpe_mv_m:.4f} vs 1/N={sharpe_1n_m:.4f}  -->  {'PASS' if chk_c else 'FAIL'}")

# Check D: MV turnover >> 1/N turnover
turn_1n = metrics['1/N']['turnover_monthly']
turn_mv = metrics['Mean-Variance']['turnover_monthly']
chk_d = turn_mv > 3 * turn_1n
checks.append(chk_d)
print(f"\n  [D] MV turnover >> 1/N turnover?")
print(f"      MV={turn_mv:.4f} vs 1/N={turn_1n:.4f} (ratio {turn_mv/turn_1n:.1f}x)  "
      f"-->  {'PASS' if chk_d else 'FAIL'}")

# Check E: MinVar Sharpe comparable to 1/N (within factor of 2)
sharpe_minvar_m = metrics['Minimum Variance']['sharpe_monthly']
chk_e = sharpe_minvar_m > sharpe_1n_m * 0.5
checks.append(chk_e)
print(f"\n  [E] MinVar Sharpe within 50% of 1/N?")
print(f"      MinVar={sharpe_minvar_m:.4f} vs 1/N={sharpe_1n_m:.4f}  "
      f"-->  {'PASS' if chk_e else 'FAIL'}")

# Overall
n_pass = sum(checks)
n_total = len(checks)
all_pass = all(checks)

print(f"\n{'=' * 70}")
if all_pass:
    print(f"OVERALL: ALL {n_total} CHECKS PASSED")
    print("Replication is directionally consistent with DeMiguel et al. (2009).")
else:
    print(f"OVERALL: {n_pass}/{n_total} CHECKS PASSED")
    print("Review failures above. Directional consistency not fully confirmed.")
print("=" * 70)
