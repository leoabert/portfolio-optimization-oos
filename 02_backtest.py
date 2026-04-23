"""
02_backtest.py - Core Rolling-Window Backtest
===============================================
Implements the out-of-sample evaluation framework of DeMiguel, Garlappi,
and Uppal (2009, RFS) for 1/N, mean-variance, and minimum-variance
strategies on the Fama-French 10 Industry Portfolios.

Inputs:  data/data_clean.csv
Outputs: output/strategy_returns.csv, output/strategy_turnover.csv,
         output/strategy_weights_mv.csv, output/strategy_weights_minvar.csv,
         output/strategy_hhi.csv, output/performance_table.csv
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

print("=" * 70)
print("BACKTESTING: DeMiguel et al. (2009) Replication")
print("=" * 70)

# ===== CONFIGURATION =====
from config import ESTIMATION_WINDOW, TRANSACTION_COST, RISK_AVERSION, INDUSTRIES

# ===== LOAD DATA =====
print("\n[1/5] Loading dataset...")
data = pd.read_csv('./data/data_clean.csv', index_col=0, parse_dates=True)
industries = INDUSTRIES
print(f"   Data: {len(data)} months, {len(industries)} industries")

# ===== RUN BACKTEST (shared engine) =====
print("\n[2/5] Running backtest via utils.run_backtest()...")

from utils import run_backtest, calculate_metrics

rf = data['RF_Monthly']
bt = run_backtest(data, rf, industries,
                  estimation_window=ESTIMATION_WINDOW,
                  transaction_cost=TRANSACTION_COST)

# Extract results
returns_1n = bt['1/N']['returns']
returns_mv = bt['Mean-Variance']['returns']
returns_minvar = bt['Minimum Variance']['returns']
turnover_1n = bt['1/N']['turnover']
turnover_mv = bt['Mean-Variance']['turnover']
turnover_minvar = bt['Minimum Variance']['turnover']
weights_1n = bt['1/N']['weights']
weights_mv = bt['Mean-Variance']['weights']
weights_minvar = bt['Minimum Variance']['weights']

common_index = returns_mv.index

print(f"   OOS period: {common_index[0].strftime('%Y-%m')} to {common_index[-1].strftime('%Y-%m')}")
print(f"   OOS months: {len(common_index)}")
print(f"   Mean turnover: 1/N={turnover_1n.mean()*100:.2f}%, "
      f"MV={turnover_mv.mean()*100:.2f}%, MinVar={turnover_minvar.mean()*100:.2f}%")

# ===== PERFORMANCE METRICS =====
print("\n[3/5] Calculating performance metrics...")

metrics_1n = calculate_metrics(returns_1n, rf, turnover_1n)
metrics_mv = calculate_metrics(returns_mv, rf, turnover_mv)
metrics_minvar = calculate_metrics(returns_minvar, rf, turnover_minvar)

results = pd.DataFrame({
    '1/N Benchmark': metrics_1n,
    'Mean-Variance': metrics_mv,
    'Minimum Variance': metrics_minvar
})

print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON (Net of Transaction Costs)")
print("=" * 70)
print(results.round(3))

ceq_loss_mv = metrics_1n['CEQ (%)'] - metrics_mv['CEQ (%)']
ceq_loss_minvar = metrics_1n['CEQ (%)'] - metrics_minvar['CEQ (%)']

print("\n--- WELFARE COSTS (CEQ Loss vs. 1/N) ---")
print(f"Mean-Variance:    {ceq_loss_mv:+.2f}% (negative = outperforms 1/N)")
print(f"Minimum Variance: {ceq_loss_minvar:+.2f}% (negative = outperforms 1/N)")

print(f"\n--- BACKTEST PERIOD ---")
print(f"Start: {common_index[0].strftime('%Y-%m')}")
print(f"End:   {common_index[-1].strftime('%Y-%m')}")
print(f"Months: {len(common_index)}")

# ===== SAVE ALL OUTPUTS =====
print("\n[4/5] Saving outputs...")

output_path = './output/performance_table.csv'
results.to_csv(output_path)
print(f"   Results saved: {output_path}")

# Strategy returns
returns_export = pd.DataFrame({
    '1/N': returns_1n,
    'Mean-Variance': returns_mv,
    'Minimum Variance': returns_minvar
})
returns_export.to_csv('./output/strategy_returns.csv')

# Strategy turnover
turnover_export = pd.DataFrame({
    '1/N': turnover_1n,
    'Mean-Variance': turnover_mv,
    'Minimum Variance': turnover_minvar
})
turnover_export.to_csv('./output/strategy_turnover.csv')

# Weights (for downstream scripts: figures, checklist)
weights_mv.to_csv('./output/strategy_weights_mv.csv')
weights_minvar.to_csv('./output/strategy_weights_minvar.csv')

# HHI
hhi_export = pd.DataFrame({
    '1/N': bt['1/N']['hhi'],
    'Mean-Variance': bt['Mean-Variance']['hhi'],
    'Minimum Variance': bt['Minimum Variance']['hhi']
})
hhi_export.to_csv('./output/strategy_hhi.csv')

print("   Exported: strategy_returns.csv, strategy_turnover.csv, "
      "strategy_weights_mv.csv, strategy_weights_minvar.csv, strategy_hhi.csv")

# ===== COUNTERFACTUAL: RF Artifact Check =====
print("\n" + "="*70)
print("COUNTERFACTUAL: Testing RF Artifact Hypothesis")
print("="*70)

# Test on post-2008 period where RF dropped
post_2008 = common_index >= '2009-01-01'
ret_1n_post = returns_1n[post_2008]
ret_mv_post = returns_mv[post_2008]
ret_minvar_post = returns_minvar[post_2008]
rf_post = rf.loc[common_index[post_2008]]

# Function to calculate CEQ spread
def calc_ceq_spread(ret1, ret2, rf_series):
    excess1 = ret1 - rf_series
    excess2 = ret2 - rf_series
    ceq1 = (excess1.mean() - (RISK_AVERSION/2) * excess1.var()) * 12 * 100
    ceq2 = (excess2.mean() - (RISK_AVERSION/2) * excess2.var()) * 12 * 100
    return ceq2 - ceq1  # Strategy - 1/N

# Scenario A: Actual RF (low ~1%)
spread_mv_actual = calc_ceq_spread(ret_1n_post, ret_mv_post, rf_post)
spread_minvar_actual = calc_ceq_spread(ret_1n_post, ret_minvar_post, rf_post)

# Scenario B: Counterfactual RF = 5% (high)
rf_counterfactual = pd.Series(0.05/12, index=rf_post.index)
spread_mv_cf = calc_ceq_spread(ret_1n_post, ret_mv_post, rf_counterfactual)
spread_minvar_cf = calc_ceq_spread(ret_1n_post, ret_minvar_post, rf_counterfactual)

# Print comparison
print(f"\nCEQ Spread (MV - 1/N) with Actual RF={rf_post.mean()*12*100:.2f}%: {spread_mv_actual:.4f}%")
print(f"CEQ Spread (MV - 1/N) with Counterfactual RF=5.00%: {spread_mv_cf:.4f}%")
print(f"Difference: {abs(spread_mv_actual - spread_mv_cf):.6f}% (should be ~0)")

print(f"\nCEQ Spread (MinVar - 1/N) with Actual RF={rf_post.mean()*12*100:.2f}%: {spread_minvar_actual:.4f}%")
print(f"CEQ Spread (MinVar - 1/N) with Counterfactual RF=5.00%: {spread_minvar_cf:.4f}%")
print(f"Difference: {abs(spread_minvar_actual - spread_minvar_cf):.6f}% (should be ~0)")

# Verdict
print(f"\n{'='*70}")
max_diff = max(abs(spread_mv_actual - spread_mv_cf), abs(spread_minvar_actual - spread_minvar_cf))
if max_diff < 0.01:  # 1 basis point tolerance
    print("ARTIFACT TEST PASSED: Spreads are RF-invariant (difference < 1bp)")
    print("  Mathematical proof: RF cancels in (CEQ_strategy - CEQ_1/N)")
    print("  Findings are NOT artifacts of low interest rate level.")
    print(f"  Max difference: {max_diff:.6f}% (economically negligible)")
else:
    print("WARNING: Difference > 1bp. Investigate further.")
    print(f"  Max difference: {max_diff:.6f}%")

print("="*70)
print("COUNTERFACTUAL TEST COMPLETE")
print("="*70)

# ===== SANITY CHECK =====
print("\n[5/5] Running sanity checks...")


def run_sanity_check(weights_1n, weights_mv, weights_minvar,
                     turnover_1n, turnover_mv, turnover_minvar,
                     common_index, industries):
    """Post-backtest sanity checks on weights, turnover, and dates."""
    n_assets = len(industries)
    tol = 1e-6

    print("\n" + "=" * 70)
    print("SANITY CHECK")
    print("=" * 70)

    # --- CHECK 1: 1/N weights = 1/N every period ---
    print("\n[1] 1/N Weight Verification:")
    expected = 1.0 / n_assets
    all_equal = np.allclose(weights_1n.values, expected, atol=tol)
    if all_equal:
        print(f"    PASS - All weights = {expected:.4f} in every period ({len(common_index)} months)")
    else:
        deviations = np.abs(weights_1n.values - expected)
        print(f"    FAIL - Max deviation from {expected:.4f}: {deviations.max():.2e}")

    # --- CHECK 2: MV weights ---
    print("\n[2] Mean-Variance Weight Verification:")
    mv_w = weights_mv
    mv_sums = mv_w.sum(axis=1)
    mv_sum_ok = np.allclose(mv_sums.values, 1.0, atol=tol)
    mv_nonneg = (mv_w.values >= -tol).all()
    mv_varies = mv_w.std(axis=0).sum() > tol  # weights change over time

    mv_hhi = (mv_w ** 2).sum(axis=1)
    print(f"    Sum-to-1:       {'PASS' if mv_sum_ok else 'FAIL'} "
          f"(range [{mv_sums.min():.8f}, {mv_sums.max():.8f}])")
    print(f"    Non-negative:   {'PASS' if mv_nonneg else 'FAIL'} "
          f"(min weight = {mv_w.values.min():.6e})")
    print(f"    Time-varying:   {'PASS' if mv_varies else 'FAIL'}")
    print(f"    HHI: min={mv_hhi.min():.4f}, mean={mv_hhi.mean():.4f}, max={mv_hhi.max():.4f}")

    # --- CHECK 3: MinVar weights ---
    print("\n[3] Minimum Variance Weight Verification:")
    minv_w = weights_minvar
    minv_sums = minv_w.sum(axis=1)
    minv_sum_ok = np.allclose(minv_sums.values, 1.0, atol=tol)
    minv_nonneg = (minv_w.values >= -tol).all()

    minv_hhi = (minv_w ** 2).sum(axis=1)
    hhi_comparison = minv_hhi.mean() < mv_hhi.mean()
    print(f"    Sum-to-1:       {'PASS' if minv_sum_ok else 'FAIL'} "
          f"(range [{minv_sums.min():.8f}, {minv_sums.max():.8f}])")
    print(f"    Non-negative:   {'PASS' if minv_nonneg else 'FAIL'} "
          f"(min weight = {minv_w.values.min():.6e})")
    print(f"    HHI: min={minv_hhi.min():.4f}, mean={minv_hhi.mean():.4f}, max={minv_hhi.max():.4f}")
    print(f"    Avg HHI < MV:   {'PASS' if hhi_comparison else 'FAIL'} "
          f"(MinVar {minv_hhi.mean():.4f} vs MV {mv_hhi.mean():.4f})")

    # --- CHECK 4: 3 random months, side-by-side weights ---
    print("\n[4] Sample Weights (3 random months):")
    np.random.seed(0)
    sample_idx = np.random.choice(len(common_index), size=3, replace=False)
    sample_idx.sort()
    for idx in sample_idx:
        date = common_index[idx]
        print(f"\n    {date.strftime('%Y-%m')}:")
        print(f"    {'Industry':<8}  {'1/N':>8}  {'MV':>8}  {'MinVar':>8}")
        print(f"    {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
        for ind in industries:
            w1 = weights_1n.loc[date, ind]
            w2 = mv_w.loc[date, ind]
            w3 = minv_w.loc[date, ind]
            print(f"    {ind:<8}  {w1:>8.4f}  {w2:>8.4f}  {w3:>8.4f}")
        print(f"    {'SUM':<8}  {weights_1n.loc[date].sum():>8.4f}  "
              f"{mv_w.loc[date].sum():>8.4f}  {minv_w.loc[date].sum():>8.4f}")

    # --- CHECK 5: Average monthly turnover ---
    print("\n[5] Average Monthly Turnover:")
    to_1n_pct = turnover_1n.mean() * 100
    to_mv_pct = turnover_mv.mean() * 100
    to_minv_pct = turnover_minvar.mean() * 100
    print(f"    1/N:      {to_1n_pct:>6.2f}%  (expected ~2-3%)")
    print(f"    MV:       {to_mv_pct:>6.2f}%  (expected ~15-20%)")
    print(f"    MinVar:   {to_minv_pct:>6.2f}%  (expected ~5-8%)")

    # --- CHECK 6: First and last OOS date ---
    print("\n[6] Out-of-Sample Date Range:")
    print(f"    First OOS month:  {common_index[0].strftime('%Y-%m')}")
    print(f"    Last OOS month:   {common_index[-1].strftime('%Y-%m')}")
    print(f"    Total OOS months: {len(common_index)}")

    # --- Overall verdict ---
    all_pass = all([all_equal, mv_sum_ok, mv_nonneg, mv_varies,
                    minv_sum_ok, minv_nonneg, hhi_comparison])
    print("\n" + "=" * 70)
    if all_pass:
        print("SANITY CHECK: ALL PASSED")
    else:
        print("SANITY CHECK: SOME CHECKS FAILED - REVIEW ABOVE")
    print("=" * 70)


run_sanity_check(weights_1n, weights_mv, weights_minvar,
                 turnover_1n, turnover_mv, turnover_minvar,
                 common_index, industries)
