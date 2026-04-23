#!/usr/bin/env python3
"""
05_robustness_bootstrap.py - Bootstrap & Sensitivity Analyses
================================================================
Transaction cost sensitivity, risk aversion sensitivity, block bootstrap
confidence intervals for DCEQ, and HHI-vs-welfare scatter plot.

Inputs:  data/data_clean.csv, output/strategy_returns.csv (from 02_backtest)
Outputs: output/bootstrap_ci.csv, output/bootstrap_ci.tex,
         output/fig_bootstrap_ci.png, output/fig_hhi_vs_welfare.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['grid.alpha'] = 0.15

from config import (OUTPUT_DIR, ESTIMATION_WINDOW, RISK_AVERSION, INDUSTRIES,
                    TRANSACTION_COSTS_SENSITIVITY, RISK_AVERSIONS_SENSITIVITY,
                    BOOTSTRAP_ITERATIONS, FIGURE_DPI, SPLIT_DATE)
from utils import (strategy_mean_variance, strategy_minimum_variance,
                   strategy_1n, calculate_drift_turnover, calculate_ceq,
                   calculate_hhi, load_cached_backtest, sync_to_latex)


def style_appendix_axes(fig, axes, y_grid_only=True):
    """Apply thesis-consistent white background and subtle grids."""
    fig.patch.set_facecolor('white')
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor('white')
        if y_grid_only:
            ax.grid(True, axis='y', alpha=0.10, linestyle='--')
            ax.grid(False, axis='x')
        else:
            ax.grid(True, alpha=0.10, linestyle='--')


def task_a_transaction_cost_sensitivity(returns_df, rf, industries):
    """Test sensitivity to transaction costs: 25, 50, 100 bps."""
    print("\n" + "="*60)
    print("TASK A: TRANSACTION COST SENSITIVITY")
    print("="*60)

    results = []
    n_assets = len(industries)
    T = len(returns_df)

    for c in TRANSACTION_COSTS_SENSITIVITY:
        print(f"Running with c = {c*10000:.0f} bps...")

        ret_1n, ret_mv, ret_minvar = [], [], []
        w_prev = {'1n': np.zeros(n_assets), 'mv': np.zeros(n_assets), 'minvar': np.zeros(n_assets)}
        dates = []

        for t in range(ESTIMATION_WINDOW, T):
            hist = returns_df[industries].iloc[t-ESTIMATION_WINDOW:t]
            mu, sigma = hist.mean().values, hist.cov().values
            r_t = returns_df[industries].iloc[t].values
            r_prev = returns_df[industries].iloc[t-1].values if t > ESTIMATION_WINDOW else r_t
            dates.append(returns_df.index[t])

            w_1n = strategy_1n(n_assets)
            w_mv = strategy_mean_variance(mu, sigma)
            w_minvar = strategy_minimum_variance(sigma)

            ret_1n.append(np.dot(w_1n, r_t) - c * calculate_drift_turnover(w_prev['1n'], w_1n, r_prev))
            ret_mv.append(np.dot(w_mv, r_t) - c * calculate_drift_turnover(w_prev['mv'], w_mv, r_prev))
            ret_minvar.append(np.dot(w_minvar, r_t) - c * calculate_drift_turnover(w_prev['minvar'], w_minvar, r_prev))

            w_prev = {'1n': w_1n, 'mv': w_mv, 'minvar': w_minvar}

        ret_1n = pd.Series(ret_1n, index=dates)
        ret_mv = pd.Series(ret_mv, index=dates)
        ret_minvar = pd.Series(ret_minvar, index=dates)
        rf_aligned = rf.loc[ret_1n.index]

        ceq_1n = calculate_ceq(ret_1n, rf_aligned)
        ceq_mv = calculate_ceq(ret_mv, rf_aligned)
        ceq_minvar = calculate_ceq(ret_minvar, rf_aligned)

        results.append({
            'TC (bps)': int(c * 10000),
            'CEQ(1/N)': ceq_1n, 'CEQ(MV)': ceq_mv, 'CEQ(MinVar)': ceq_minvar,
            'ΔCEQ(MV)': ceq_1n - ceq_mv, 'ΔCEQ(MinVar)': ceq_1n - ceq_minvar
        })

    df = pd.DataFrame(results).set_index('TC (bps)')
    print(df.round(2))
    # Publication table is in 08_robustness_tables.py (Table 5)
    return df


def task_b_risk_aversion_sensitivity(returns_df, rf, industries):
    """Test sensitivity to risk aversion: γ = 1, 2, 5."""
    print("\n" + "="*60)
    print("TASK B: RISK AVERSION SENSITIVITY")
    print("="*60)

    results = []
    n_assets = len(industries)
    T = len(returns_df)
    c = 0.0050

    for gamma in RISK_AVERSIONS_SENSITIVITY:
        print(f"Running with γ = {gamma}...")

        ret_1n, ret_mv, hhi_mv = [], [], []
        w_prev = {'1n': np.zeros(n_assets), 'mv': np.zeros(n_assets)}
        dates = []

        for t in range(ESTIMATION_WINDOW, T):
            hist = returns_df[industries].iloc[t-ESTIMATION_WINDOW:t]
            mu, sigma = hist.mean().values, hist.cov().values
            r_t = returns_df[industries].iloc[t].values
            r_prev = returns_df[industries].iloc[t-1].values if t > ESTIMATION_WINDOW else r_t
            dates.append(returns_df.index[t])

            w_1n = strategy_1n(n_assets)
            w_mv = strategy_mean_variance(mu, sigma, gamma=gamma)

            ret_1n.append(np.dot(w_1n, r_t) - c * calculate_drift_turnover(w_prev['1n'], w_1n, r_prev))
            ret_mv.append(np.dot(w_mv, r_t) - c * calculate_drift_turnover(w_prev['mv'], w_mv, r_prev))
            hhi_mv.append(calculate_hhi(w_mv))

            w_prev = {'1n': w_1n, 'mv': w_mv}

        ret_1n = pd.Series(ret_1n, index=dates)
        ret_mv = pd.Series(ret_mv, index=dates)
        rf_aligned = rf.loc[ret_1n.index]

        ceq_1n = calculate_ceq(ret_1n, rf_aligned, gamma=gamma)
        ceq_mv = calculate_ceq(ret_mv, rf_aligned, gamma=gamma)

        results.append({
            'γ': gamma, 'CEQ(1/N)': ceq_1n, 'CEQ(MV)': ceq_mv,
            'ΔCEQ(MV)': ceq_1n - ceq_mv, 'HHI(MV)': np.mean(hhi_mv)
        })

    df = pd.DataFrame(results).set_index('γ')
    print(df.round(3))
    # Publication table is in 08_robustness_tables.py (Table 4)
    return df


# Momentum decomposition figure removed — not referenced in thesis
# def task_c_momentum_decomposition(returns_df, rf, industries):
#     """Analyze MV weight-momentum correlation."""
#     (entire function removed — generated fig_momentum_decomposition.png
#      and momentum_decomposition.csv which are not used in the thesis)


def task_d_bootstrap_ci(returns_df, rf, industries, n_bootstrap=None):
    """Bootstrap confidence intervals for ΔCEQ."""
    if n_bootstrap is None:
        n_bootstrap = BOOTSTRAP_ITERATIONS

    print("\n" + "="*60)
    print(f"TASK D: BOOTSTRAP CI ({n_bootstrap} iterations)")
    print("="*60)

    from utils import run_backtest
    bt = load_cached_backtest(OUTPUT_DIR)
    if bt is None:
        bt = run_backtest(returns_df, rf, industries, ESTIMATION_WINDOW)

    ret_1n = bt['1/N']['returns']
    ret_mv = bt['Mean-Variance']['returns']
    ret_minvar = bt['Minimum Variance']['returns']
    rf_aligned = rf.loc[ret_1n.index]

    block_size = 12
    n_obs = len(ret_1n)
    n_blocks = n_obs // block_size

    def ceq_boot(r, rf_b):
        excess = r.values - rf_b.values
        return (excess.mean() - (RISK_AVERSION / 2) * excess.var()) * 12 * 100

    delta_mv_boot, delta_minvar_boot = [], []
    np.random.seed(42)

    for b in range(n_bootstrap):
        if (b+1) % 2000 == 0:
            print(f"  Iteration {b+1}/{n_bootstrap}")

        block_idx = np.random.choice(n_blocks, size=n_blocks, replace=True)
        boot_idx = []
        for bi in block_idx:
            boot_idx.extend(range(bi * block_size, min((bi+1) * block_size, n_obs)))
        boot_idx = boot_idx[:n_obs]

        ceq_1n = ceq_boot(ret_1n.iloc[boot_idx], rf_aligned.iloc[boot_idx])
        ceq_mv = ceq_boot(ret_mv.iloc[boot_idx], rf_aligned.iloc[boot_idx])
        ceq_minvar = ceq_boot(ret_minvar.iloc[boot_idx], rf_aligned.iloc[boot_idx])

        delta_mv_boot.append(ceq_1n - ceq_mv)
        delta_minvar_boot.append(ceq_1n - ceq_minvar)

    ci_mv = np.percentile(delta_mv_boot, [2.5, 97.5])
    ci_minvar = np.percentile(delta_minvar_boot, [2.5, 97.5])

    delta_mv_point = calculate_ceq(ret_1n, rf_aligned) - calculate_ceq(ret_mv, rf_aligned)
    delta_minvar_point = calculate_ceq(ret_1n, rf_aligned) - calculate_ceq(ret_minvar, rf_aligned)

    print(f"\nΔCEQ(MV): {delta_mv_point:.2f}% [{ci_mv[0]:.2f}%, {ci_mv[1]:.2f}%]")
    print(f"ΔCEQ(MinVar): {delta_minvar_point:.2f}% [{ci_minvar[0]:.2f}%, {ci_minvar[1]:.2f}%]")
    print(f"MV significant: {ci_mv[0] > 0}, MinVar significant: {ci_minvar[0] > 0}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    style_appendix_axes(fig, axes, y_grid_only=True)
    axes[0].hist(delta_mv_boot, bins=50, density=True, alpha=0.7, color='#E74C3C')
    axes[0].axvline(delta_mv_point, color='black', linewidth=2)
    axes[0].axvline(ci_mv[0], color='gray', linestyle='--')
    axes[0].axvline(ci_mv[1], color='gray', linestyle='--')
    axes[0].axvline(0, color='blue', linestyle=':')
    axes[0].set_title('Bootstrap: ΔCEQ(MV)')

    axes[1].hist(delta_minvar_boot, bins=50, density=True, alpha=0.7, color='#3498DB')
    axes[1].axvline(delta_minvar_point, color='black', linewidth=2)
    axes[1].axvline(ci_minvar[0], color='gray', linestyle='--')
    axes[1].axvline(ci_minvar[1], color='gray', linestyle='--')
    axes[1].axvline(0, color='blue', linestyle=':')
    axes[1].set_title('Bootstrap: ΔCEQ(MinVar)')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}fig_bootstrap_ci.png', dpi=FIGURE_DPI,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    # LaTeX export (disabled for portable submission):
    # sync_to_latex(f'{OUTPUT_DIR}fig_bootstrap_ci.png', LATEX_FIG_APP_DIR)

    summary = pd.DataFrame({
        'Strategy': ['Mean-Variance', 'Minimum Variance'],
        'Point (%)': [delta_mv_point, delta_minvar_point],
        'CI Low (%)': [ci_mv[0], ci_minvar[0]],
        'CI High (%)': [ci_mv[1], ci_minvar[1]],
        'Significant': [ci_mv[0] > 0, ci_minvar[0] > 0]
    })
    summary.to_csv(f'{OUTPUT_DIR}bootstrap_ci.csv', index=False)
    # Escape % in column names for LaTeX safety
    summary_latex = summary.copy()
    summary_latex.columns = [c.replace('%', r'\%') for c in summary_latex.columns]
    summary_latex.to_latex(f'{OUTPUT_DIR}bootstrap_ci.tex', float_format='%.2f',
                           index=False, escape=False)
    # LaTeX export (disabled for portable submission):
    # sync_to_latex(f'{OUTPUT_DIR}bootstrap_ci.tex', LATEX_TABLES_DIR)
    return summary


def task_e_hhi_welfare_scatter(returns_df, rf, industries):
    """Scatter: HHI vs. ΔCEQ."""
    print("\n" + "="*60)
    print("TASK E: HHI vs. WELFARE SCATTER")
    print("="*60)

    from utils import run_backtest
    bt = load_cached_backtest(OUTPUT_DIR)
    if bt is None:
        bt = run_backtest(returns_df, rf, industries, ESTIMATION_WINDOW)

    ret_1n = bt['1/N']['returns']
    ret_mv = bt['Mean-Variance']['returns']
    hhi_mv = bt['Mean-Variance']['hhi']
    rf_aligned = rf.loc[ret_1n.index]

    excess_1n = ret_1n - rf_aligned
    excess_mv = ret_mv - rf_aligned

    delta_ceq_rolling, hhi_rolling, dates_rolling = [], [], []
    for t in range(12, len(ret_1n)):
        e1n = excess_1n.iloc[t-12:t]
        emv = excess_mv.iloc[t-12:t]
        ceq_1n = (e1n.mean() - (RISK_AVERSION / 2) * e1n.var()) * 12 * 100
        ceq_mv = (emv.mean() - (RISK_AVERSION / 2) * emv.var()) * 12 * 100
        delta_ceq_rolling.append(ceq_1n - ceq_mv)
        hhi_rolling.append(hhi_mv.iloc[t-12:t].mean())
        dates_rolling.append(ret_1n.index[t])

    delta_ceq_rolling = pd.Series(delta_ceq_rolling, index=dates_rolling)
    hhi_rolling = pd.Series(hhi_rolling, index=dates_rolling)

    corr, p_val = stats.pearsonr(hhi_rolling, delta_ceq_rolling)
    print(f"Correlation: ρ = {corr:.3f}, p = {p_val:.4f}")

    fig, ax = plt.subplots(figsize=(10, 7))
    style_appendix_axes(fig, ax, y_grid_only=True)
    split = pd.Timestamp(SPLIT_DATE)
    pre_mask = delta_ceq_rolling.index < split

    ax.scatter(hhi_rolling[pre_mask], delta_ceq_rolling[pre_mask], alpha=0.5, c='#3498DB', label='Pre-2009', s=40)
    ax.scatter(hhi_rolling[~pre_mask], delta_ceq_rolling[~pre_mask], alpha=0.5, c='#E74C3C', label='Post-2009', s=40)

    z = np.polyfit(hhi_rolling, delta_ceq_rolling, 1)
    x_line = np.linspace(hhi_rolling.min(), hhi_rolling.max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), 'k-', linewidth=2)

    ax.set_xlabel('HHI (Concentration)')
    ax.set_ylabel('ΔCEQ (%, 12-month rolling)')
    ax.set_title('Concentration Risk and Welfare Costs')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}fig_hhi_vs_welfare.png', dpi=FIGURE_DPI,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    # LaTeX export (disabled for portable submission):
    # sync_to_latex(f'{OUTPUT_DIR}fig_hhi_vs_welfare.png', LATEX_FIG_APP_DIR)

    return corr, p_val


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*70)
    print("ADDITIONAL ROBUSTNESS ANALYSES")
    print("="*70)

    # Use local data (no network calls — avoids hangs)
    import pandas as pd
    data = pd.read_csv('./data/data_clean.csv', index_col=0, parse_dates=True)
    industries = INDUSTRIES
    returns_df = data[industries].copy()
    rf = data['RF_Monthly']

    task_a_transaction_cost_sensitivity(returns_df, rf, industries)
    task_b_risk_aversion_sensitivity(returns_df, rf, industries)
    # task_c_momentum_decomposition removed — not referenced in thesis
    task_d_bootstrap_ci(returns_df, rf, industries)
    task_e_hhi_welfare_scatter(returns_df, rf, industries)

    print("\n" + "="*70)
    print("ALL ADDITIONAL ANALYSES COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
