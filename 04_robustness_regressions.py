#!/usr/bin/env python3
"""
04_robustness_regressions.py - Robustness Regressions & Visualisations
========================================================================
OLS regressions of rolling welfare cost (DCEQ) on the real interest rate,
subperiod bar charts, regression scatter plots, and dual-axis time series.

Inputs:  data/data_clean.csv
Outputs: output/regression_simple.txt, output/regression_with_vix.txt,
         output/fig_subperiod_barplot.png, output/fig_regression_scatter.png,
         output/fig_timeseries_dual.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['grid.alpha'] = 0.15

# Import from central modules
from config import (OUTPUT_DIR, ESTIMATION_WINDOW, TRANSACTION_COST,
                    RISK_AVERSION, INDUSTRIES, SPLIT_DATE,
                    ESTIMATION_WINDOWS_SENSITIVITY, FIGURE_DPI)
from utils import (run_backtest, load_cached_backtest,
                   calculate_ceq, calculate_sharpe, calculate_metrics)


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


def task1_subperiod_comparison(returns_dict, rf, hhi_mv, split_date):
    """Compare welfare costs pre-2008 vs post-2008."""
    print("\n" + "="*60)
    print("TASK 1: SUBPERIOD COMPARISON")
    print("="*60)

    split = pd.Timestamp(split_date)
    pre_mask = returns_dict['1/N'].index < split
    post_mask = returns_dict['1/N'].index >= split

    metrics = {}
    for name, mask in [('Pre-2009', pre_mask), ('Post-2009', post_mask)]:
        ret_1n = returns_dict['1/N'][mask]
        ret_mv = returns_dict['Mean-Variance'][mask]
        ret_minvar = returns_dict['Minimum Variance'][mask]
        rf_period = rf.loc[ret_1n.index]
        hhi_period = hhi_mv.loc[hhi_mv.index.isin(ret_1n.index)]

        ceq_1n = calculate_ceq(ret_1n, rf_period)
        ceq_mv = calculate_ceq(ret_mv, rf_period)
        ceq_minvar = calculate_ceq(ret_minvar, rf_period)

        metrics[name] = {
            'ΔCEQ (MV) [%]': ceq_1n - ceq_mv,
            'ΔCEQ (MinVar) [%]': ceq_1n - ceq_minvar,
            'HHI (MV)': hhi_period.mean(),
            'Observations': len(ret_1n)
        }

    metrics['Difference'] = {
        'ΔCEQ (MV) [%]': metrics['Post-2009']['ΔCEQ (MV) [%]'] - metrics['Pre-2009']['ΔCEQ (MV) [%]'],
        'ΔCEQ (MinVar) [%]': metrics['Post-2009']['ΔCEQ (MinVar) [%]'] - metrics['Pre-2009']['ΔCEQ (MinVar) [%]'],
        'HHI (MV)': metrics['Post-2009']['HHI (MV)'] - metrics['Pre-2009']['HHI (MV)'],
        'Observations': '-'
    }

    df = pd.DataFrame(metrics).T
    print("\nSubperiod Comparison Results:")
    print("-"*60)
    print(df.to_string())

    return df


def task2_window_robustness(returns_df, rf, industries, windows):
    """Test sensitivity to estimation window length."""
    print("\n" + "="*60)
    print("TASK 2: ESTIMATION WINDOW ROBUSTNESS")
    print("="*60)

    results = []
    for M in windows:
        print(f"\nRunning backtest with M={M} months...")
        bt = run_backtest(returns_df, rf, industries, estimation_window=M)

        ret_1n = bt['1/N']['returns']
        ret_mv = bt['Mean-Variance']['returns']
        ret_minvar = bt['Minimum Variance']['returns']
        rf_aligned = rf.loc[ret_1n.index]

        ceq_1n = calculate_ceq(ret_1n, rf_aligned)
        ceq_mv = calculate_ceq(ret_mv, rf_aligned)
        ceq_minvar = calculate_ceq(ret_minvar, rf_aligned)

        results.append({
            'Window (M)': M,
            'OOS Start': returns_df.index[M].strftime('%Y-%m'),
            'ΔCEQ(MV) [%]': ceq_1n - ceq_mv,
            'ΔCEQ(MinVar) [%]': ceq_1n - ceq_minvar,
            'HHI(MV)': bt['Mean-Variance']['hhi'].mean(),
            'Sharpe(1/N)': calculate_sharpe(ret_1n, rf_aligned),
            'Observations': len(ret_1n)
        })

    df = pd.DataFrame(results).set_index('Window (M)')
    print("\nResults:")
    print(df.to_string())
    return df


def task3_regression_simple(delta_ceq, real_rate_lagged):
    """Simple regression: ΔCEQ ~ Real Rate (lagged)."""
    print("\n" + "="*60)
    print("TASK 3: ΔCEQ REGRESSION ON REAL RATE")
    print("="*60)

    common_idx = delta_ceq.index.intersection(real_rate_lagged.index)
    y = delta_ceq.loc[common_idx]
    X = sm.add_constant(real_rate_lagged.loc[common_idx])

    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    print(model.summary())

    coef = model.params.iloc[1]
    pval = model.pvalues.iloc[1]
    print(f"\nβ₁ = {coef:.4f} ({'NEGATIVE ✓' if coef < 0 else 'positive'}), p = {pval:.4f}")

    return model


def task4_regression_with_vix(delta_ceq, real_rate_lagged, vix):
    """Regression with VIX control."""
    print("\n" + "="*60)
    print("TASK 4: ΔCEQ REGRESSION WITH VIX CONTROL")
    print("="*60)

    common_idx = delta_ceq.index.intersection(real_rate_lagged.index).intersection(vix.index)
    y = delta_ceq.loc[common_idx]
    X = pd.DataFrame({
        'Real_Rate_Lag': real_rate_lagged.loc[common_idx],
        'VIX': vix.loc[common_idx]
    })
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    print(model.summary())

    coef = model.params['Real_Rate_Lag']
    pval = model.pvalues['Real_Rate_Lag']

    if coef < 0 and pval < 0.10:
        print("\n✓ Coefficient significant at 10%")
    else:
        print("\n△ Coefficient not significant")

    return model


def create_visualizations(subperiod_df, delta_ceq, real_rate_lagged, real_rate, model_simple):
    """Create all figures."""
    # Subperiod barplot
    fig, ax = plt.subplots(figsize=(10, 6))
    style_appendix_axes(fig, ax, y_grid_only=True)
    periods = ['Pre-2009', 'Post-2009']
    x = np.arange(len(periods))
    width = 0.35

    delta_mv = [subperiod_df.loc['Pre-2009', 'ΔCEQ (MV) [%]'],
                subperiod_df.loc['Post-2009', 'ΔCEQ (MV) [%]']]
    delta_minvar = [subperiod_df.loc['Pre-2009', 'ΔCEQ (MinVar) [%]'],
                    subperiod_df.loc['Post-2009', 'ΔCEQ (MinVar) [%]']]

    bars1 = ax.bar(x - width/2, delta_mv, width, label='Mean-Variance', color='#E74C3C')
    bars2 = ax.bar(x + width/2, delta_minvar, width, label='Minimum Variance', color='#3498DB')

    ax.set_ylabel('ΔCEQ (%)')
    ax.set_title('Welfare Costs by Period: 1/N Advantage')
    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.legend()

    for bar in bars1 + bars2:
        ax.annotate(f'{bar.get_height():.2f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}fig_subperiod_barplot.png', dpi=FIGURE_DPI,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    # LaTeX export (disabled for portable submission):
    # sync_to_latex(f'{OUTPUT_DIR}fig_subperiod_barplot.png', LATEX_FIG_APP_DIR)

    # Regression scatter
    fig, ax = plt.subplots(figsize=(10, 7))
    style_appendix_axes(fig, ax, y_grid_only=True)
    common_idx = delta_ceq.index.intersection(real_rate_lagged.index)
    y = delta_ceq.loc[common_idx]
    x_data = real_rate_lagged.loc[common_idx]

    split = pd.Timestamp(SPLIT_DATE)
    pre_mask = common_idx < split
    post_mask = common_idx >= split

    ax.scatter(x_data[pre_mask], y[pre_mask], alpha=0.6, c='#3498DB', label='Pre-2009', s=50)
    ax.scatter(x_data[post_mask], y[post_mask], alpha=0.6, c='#E74C3C', label='Post-2009', s=50)

    x_line = np.linspace(x_data.min(), x_data.max(), 100)
    y_line = model_simple.params.iloc[0] + model_simple.params.iloc[1] * x_line
    ax.plot(x_line, y_line, 'k-', linewidth=2, label='Regression')

    ax.set_xlabel('Real Rate (t-1)')
    ax.set_ylabel('ΔCEQ [%]')
    ax.set_title('Real Rate vs. Welfare Costs of Optimization')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}fig_regression_scatter.png', dpi=FIGURE_DPI,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    # LaTeX export (disabled for portable submission):
    # sync_to_latex(f'{OUTPUT_DIR}fig_regression_scatter.png', LATEX_FIG_APP_DIR)

    # Time series dual axis
    fig, ax1 = plt.subplots(figsize=(14, 6))
    common_idx = delta_ceq.index.intersection(real_rate.index)
    y1 = delta_ceq.loc[common_idx].rolling(12, min_periods=1).mean()
    y2 = real_rate.loc[common_idx].rolling(12, min_periods=1).mean()

    ax1.plot(y1.index, y1, color='#E74C3C', linewidth=1.5, label='ΔCEQ(MV)')
    ax1.set_ylabel('ΔCEQ (%, 12-month MA)', color='#E74C3C')
    ax1.tick_params(axis='y', labelcolor='#E74C3C')

    ax2 = ax1.twinx()
    ax2.plot(y2.index, y2, color='#3498DB', linewidth=1.5, label='Real Rate')
    ax2.set_ylabel('Real Rate (%, 12-month MA)', color='#3498DB')
    ax2.tick_params(axis='y', labelcolor='#3498DB')

    style_appendix_axes(fig, [ax1, ax2], y_grid_only=True)
    ax2.grid(False)
    ax1.axvspan(pd.Timestamp('2008-01-01'), pd.Timestamp('2015-12-01'), alpha=0.08, color='yellow')
    ax1.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2022-03-01'), alpha=0.08, color='yellow')
    ax1.set_title('Time Series: Welfare Costs vs. Real Interest Rate')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}fig_timeseries_dual.png', dpi=FIGURE_DPI,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    # LaTeX export (disabled for portable submission):
    # sync_to_latex(f'{OUTPUT_DIR}fig_timeseries_dual.png', LATEX_FIG_APP_DIR)

    print(f"Figures saved to {OUTPUT_DIR}")


def main():
    """Main execution."""
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*70)
    print("ROBUSTNESS ANALYSES")
    print("="*70)

    # Use local data (no network calls for main data — avoids hangs)
    data = pd.read_csv('./data/data_clean.csv', index_col=0, parse_dates=True)
    industries = INDUSTRIES
    returns_df = data[industries].copy()
    rf = data['RF_Monthly']

    # Real rate from local data
    real_rate = data['Real_Rate'] * 12 * 100  # annualised %
    real_rate_lagged = real_rate.shift(1).dropna()

    # VIX from local data (downloaded in 01_build_dataset.py)
    vix_df = None
    if 'VIX' in data.columns and data['VIX'].notna().sum() > 0:
        vix_df = data[['VIX']].dropna()
        print(f"VIX: {len(vix_df)} months from local data")
    else:
        print("Warning: VIX not available in local data")

    print(f"Data: {len(returns_df)} months")

    # Use cached baseline backtest if available, otherwise run fresh
    bt = load_cached_backtest(OUTPUT_DIR)
    if bt is None:
        print("  Cache not found, running baseline backtest...")
        bt = run_backtest(returns_df, rf, industries, ESTIMATION_WINDOW)
    else:
        print("  Loaded cached baseline backtest")

    ret_1n = bt['1/N']['returns']
    ret_mv = bt['Mean-Variance']['returns']
    ret_minvar = bt['Minimum Variance']['returns']
    hhi_mv = bt['Mean-Variance']['hhi']
    rf_aligned = rf.loc[ret_1n.index]

    # Calculate rolling ΔCEQ (12-month rolling window for proper mean/variance estimation)
    excess_1n = ret_1n - rf_aligned
    excess_mv = ret_mv - rf_aligned

    delta_ceq_list = []
    delta_ceq_dates = []
    roll_window = 12
    for i in range(roll_window, len(excess_1n)):
        e1n = excess_1n.iloc[i-roll_window:i]
        emv = excess_mv.iloc[i-roll_window:i]
        ceq_1n = (e1n.mean() - (RISK_AVERSION/2) * e1n.var()) * 12 * 100
        ceq_mv = (emv.mean() - (RISK_AVERSION/2) * emv.var()) * 12 * 100
        delta_ceq_list.append(ceq_1n - ceq_mv)
        delta_ceq_dates.append(excess_1n.index[i])

    delta_ceq = pd.Series(delta_ceq_list, index=delta_ceq_dates)
    delta_ceq.index = delta_ceq.index.to_period('M').to_timestamp()

    # Task 1 (subperiod - console output only; publication table is in 03_regime_comparison.py)
    returns_dict = {'1/N': ret_1n, 'Mean-Variance': ret_mv, 'Minimum Variance': ret_minvar}
    subperiod_df = task1_subperiod_comparison(returns_dict, rf, hhi_mv, SPLIT_DATE)

    # Task 2 (window robustness - console output only; publication table is in 08_robustness_tables.py)
    window_df = task2_window_robustness(returns_df, rf, industries, ESTIMATION_WINDOWS_SENSITIVITY)

    # Task 3
    model_simple = task3_regression_simple(delta_ceq, real_rate_lagged)
    with open(f'{OUTPUT_DIR}regression_simple.txt', 'w', encoding='utf-8') as f:
        f.write(model_simple.summary().as_text())

    # Task 4
    if vix_df is not None:
        vix = vix_df['VIX'].copy()
        vix.index = vix.index.to_period('M').to_timestamp()
        model_vix = task4_regression_with_vix(delta_ceq, real_rate_lagged, vix)
        with open(f'{OUTPUT_DIR}regression_with_vix.txt', 'w', encoding='utf-8') as f:
            f.write(model_vix.summary().as_text())

    # Visualizations
    create_visualizations(subperiod_df, delta_ceq, real_rate_lagged, real_rate, model_simple)

    print("\n" + "="*70)
    print("ROBUSTNESS ANALYSES COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
