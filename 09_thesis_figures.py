#!/usr/bin/env python3
"""
09_thesis_figures.py - Publication-Quality Figures (1-5)
=========================================================
Generates all thesis figures in a clean, academic style.
Requires: strategy_returns.csv, data_clean.csv from prior scripts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

from config import (OUTPUT_DIR, ESTIMATION_WINDOW, RISK_AVERSION,
                    TRANSACTION_COST, INDUSTRIES, FIGURE_DPI)
from utils import run_backtest, load_cached_backtest

# ── Output directory ────────────────────────────────────────────────
FIG_DIR = f'{OUTPUT_DIR}figures/'
os.makedirs(FIG_DIR, exist_ok=True)

# ── Global matplotlib style: clean academic ─────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.size':          11,
    'axes.titlesize':     13,
    'axes.labelsize':     12,
    'legend.fontsize':    10,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.15,
    'grid.linestyle':     '--',
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'savefig.facecolor':  'white',
    'savefig.bbox':       'tight',
    'savefig.dpi':        FIGURE_DPI,
})

SPLIT = pd.Timestamp('2009-01-01')
SPLIT_LABEL = SPLIT.strftime('%b %Y')

# Strategy colours (colourblind-safe palette)
C_1N     = '#2E86AB'   # blue
C_MV     = '#A23B72'   # magenta
C_MINVAR = '#F18F01'   # orange

# ── Load data ───────────────────────────────────────────────────────
print("=" * 70)
print("GENERATING THESIS FIGURES 1-5")
print("=" * 70)

returns = pd.read_csv(f'{OUTPUT_DIR}strategy_returns.csv',
                      index_col=0, parse_dates=True)
data    = pd.read_csv('./data/data_clean.csv',
                      index_col=0, parse_dates=True)
rf      = data['RF_Monthly']
industries = INDUSTRIES

print(f"  Returns: {len(returns)} months "
      f"({returns.index[0].strftime('%Y-%m')} to "
      f"{returns.index[-1].strftime('%Y-%m')})")


# =====================================================================
# FIGURE 1 — Cumulative Wealth (log scale)
# =====================================================================
def figure1():
    print("\n  [1/5] Cumulative wealth ...")
    wealth = (1 + returns).cumprod()          # $1 invested

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.plot(wealth.index, wealth['1/N'],
            lw=2.2, color=C_1N, label='$1/N$', zorder=3)
    ax.plot(wealth.index, wealth['Mean-Variance'],
            lw=2.2, color=C_MV, ls='--', label='Mean-Variance', zorder=2)
    ax.plot(wealth.index, wealth['Minimum Variance'],
            lw=2.2, color=C_MINVAR, ls='-.', label='Minimum-Variance', zorder=2)

    # Crisis line
    ax.axvline(SPLIT, color='grey', ls=':', lw=1.2, zorder=1)
    ypos = ax.get_ylim()[1] * 0.92
    ax.text(SPLIT, ypos, f'  {SPLIT_LABEL}', va='top', fontsize=9, color='grey')

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f'${y:,.0f}' if y >= 1 else f'${y:.2f}'))
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (\\$, log scale)')
    ax.set_title('Cumulative Wealth: \\$1 Invested at Start of Out-of-Sample Period')
    ax.legend(loc='upper left', frameon=True, edgecolor='grey')

    # Annotate final values
    for col, color in [('1/N', C_1N),
                       ('Mean-Variance', C_MV),
                       ('Minimum Variance', C_MINVAR)]:
        final = wealth[col].iloc[-1]
        ax.annotate(f'${final:,.0f}',
                    xy=(wealth.index[-1], final),
                    xytext=(6, 0), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=color, va='center')

    fig.tight_layout()
    path = f'{FIG_DIR}fig1_cumulative_wealth.png'
    fig.savefig(path, facecolor='white')
    plt.close(fig)
    print(f"    Saved: {path}")
    # LaTeX export (disabled for portable submission):
    # sync_to_latex(path, LATEX_FIGURES_DIR)


# =====================================================================
# FIGURES 2-4 need per-period weights → run one backtest
# =====================================================================
def get_weights_simple():
    """Load cached backtest weights, or run backtest if cache is missing."""
    cached = load_cached_backtest(OUTPUT_DIR)
    if cached is not None:
        print("\n  Loading cached weights ...", flush=True)
        w_mv     = cached['Mean-Variance']['weights']
        w_minvar = cached['Minimum Variance']['weights']
        hhi_mv     = cached['Mean-Variance']['hhi']
        hhi_minvar = cached['Minimum Variance']['hhi']
    else:
        print("\n  Computing weights (one backtest) ...", flush=True)
        bt = run_backtest(data, rf, industries,
                          estimation_window=ESTIMATION_WINDOW,
                          transaction_cost=TRANSACTION_COST)
        w_mv     = bt['Mean-Variance']['weights']
        w_minvar = bt['Minimum Variance']['weights']
        hhi_mv     = bt['Mean-Variance']['hhi']
        hhi_minvar = bt['Minimum Variance']['hhi']
    print(f"    Weights loaded: {len(w_mv)} months")
    return w_mv, w_minvar, hhi_mv, hhi_minvar


# =====================================================================
# FIGURE 2 — MV Weight Evolution (stacked area)
# =====================================================================
def figure2(w_mv):
    print("\n  [2/5] MV weight evolution ...")
    fig, ax = plt.subplots(figsize=(12, 5.5))

    ax.stackplot(w_mv.index,
                 *[w_mv[ind] for ind in industries],
                 labels=industries, alpha=0.85)
    ax.axvline(SPLIT, color='black', ls=':', lw=1, zorder=5)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Weight')
    ax.set_title('Mean-Variance Portfolio Weights Over Time')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
              ncol=1, fontsize=9, frameon=False)

    fig.tight_layout()
    path = f'{FIG_DIR}fig2_mv_weights.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"    Saved: {path}")
    # LaTeX export (disabled for portable submission):
    # sync_to_latex(path, LATEX_FIGURES_DIR)


# =====================================================================
# FIGURE 3 — MinVar Weight Evolution (stacked area)
# =====================================================================
def figure3(w_minvar):
    print("\n  [3/5] MinVar weight evolution ...")
    fig, ax = plt.subplots(figsize=(12, 5.5))

    ax.stackplot(w_minvar.index,
                 *[w_minvar[ind] for ind in industries],
                 labels=industries, alpha=0.85)
    ax.axvline(SPLIT, color='black', ls=':', lw=1, zorder=5)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Weight')
    ax.set_title('Minimum-Variance Portfolio Weights Over Time')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
              ncol=1, fontsize=9, frameon=False)

    fig.tight_layout()
    path = f'{FIG_DIR}fig3_minvar_weights.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"    Saved: {path}")
    # LaTeX export (disabled for portable submission):
    # sync_to_latex(path, LATEX_FIGURES_DIR)


# =====================================================================
# FIGURE 4 — HHI Concentration Over Time
# =====================================================================
def figure4(hhi_mv, hhi_minvar):
    print("\n  [4/5] HHI concentration ...")
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    n = len(industries)
    hhi_1n = 1.0 / n    # constant = 0.10

    ax.plot(hhi_mv.index, hhi_mv, lw=1.6, color=C_MV,
            label='Mean-Variance', alpha=0.9)
    ax.plot(hhi_minvar.index, hhi_minvar, lw=1.6, color=C_MINVAR,
            label='Minimum-Variance', alpha=0.9)
    ax.axhline(hhi_1n, color=C_1N, ls='-', lw=2, label=f'$1/N$ (HHI = {hhi_1n:.2f})')

    ax.axvline(SPLIT, color='grey', ls=':', lw=1)
    ax.text(SPLIT, ax.get_ylim()[1] * 0.95, f'  {SPLIT_LABEL}',
            va='top', fontsize=9, color='grey')
    # Keep only subtle horizontal guides for readability.
    ax.grid(True, axis='y', alpha=0.10, linestyle='--')
    ax.grid(False, axis='x')

    ax.set_xlabel('Date')
    ax.set_ylabel('Herfindahl-Hirschman Index (HHI)')
    ax.set_title('Portfolio Concentration Over Time')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', frameon=True, edgecolor='grey')

    fig.tight_layout()
    path = f'{FIG_DIR}fig4_hhi_timeseries.png'
    fig.savefig(path, facecolor='white')
    plt.close(fig)
    print(f"    Saved: {path}")
    # LaTeX export (disabled for portable submission):
    # sync_to_latex(path, LATEX_FIGURES_DIR)


# =====================================================================
# FIGURE 5 — Interest Rate Environment
# =====================================================================
def figure5():
    print("\n  [5/5] Interest rate environment ...")

    # Use full dataset (starts 1963) for context
    rf_ann = data['RF_Monthly'] * 12 * 100      # annualised %

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.plot(data.index, rf_ann, lw=1.2, color='#2c3e50', label='1-Month T-Bill (ann.)')

    # Shade the post-2009 low-rate region
    ax.axvspan(SPLIT, data.index[-1], alpha=0.06, color='red',
               label='Post-2009 low-rate era')
    ax.axvline(SPLIT, color='grey', ls=':', lw=1)

    # OOS start
    oos_start = returns.index[0]
    ax.axvline(oos_start, color=C_1N, ls='--', lw=1,
               label=f'OOS start ({oos_start.strftime("%Y-%m")})')

    ax.set_xlabel('Date')
    ax.set_ylabel('Annualised Rate (%)')
    ax.set_title('1-Month Treasury Bill Rate')
    ax.legend(loc='upper right', frameon=True, edgecolor='grey')
    ax.set_ylim(bottom=-0.5)
    # Minimal grid to avoid heavy background styling.
    ax.grid(True, axis='y', alpha=0.10, linestyle='--')
    ax.grid(False, axis='x')

    fig.tight_layout()
    path = f'{FIG_DIR}fig5_interest_rates.png'
    fig.savefig(path, facecolor='white')
    plt.close(fig)
    print(f"    Saved: {path}")
    # LaTeX export (disabled for portable submission):
    # sync_to_latex(path, LATEX_FIGURES_DIR)


# =====================================================================
#  MAIN
# =====================================================================
if __name__ == '__main__':
    figure1()

    w_mv, w_minvar, hhi_mv, hhi_minvar = get_weights_simple()
    figure2(w_mv)
    figure3(w_minvar)
    figure4(hhi_mv, hhi_minvar)

    figure5()

    # Verify
    created = sorted(os.listdir(FIG_DIR))
    print("\n" + "=" * 70)
    print(f"ALL FIGURES COMPLETE ({len(created)} files):")
    for f in created:
        sz = os.path.getsize(os.path.join(FIG_DIR, f))
        print(f"  {sz:>8} bytes  {f}")
    print("=" * 70)

