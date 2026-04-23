#!/usr/bin/env python3
"""
08_robustness_tables.py - Publication-Quality Robustness Tables (3-5)
=====================================================================
Produces:
  Table 3: Estimation Window Sensitivity   (M = 60, 90, 120, 180)
  Table 4: Risk Aversion Sensitivity       (gamma = 1, 3, 5)
  Table 5: Transaction Cost Sensitivity    (TC = 0, 25, 50, 100 bps)

Prerequisites: Tables 1-2 must already exist in 03_Output/tables/.
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from config import (OUTPUT_DIR, ESTIMATION_WINDOW, RISK_AVERSION,
                    TRANSACTION_COST, INDUSTRIES,
                    ESTIMATION_WINDOWS_SENSITIVITY,
                    RISK_AVERSIONS_SENSITIVITY,
                    TRANSACTION_COSTS_SENSITIVITY)
from utils import (run_backtest, calculate_ceq, calculate_sharpe,
                   strategy_mean_variance, strategy_minimum_variance,
                   strategy_1n, calculate_drift_turnover,
                   latex_minus)

TABLE_DIR = f'{OUTPUT_DIR}tables/'
os.makedirs(TABLE_DIR, exist_ok=True)

# ── Formatting helpers (match Tables 1-2 style) ─────────────────────
def fmt2(v):
    if isinstance(v, float) and np.isnan(v):
        return '--'
    return latex_minus(f'{v:.2f}')

def fmt3(v):
    if isinstance(v, float) and np.isnan(v):
        return '--'
    return latex_minus(f'{v:.3f}')

STRAT_DISPLAY = {
    '1/N': r'$1/N$',
    'Mean-Variance': 'Mean--Variance',
    'Minimum Variance': 'Minimum--Variance',
}
STRATEGIES = ['1/N', 'Mean-Variance', 'Minimum Variance']

# ── Load local data (no network needed) ─────────────────────────────
print("=" * 70)
print("ROBUSTNESS TABLES 3-5")
print("=" * 70)

data = pd.read_csv('./data/data_clean.csv', index_col=0, parse_dates=True)
industries = INDUSTRIES
returns_df = data[industries].copy()
rf = data['RF_Monthly']

print(f"  Data: {len(data)} months, {len(industries)} industries")


# =====================================================================
#  TABLE 3 — Estimation Window Sensitivity
# =====================================================================
def build_table3():
    print("\n" + "=" * 70)
    print("TABLE 3: Estimation Window Sensitivity")
    print("=" * 70)

    windows = ESTIMATION_WINDOWS_SENSITIVITY
    rows = []

    for M in windows:
        print(f"  Running backtest M={M} ...", flush=True)
        bt = run_backtest(returns_df, rf, industries, estimation_window=M)

        ret = {s: bt[s]['returns']  for s in STRATEGIES}
        to  = {s: bt[s]['turnover'] for s in STRATEGIES}
        rf_a = rf.loc[ret['1/N'].index]

        oos_start = ret['1/N'].index[0].strftime('%Y-%m')
        oos_end   = ret['1/N'].index[-1].strftime('%Y-%m')
        n_oos     = len(ret['1/N'])

        ceq    = {s: calculate_ceq(ret[s], rf_a)    for s in STRATEGIES}
        sharpe = {s: calculate_sharpe(ret[s], rf_a)  for s in STRATEGIES}
        turnov = {s: to[s].mean() * 100              for s in STRATEGIES}

        row = {
            'M': M,
            'OOS': f'{oos_start} -- {oos_end}',
            'N': n_oos,
        }
        for s in STRATEGIES:
            tag = {'1/N': '1N', 'Mean-Variance': 'MV', 'Minimum Variance': 'MinVar'}[s]
            row[f'CEQ_{tag}']    = ceq[s]
            row[f'Sharpe_{tag}'] = sharpe[s]
            row[f'TO_{tag}']     = turnov[s]
        row['dCEQ_MV']     = ceq['1/N'] - ceq['Mean-Variance']
        row['dCEQ_MinVar'] = ceq['1/N'] - ceq['Minimum Variance']
        rows.append(row)

        print(f"    OOS {oos_start}-{oos_end} ({n_oos} mo), "
              f"dCEQ(MV)={row['dCEQ_MV']:+.2f}%, dCEQ(MinVar)={row['dCEQ_MinVar']:+.2f}%")

    df = pd.DataFrame(rows)

    # CSV
    df.to_csv(f'{TABLE_DIR}table3_window_robustness.csv', index=False)

    # LaTeX
    lines = []
    lines.append(r'\begin{table}[H]')
    lines.append(r'  \centering')
    lines.append(r'  \caption[Sensitivity to estimation window length]'
                 r'{Sensitivity to estimation window length. '
                 r'Each row reports out-of-sample results for a different '
                 r'rolling estimation window $M$. Baseline is $M=120$. '
                 r'Transaction cost = 50\,bps, $\gamma=' + str(RISK_AVERSION) + r'$.}')
    lines.append(r'  \label{tab:window_robust}')
    lines.append(r'  \addcontentsline{loa}{appendixentry}{\protect\numberline{\thetable}Sensitivity to estimation window length}')
    lines.append(r'  \small')
    lines.append(r'  \resizebox{\textwidth}{!}{%')
    lines.append(r'  \begin{tabular}{r l *{8}{r}}')
    lines.append(r'    \toprule')
    lines.append(r'    & & \multicolumn{3}{c}{CEQ (\%)} & \multicolumn{2}{c}{$\Delta$CEQ (\%)}'
                 r' & \multicolumn{3}{c}{Sharpe Ratio} \\')
    lines.append(r'    \cmidrule(lr){3-5} \cmidrule(lr){6-7} \cmidrule(lr){8-10}')
    lines.append(r'    $M$ & OOS Period & $1/N$ & MV & MinVar & MV & MinVar & $1/N$ & MV & MinVar \\')
    lines.append(r'    \midrule')

    for _, r in df.iterrows():
        is_base = (r['M'] == ESTIMATION_WINDOW)
        prefix = r'    ' + (r'\textbf{' if is_base else '')
        suffix = (r'}' if is_base else '')
        cells = [
            f"{prefix}{int(r['M'])}{suffix}",
            f"{r['OOS']}",
            fmt2(r['CEQ_1N']),
            fmt2(r['CEQ_MV']),
            fmt2(r['CEQ_MinVar']),
            fmt2(r['dCEQ_MV']),
            fmt2(r['dCEQ_MinVar']),
            fmt3(r['Sharpe_1N']),
            fmt3(r['Sharpe_MV']),
            fmt3(r['Sharpe_MinVar']),
        ]
        lines.append('    ' + ' & '.join(cells) + r' \\')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}%')
    lines.append(r'  }')
    lines.append('')
    lines.append(r'  \smallskip')
    lines.append(r'  \begin{minipage}{\textwidth}')
    lines.append(r'    \footnotesize')
    lines.append(r'    \textit{Notes.} '
                 r'CEQ and Sharpe are annualised and computed on excess returns. '
                 r'$\Delta\text{CEQ} = \text{CEQ}_{1/N} - \text{CEQ}_{\text{strategy}}$; '
                 r'positive values indicate $1/N$ outperformance. '
                 r'Different $M$ values imply different OOS start dates. '
                 r'Baseline $M=120$ is shown in bold.')
    lines.append(r"    \textit{Source:} Author's calculations based on data from"
                 r" Kenneth R.\ French's Data Library.")
    lines.append(r'  \end{minipage}')
    lines.append(r'\end{table}')
    tex = '\n'.join(lines) + '\n'

    tex = tex.replace('\xa0', ' ')
    with open(f'{TABLE_DIR}table3_window_robustness.tex', 'w', encoding='utf-8') as f:
        f.write(tex)
    print(f"\n  Saved: table3_window_robustness.tex / .csv")
    # LaTeX export (disabled for portable submission):
    # sync_to_latex(f'{TABLE_DIR}table3_window_robustness.tex', LATEX_TABLES_DIR)
    print(tex)
    return df


# =====================================================================
#  TABLE 4 — Risk Aversion Sensitivity
# =====================================================================
def build_table4():
    print("\n" + "=" * 70)
    print("TABLE 4: Risk Aversion Sensitivity")
    print("=" * 70)

    gammas = RISK_AVERSIONS_SENSITIVITY
    rows = []

    for gamma in gammas:
        print(f"  Running backtest gamma={gamma} ...", flush=True)

        # Must re-optimise MV for each gamma (gamma enters objective).
        # MinVar is gamma-independent (minimise variance), but CEQ evaluation changes.
        n_assets = len(industries)
        T = len(returns_df)
        c = TRANSACTION_COST

        ret = {s: [] for s in STRATEGIES}
        to  = {s: [] for s in STRATEGIES}
        w_prev = {s: np.zeros(n_assets) for s in STRATEGIES}
        dates = []

        for t in range(ESTIMATION_WINDOW, T):
            hist = returns_df[industries].iloc[t - ESTIMATION_WINDOW:t]
            mu = hist.mean().values
            sigma = hist.cov().values
            r_t = returns_df[industries].iloc[t].values
            r_prev = returns_df[industries].iloc[t - 1].values if t > ESTIMATION_WINDOW else r_t
            dates.append(returns_df.index[t])

            w_1n     = strategy_1n(n_assets)
            w_mv     = strategy_mean_variance(mu, sigma, gamma=gamma)
            w_minvar = strategy_minimum_variance(sigma)

            for s, w in [('1/N', w_1n), ('Mean-Variance', w_mv), ('Minimum Variance', w_minvar)]:
                turnover_t = calculate_drift_turnover(w_prev[s], w, r_prev)
                ret[s].append(np.dot(w, r_t) - c * turnover_t)
                to[s].append(turnover_t)
                w_prev[s] = w

        for s in STRATEGIES:
            ret[s] = pd.Series(ret[s], index=dates)
            to[s]  = pd.Series(to[s], index=dates)

        rf_a = rf.loc[ret['1/N'].index]

        ceq    = {s: calculate_ceq(ret[s], rf_a, gamma=gamma)   for s in STRATEGIES}
        sharpe = {s: calculate_sharpe(ret[s], rf_a)              for s in STRATEGIES}
        turnov = {s: to[s].mean() * 100                          for s in STRATEGIES}

        row = {'gamma': gamma}
        for s in STRATEGIES:
            tag = {'1/N': '1N', 'Mean-Variance': 'MV', 'Minimum Variance': 'MinVar'}[s]
            row[f'CEQ_{tag}']    = ceq[s]
            row[f'Sharpe_{tag}'] = sharpe[s]
            row[f'TO_{tag}']     = turnov[s]
        row['dCEQ_MV']     = ceq['1/N'] - ceq['Mean-Variance']
        row['dCEQ_MinVar'] = ceq['1/N'] - ceq['Minimum Variance']
        rows.append(row)

        print(f"    CEQ(1/N)={ceq['1/N']:.2f}, dCEQ(MV)={row['dCEQ_MV']:+.2f}%, "
              f"dCEQ(MinVar)={row['dCEQ_MinVar']:+.2f}%")

    df = pd.DataFrame(rows)
    df.to_csv(f'{TABLE_DIR}table4_gamma_sensitivity.csv', index=False)

    # LaTeX
    lines = []
    lines.append(r'\begin{table}[H]')
    lines.append(r'  \centering')
    lines.append(r'  \caption[Sensitivity to risk aversion]'
                 r'{Sensitivity to risk aversion. '
                 r'Each row re-optimises the mean--variance portfolio for a different '
                 r'$\gamma$ and re-evaluates CEQ accordingly. '
                 r'Minimum--variance weights are $\gamma$-independent but CEQ changes. '
                 r'$M=' + str(ESTIMATION_WINDOW) + r'$, TC = 50\,bps.}')
    lines.append(r'  \label{tab:gamma_robust}')
    lines.append(r'  \addcontentsline{loa}{appendixentry}{\protect\numberline{\thetable}Sensitivity to risk aversion}')
    lines.append(r'  \small')
    lines.append(r'  \begin{tabular}{r *{8}{r}}')
    lines.append(r'    \toprule')
    lines.append(r'    & \multicolumn{3}{c}{CEQ (\%)} & \multicolumn{2}{c}{$\Delta$CEQ (\%)}'
                 r' & \multicolumn{3}{c}{Turnover (\%)} \\')
    lines.append(r'    \cmidrule(lr){2-4} \cmidrule(lr){5-6} \cmidrule(lr){7-9}')
    lines.append(r'    $\gamma$ & $1/N$ & MV & MinVar & MV & MinVar & $1/N$ & MV & MinVar \\')
    lines.append(r'    \midrule')

    for _, r in df.iterrows():
        is_base = (r['gamma'] == RISK_AVERSION)
        prefix = r'\textbf{' if is_base else ''
        suffix = r'}' if is_base else ''
        cells = [
            f'{prefix}{int(r["gamma"])}{suffix}',
            fmt2(r['CEQ_1N']),
            fmt2(r['CEQ_MV']),
            fmt2(r['CEQ_MinVar']),
            fmt2(r['dCEQ_MV']),
            fmt2(r['dCEQ_MinVar']),
            fmt2(r['TO_1N']),
            fmt2(r['TO_MV']),
            fmt2(r['TO_MinVar']),
        ]
        lines.append('    ' + ' & '.join(cells) + r' \\')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append('')
    lines.append(r'  \smallskip')
    lines.append(r'  \begin{minipage}{\textwidth}')
    lines.append(r'    \footnotesize')
    lines.append(r'    \textit{Notes.} '
                 r'All values are annualised. '
                 r'Higher $\gamma$ penalises variance more heavily, '
                 r'reducing CEQ for all strategies. '
                 r'Turnover is the monthly average. '
                 r'Baseline $\gamma=1$ in bold.')
    lines.append(r"    \textit{Source:} Author's calculations based on data from"
                 r" Kenneth R.\ French's Data Library.")
    lines.append(r'  \end{minipage}')
    lines.append(r'\end{table}')
    tex = '\n'.join(lines) + '\n'

    tex = tex.replace('\xa0', ' ')
    with open(f'{TABLE_DIR}table4_gamma_sensitivity.tex', 'w', encoding='utf-8') as f:
        f.write(tex)
    print(f"\n  Saved: table4_gamma_sensitivity.tex / .csv")
    # LaTeX export (disabled for portable submission):
    # sync_to_latex(f'{TABLE_DIR}table4_gamma_sensitivity.tex', LATEX_TABLES_DIR)
    print(tex)
    return df


# =====================================================================
#  TABLE 5 — Transaction Cost Sensitivity
# =====================================================================
def build_table5():
    print("\n" + "=" * 70)
    print("TABLE 5: Transaction Cost Sensitivity")
    print("=" * 70)

    tc_levels = TRANSACTION_COSTS_SENSITIVITY

    # Run ONE backtest at TC=0 to get gross returns and turnover series.
    # Then net returns for any TC = gross - TC * turnover.
    print("  Running baseline backtest (TC=0) ...", flush=True)
    bt = run_backtest(returns_df, rf, industries,
                      estimation_window=ESTIMATION_WINDOW,
                      transaction_cost=0.0)

    gross_ret = {s: bt[s]['returns']  for s in STRATEGIES}
    to_series = {s: bt[s]['turnover'] for s in STRATEGIES}
    rf_a = rf.loc[gross_ret['1/N'].index]

    rows = []
    for c in tc_levels:
        bps = int(c * 10000)
        print(f"  TC = {bps} bps")

        net_ret = {}
        for s in STRATEGIES:
            net_ret[s] = gross_ret[s] - c * to_series[s]

        ceq    = {s: calculate_ceq(net_ret[s], rf_a)   for s in STRATEGIES}
        sharpe = {s: calculate_sharpe(net_ret[s], rf_a) for s in STRATEGIES}

        row = {'TC_bps': bps}
        for s in STRATEGIES:
            tag = {'1/N': '1N', 'Mean-Variance': 'MV', 'Minimum Variance': 'MinVar'}[s]
            row[f'CEQ_{tag}']    = ceq[s]
            row[f'Sharpe_{tag}'] = sharpe[s]
        row['dCEQ_MV']     = ceq['1/N'] - ceq['Mean-Variance']
        row['dCEQ_MinVar'] = ceq['1/N'] - ceq['Minimum Variance']
        rows.append(row)

        print(f"    CEQ(1/N)={ceq['1/N']:.2f}, dCEQ(MV)={row['dCEQ_MV']:+.2f}%, "
              f"dCEQ(MinVar)={row['dCEQ_MinVar']:+.2f}%")

    df = pd.DataFrame(rows)
    df.to_csv(f'{TABLE_DIR}table5_tc_sensitivity.csv', index=False)

    # LaTeX
    lines = []
    lines.append(r'\begin{table}[H]')
    lines.append(r'  \centering')
    lines.append(r'  \caption[Sensitivity to transaction costs]'
                 r'{Sensitivity to transaction costs. '
                 r'Net returns are computed as $r_{p,t}^{net} = r_{p,t}^{gross} - c \cdot TO_{t}$ for each cost level $c$. '
                 r'$M=' + str(ESTIMATION_WINDOW) + r'$, $\gamma=' + str(RISK_AVERSION) + r'$.}')
    lines.append(r'  \label{tab:tc_robust}')
    lines.append(r'  \addcontentsline{loa}{appendixentry}{\protect\numberline{\thetable}Sensitivity to transaction costs}')
    lines.append(r'  \small')
    lines.append(r'  \begin{tabular}{r *{8}{r}}')
    lines.append(r'    \toprule')
    lines.append(r'    & \multicolumn{3}{c}{CEQ (\%)} & \multicolumn{2}{c}{$\Delta$CEQ (\%)}'
                 r' & \multicolumn{3}{c}{Sharpe Ratio} \\')
    lines.append(r'    \cmidrule(lr){2-4} \cmidrule(lr){5-6} \cmidrule(lr){7-9}')
    lines.append(r'    TC (bps) & $1/N$ & MV & MinVar & MV & MinVar & $1/N$ & MV & MinVar \\')
    lines.append(r'    \midrule')

    for _, r in df.iterrows():
        is_base = (r['TC_bps'] == int(TRANSACTION_COST * 10000))
        prefix = r'\textbf{' if is_base else ''
        suffix = r'}' if is_base else ''
        cells = [
            f'{prefix}{int(r["TC_bps"])}{suffix}',
            fmt2(r['CEQ_1N']),
            fmt2(r['CEQ_MV']),
            fmt2(r['CEQ_MinVar']),
            fmt2(r['dCEQ_MV']),
            fmt2(r['dCEQ_MinVar']),
            fmt3(r['Sharpe_1N']),
            fmt3(r['Sharpe_MV']),
            fmt3(r['Sharpe_MinVar']),
        ]
        lines.append('    ' + ' & '.join(cells) + r' \\')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append('')
    lines.append(r'  \smallskip')
    lines.append(r'  \begin{minipage}{\textwidth}')
    lines.append(r'    \footnotesize')
    lines.append(r'    \textit{Notes.} '
                 r'All values are annualised. '
                 r'At TC = 0\,bps gross returns are used; higher TC levels '
                 r'increasingly penalise high-turnover strategies. '
                 r'Baseline TC = 50\,bps is shown in bold. '
                 r'$\Delta\text{CEQ} = \text{CEQ}_{1/N} - \text{CEQ}_{\text{strategy}}$.')
    lines.append(r"    \textit{Source:} Author's calculations based on data from"
                 r" Kenneth R.\ French's Data Library.")
    lines.append(r'  \end{minipage}')
    lines.append(r'\end{table}')
    tex = '\n'.join(lines) + '\n'

    tex = tex.replace('\xa0', ' ')
    with open(f'{TABLE_DIR}table5_tc_sensitivity.tex', 'w', encoding='utf-8') as f:
        f.write(tex)
    print(f"\n  Saved: table5_tc_sensitivity.tex / .csv")
    # LaTeX export (disabled for portable submission):
    # sync_to_latex(f'{TABLE_DIR}table5_tc_sensitivity.tex', LATEX_TABLES_DIR)
    print(tex)
    return df


# =====================================================================
#  Main
# =====================================================================
if __name__ == '__main__':
    # Verify Tables 1-2 exist
    for req in ['table1_full_sample.tex', 'table2_subperiod.tex']:
        if not os.path.exists(f'{TABLE_DIR}{req}'):
            print(f"ERROR: {req} not found. Run 10_export_latex.py and "
                  f"03_regime_comparison.py first.")
            exit(1)
    print("  Tables 1-2 verified.")

    t3 = build_table3()
    t4 = build_table4()
    t5 = build_table5()

    print("\n" + "=" * 70)
    print("ALL ROBUSTNESS TABLES COMPLETE (3, 4, 5)")
    print("=" * 70)

