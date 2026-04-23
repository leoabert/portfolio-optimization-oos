"""
utils.py - Shared Functions for Bachelor Thesis
================================================
Contains optimization and metric functions used across all scripts.
"""

import os
import shutil
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Import config
try:
    from config import RISK_AVERSION, TRANSACTION_COST, INDUSTRIES
except ImportError:
    RISK_AVERSION = 1
    TRANSACTION_COST = 0.0050
    INDUSTRIES = ['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec',
                  'Telcm', 'Shops', 'Hlth', 'Utils', 'Other']


# =============================================================================
# OPTIMIZATION STRATEGIES
# =============================================================================

def strategy_1n(n_assets):
    """Return equal weights."""
    return np.array([1/n_assets] * n_assets)


def strategy_mean_variance(mu, sigma, gamma=None):
    """
    Constrained mean-variance optimization (long-only).

    Solves:  max  w'mu - (gamma/2) w'Sigma w
             s.t. w'1 = 1,  w_i >= 0

    Parameters
    ----------
    mu : np.ndarray (N,)        Sample mean return vector.
    sigma : np.ndarray (N, N)   Sample covariance matrix.
    gamma : float, optional     Risk aversion coefficient (default: config.RISK_AVERSION).

    Returns
    -------
    np.ndarray (N,)  Optimal portfolio weights (falls back to 1/N on failure).
    """
    if gamma is None:
        gamma = RISK_AVERSION

    n = len(mu)

    def objective(w):
        return -((w @ mu) - (gamma/2) * (w @ sigma @ w))

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    x0 = np.array([1/n] * n)

    result = minimize(objective, x0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'ftol': 1e-9, 'maxiter': 1000})

    return result.x if result.success else x0


def strategy_minimum_variance(sigma):
    """
    Constrained minimum-variance optimization (long-only).

    Solves:  min  w'Sigma w
             s.t. w'1 = 1,  w_i >= 0

    Parameters
    ----------
    sigma : np.ndarray (N, N)   Sample covariance matrix.

    Returns
    -------
    np.ndarray (N,)  Optimal portfolio weights (falls back to 1/N on failure).
    """
    n = sigma.shape[0]

    def objective(w):
        return w @ sigma @ w

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    x0 = np.array([1/n] * n)

    result = minimize(objective, x0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'ftol': 1e-9, 'maxiter': 1000})

    return result.x if result.success else x0


# =============================================================================
# TURNOVER CALCULATION
# =============================================================================

def calculate_drift_turnover(w_prev, w_target, returns):
    """
    Turnover accounting for portfolio drift, per DeMiguel et al. (2009) Eq. 15.

    Weights drift as assets earn different returns between rebalancing dates:
        w_drift = w_prev * (1+r) / sum(w_prev * (1+r))
        turnover = sum(|w_target - w_drift|)

    Parameters
    ----------
    w_prev : np.ndarray (N,)    Weights at the start of the previous period.
    w_target : np.ndarray (N,)  Target weights for the current period.
    returns : np.ndarray (N,)   Asset returns during the previous period.

    Returns
    -------
    float  Total turnover (sum of absolute weight changes).
    """
    if np.sum(w_prev) == 0:
        return 0.0  # First period: no prior portfolio to rebalance from (DeMiguel convention)

    w_drifted = w_prev * (1 + returns)
    w_drifted = w_drifted / np.sum(w_drifted)

    return np.sum(np.abs(w_target - w_drifted))


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def calculate_ceq(returns, rf, gamma=None):
    """
    Annualized Certainty Equivalent Return from out-of-sample excess returns.

    CEQ_monthly = E[r_excess] - (gamma/2) * Var(r_excess)
    CEQ_annual  = CEQ_monthly * 12 * 100   (percentage points)

    See DeMiguel et al. (2009, RFS), Section II.A.

    Parameters
    ----------
    returns : pd.Series   Monthly portfolio returns (total, not excess).
    rf : pd.Series        Monthly risk-free rate (aligned by date index).
    gamma : float, opt.   Risk aversion (default: config.RISK_AVERSION).

    Returns
    -------
    float  Annualized CEQ in percentage points.
    """
    if gamma is None:
        gamma = RISK_AVERSION

    rf_aligned = rf.loc[returns.index]
    excess_returns = returns - rf_aligned

    ceq_monthly = excess_returns.mean() - (gamma/2) * excess_returns.var()
    ceq_annual = ceq_monthly * 12 * 100  # Convert to percentage

    return ceq_annual


def calculate_sharpe(returns, rf):
    """Calculate annualized Sharpe ratio."""
    rf_aligned = rf.loc[returns.index]
    excess = returns - rf_aligned
    std_val = excess.std()
    if std_val == 0:
        return 0
    return (excess.mean() / std_val) * np.sqrt(12)


def calculate_hhi(weights):
    """
    Calculate Herfindahl-Hirschman Index (concentration).
    HHI = Σw_i^2
    """
    return np.sum(weights ** 2)


def calculate_metrics(returns, rf, turnover, gamma=None):
    """
    Calculate all performance metrics for a strategy.
    Returns dict with Mean Return, Volatility, Sharpe, CEQ, Turnover.
    """
    if gamma is None:
        gamma = RISK_AVERSION

    rf_aligned = rf.loc[returns.index]
    excess = returns - rf_aligned

    std_val = excess.std()
    sharpe = (excess.mean() / std_val) * np.sqrt(12) if std_val > 0 else 0
    ceq = (excess.mean() - (gamma/2) * excess.var()) * 12 * 100

    return {
        'Mean Return (%)': returns.mean() * 12 * 100,
        'Volatility (%)': excess.std() * np.sqrt(12) * 100,
        'Sharpe Ratio': sharpe,
        'CEQ (%)': ceq,
        'Turnover (%)': turnover.mean() * 100
    }


def latex_minus(s):
    """Replace leading text hyphen with LaTeX math minus for table cells."""
    if isinstance(s, str) and s.startswith('-'):
        return '$-$' + s[1:]
    return s


def calculate_beta(portfolio_ret, market_ret):
    """Calculate portfolio beta."""
    covariance = np.cov(portfolio_ret, market_ret)[0, 1]
    market_variance = np.var(market_ret, ddof=1)
    return covariance / market_variance if market_variance > 0 else np.nan


# =============================================================================
# DATA LOADING
# =============================================================================

def load_fama_french_data(start_year=1963, end_year=2025):
    """Load Fama-French 10 Industry Portfolios and Risk-Free Rate."""
    import pandas_datareader.data as web
    from datetime import datetime

    start = datetime(start_year, 7, 1)
    end = datetime(end_year, 12, 31)

    print("Downloading Fama-French 10 Industry Portfolios...")
    ff_industries = web.DataReader('10_Industry_Portfolios', 'famafrench', start, end)[0]
    ff_industries.index = ff_industries.index.to_timestamp()
    ff_industries = ff_industries / 100.0

    print("Downloading Risk-Free Rate...")
    ff_factors = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start, end)[0]
    ff_factors.index = ff_factors.index.to_timestamp()
    rf = ff_factors['RF'] / 100.0

    return ff_industries, rf


def load_vix_data():
    """Download VIX from FRED (with 30s timeout)."""
    import pandas_datareader as pdr
    import concurrent.futures

    print("Downloading VIX from FRED...")
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                pdr.get_data_fred, 'VIXCLS', start='1990-01-01', end='2025-12-31'
            )
            vix = future.result(timeout=30)
        vix_monthly = vix.resample('MS').last()
        vix_monthly.columns = ['VIX']
        return vix_monthly
    except concurrent.futures.TimeoutError:
        print("Warning: VIX download timed out after 30s — skipping")
        return None
    except Exception as e:
        print(f"Warning: Could not download VIX: {e}")
        return None


def load_local_data(filepath='./data/data_clean.csv'):
    """Load pre-built local dataset."""
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return data


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(returns_df, rf, industries, estimation_window=120, transaction_cost=None):
    """
    Run complete backtest for all three strategies.

    Returns dict with strategy returns, weights, turnover, and HHI.
    """
    if transaction_cost is None:
        transaction_cost = TRANSACTION_COST

    n_assets = len(industries)
    T = len(returns_df)

    results = {
        '1/N': {'returns': [], 'weights': [], 'turnover': [], 'hhi': []},
        'Mean-Variance': {'returns': [], 'weights': [], 'turnover': [], 'hhi': []},
        'Minimum Variance': {'returns': [], 'weights': [], 'turnover': [], 'hhi': []}
    }

    dates = []
    w_prev = {s: np.zeros(n_assets) for s in results.keys()}

    for t in range(estimation_window, T):
        hist = returns_df[industries].iloc[t-estimation_window:t]
        # pandas .var() and .cov() use ddof=1 by default, matching DeMiguel Eq. (2)-(3)
        mu = hist.mean().values
        sigma = hist.cov().values
        r_t = returns_df[industries].iloc[t].values
        # Previous period's return for drift calculation
        # t >= estimation_window >= 60, so t-1 is always a valid index
        r_prev = returns_df[industries].iloc[t-1].values
        date_t = returns_df.index[t]
        dates.append(date_t)

        # 1/N
        w_1n = strategy_1n(n_assets)
        to_1n = calculate_drift_turnover(w_prev['1/N'], w_1n, r_prev)
        results['1/N']['returns'].append(np.dot(w_1n, r_t) - transaction_cost * to_1n)
        results['1/N']['weights'].append(w_1n)
        results['1/N']['turnover'].append(to_1n)
        results['1/N']['hhi'].append(calculate_hhi(w_1n))
        w_prev['1/N'] = w_1n

        # Mean-Variance
        w_mv = strategy_mean_variance(mu, sigma)
        to_mv = calculate_drift_turnover(w_prev['Mean-Variance'], w_mv, r_prev)
        results['Mean-Variance']['returns'].append(np.dot(w_mv, r_t) - transaction_cost * to_mv)
        results['Mean-Variance']['weights'].append(w_mv)
        results['Mean-Variance']['turnover'].append(to_mv)
        results['Mean-Variance']['hhi'].append(calculate_hhi(w_mv))
        w_prev['Mean-Variance'] = w_mv

        # Minimum Variance
        w_minvar = strategy_minimum_variance(sigma)
        to_minvar = calculate_drift_turnover(w_prev['Minimum Variance'], w_minvar, r_prev)
        results['Minimum Variance']['returns'].append(np.dot(w_minvar, r_t) - transaction_cost * to_minvar)
        results['Minimum Variance']['weights'].append(w_minvar)
        results['Minimum Variance']['turnover'].append(to_minvar)
        results['Minimum Variance']['hhi'].append(calculate_hhi(w_minvar))
        w_prev['Minimum Variance'] = w_minvar

    # Convert to DataFrames/Series
    for strategy in results.keys():
        results[strategy]['returns'] = pd.Series(results[strategy]['returns'], index=dates)
        results[strategy]['turnover'] = pd.Series(results[strategy]['turnover'], index=dates)
        results[strategy]['hhi'] = pd.Series(results[strategy]['hhi'], index=dates)
        results[strategy]['weights'] = pd.DataFrame(results[strategy]['weights'],
                                                     index=dates, columns=industries)

    return results


def load_cached_backtest(output_path='./output/'):
    """
    Load pre-computed baseline backtest results from CSVs.
    Returns dict with same structure as run_backtest().
    Falls back to run_backtest() if cache files are missing.
    """
    import os
    required = ['strategy_returns.csv', 'strategy_turnover.csv',
                'strategy_weights_mv.csv', 'strategy_weights_minvar.csv',
                'strategy_hhi.csv']
    if not all(os.path.exists(os.path.join(output_path, f)) for f in required):
        return None

    returns = pd.read_csv(f'{output_path}strategy_returns.csv',
                          index_col=0, parse_dates=True)
    turnover = pd.read_csv(f'{output_path}strategy_turnover.csv',
                           index_col=0, parse_dates=True)
    w_mv = pd.read_csv(f'{output_path}strategy_weights_mv.csv',
                        index_col=0, parse_dates=True)
    w_minvar = pd.read_csv(f'{output_path}strategy_weights_minvar.csv',
                            index_col=0, parse_dates=True)
    hhi = pd.read_csv(f'{output_path}strategy_hhi.csv',
                       index_col=0, parse_dates=True)

    n_assets = w_mv.shape[1]
    w_1n = pd.DataFrame(1.0 / n_assets,
                         index=returns.index, columns=w_mv.columns)

    results = {}
    for strategy in ['1/N', 'Mean-Variance', 'Minimum Variance']:
        if strategy == '1/N':
            w = w_1n
        elif strategy == 'Mean-Variance':
            w = w_mv
        else:
            w = w_minvar
        results[strategy] = {
            'returns': returns[strategy],
            'turnover': turnover[strategy],
            'weights': w,
            'hhi': hhi[strategy],
        }
    return results


# =============================================================================
# LaTeX export (disabled for portable submission):
# To re-enable, uncomment and adjust the paths in config.py, then
# uncomment sync_to_latex() calls in the individual scripts.
# =============================================================================

def sync_to_latex(local_file_path, target_dir):
    """
    Copy a local output file to the corresponding LaTeX thesis subdirectory.

    Silently skips if the target directory is unavailable or the file is
    locked, so pipeline execution is never interrupted.
    """
    try:
        os.makedirs(target_dir, exist_ok=True)
        dest = os.path.join(target_dir, os.path.basename(local_file_path))
        shutil.copy2(local_file_path, dest)
        print(f"    [sync] {os.path.basename(local_file_path)} -> {target_dir}")
    except Exception as e:
        print(f"    [sync] SKIPPED {os.path.basename(local_file_path)}: {e}")
