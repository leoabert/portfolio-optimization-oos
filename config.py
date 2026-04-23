"""
config.py - Central Configuration for Bachelor Thesis
======================================================
All paths, parameters, and settings in one place.
Import this in all other scripts.
"""

import os

# =============================================================================
# PATHS (relative to project root)
# =============================================================================
DATA_DIR = './data/'
OUTPUT_DIR = './output/'

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}tables/', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}figures/', exist_ok=True)

# =============================================================================
# DATA PARAMETERS
# =============================================================================
START_DATE = '1963-07-01'     # Earliest available Fama-French data
END_DATE = '2025-12-31'       # Request through end of 2025 (returns whatever is available)

INDUSTRIES = ['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec',
              'Telcm', 'Shops', 'Hlth', 'Utils', 'Other']

# =============================================================================
# BACKTEST PARAMETERS
# =============================================================================
ESTIMATION_WINDOW = 120      # months (10 years)
TRANSACTION_COST = 0.0050    # 50 basis points
RISK_AVERSION = 1            # γ (DeMiguel standard)

# =============================================================================
# REGIME SPLIT
# =============================================================================
SPLIT_DATE = '2009-01-01'

# =============================================================================
# ROBUSTNESS PARAMETERS
# =============================================================================
TRANSACTION_COSTS_SENSITIVITY = [0.0000, 0.0025, 0.0050, 0.0100]  # 0, 25, 50, 100 bps
RISK_AVERSIONS_SENSITIVITY = [1, 3, 5]
ESTIMATION_WINDOWS_SENSITIVITY = [60, 90, 120, 180]
BOOTSTRAP_ITERATIONS = 10_000

# =============================================================================
# API KEYS
# =============================================================================
# NOTE: Insert your own FRED API key here, or set the FRED_API_KEY environment
# variable. Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = os.environ.get('FRED_API_KEY', '')

# =============================================================================
# LaTeX export (disabled for portable submission):
# To re-enable, uncomment and adjust the paths below to your LaTeX project.
# =============================================================================
# LATEX_BASE_DIR = r"/path/to/your/latex/project"
# LATEX_TABLES_DIR = os.path.join(LATEX_BASE_DIR, "tables")
# LATEX_FIGURES_DIR = os.path.join(LATEX_BASE_DIR, "figures")
# LATEX_FIG_APP_DIR = os.path.join(LATEX_BASE_DIR, "figures_appendix")

# =============================================================================
# PLOT SETTINGS
# =============================================================================
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
