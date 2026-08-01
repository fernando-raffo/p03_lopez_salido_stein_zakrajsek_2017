"""
Helper functions to calculate growth rates and other transformations
of time series data needed for the regressions.
"""

import numpy as np


def year_over_year_growth(series):
    """Annual YOY log growth (percent) of a level series indexed by year."""
    s = series.astype(float).sort_index()
    return 100.0 * np.log(s).diff()


def quarter_over_quarter_growth(series):
    """QOQ log growth (percent) of a level series indexed by quarterly dates."""
    s = series.astype(float).sort_index()
    return 100.0 * np.log(s).diff()


def forward_cumulative_growth(level_series, horizon):
    """Cumulative log growth (percent) from t to t+horizon."""
    s = np.log(level_series.astype(float).sort_index())
    return 100.0 * (s.shift(-horizon) - s)
