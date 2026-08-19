"""
Helper functions to calculate growth rates and other transformations
of time series data needed for the regressions.
"""

import numpy as np


def year_over_year_growth(series):
    """Annual YOY log growth (percent) of a level series indexed by year."""
    s = series.astype(float).sort_index()
    return 100.0 * np.log(s).diff()


def log_total_return(price, div):
    """Annual log total return (percent) of an asset paying dividends: 100 * ln((P_t + div_t) / P_{t-1})."""
    p = price.astype(float).sort_index()
    div = div.astype(float).sort_index()
    return 100.0 * np.log((p + div) / p.shift(1))


def to_percent(series):
    """Convert a decimal (fractional) series to percentage-point units."""
    return 100.0 * series.astype(float)
