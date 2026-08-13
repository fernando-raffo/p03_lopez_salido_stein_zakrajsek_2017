# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Replicating Lopez-Salido, Stein & Zakrajsek (2017)
# ### *Credit-Market Sentiment and the Business Cycle*, QJE 132(3)
#
# This notebook walks through our replication of the paper's core results on
# annual U.S. data, 1929-2015. The thesis: **elevated credit-market
# sentiment** — unusually narrow credit spreads and a high high-yield issuance
# share — **predicts a subsequent slowdown in real activity**, because frothy
# credit conditions mean-revert.
#
# | Exhibit | What it shows | Status |
# |---|---|---|
# | Figure I | The Baa-Treasury credit spread, 1929-2015 | done |
# | Table I  | Credit spread changes vs. equity returns as growth predictors | done |
# | Table II | Two-step sentiment forecast of growth | pending (#6) |
# | Figure II | Fitted sentiment vs. realized growth | pending (#6) |

# %%
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.cwd().parent / "src"))
from settings import config

PROCESSED_DATA_DIR = Path(config("PROCESSED_DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

# %% [markdown]
# ## Data
#
# All series come from the pipeline (`doit pull_fred process_fred_data`,
# `doit pull_shiller`). We read the processed files directly rather than
# re-pulling.

# %%
annual = pd.read_parquet(PROCESSED_DATA_DIR / "fred_final_series_annual.parquet")
monthly = pd.read_parquet(PROCESSED_DATA_DIR / "fred_final_series_monthly.parquet")
print("Annual columns:", annual.columns.tolist())
annual.head()

# %% [markdown]
# ## Figure I — The Baa-Treasury credit spread
#
# The paper's opening exhibit: the credit spread is strongly countercyclical,
# spiking into every recession. Its *change* is the key predictor in Table I.

# %%
spread = monthly["BAA_Treasury_spread"].dropna()
spread = spread[(spread.index >= "1929") & (spread.index <= "2015")]

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(spread.index, spread.values, color="#1f4e79", lw=1.1)
ax.set_title("Figure I: Baa-Treasury Credit Spread, 1929-2015")
ax.set_ylabel("Percentage points")
plt.show()

# %% [markdown]
# ## Table I — Credit beats equities
#
# Table I forecasts next-year real GDP-per-capita growth ($\Delta y_{t+1}$)
# with the current-year change in the credit spread ($\Delta s_t$) and the S&P
# 500 total return ($r^{SP}_t$), Newey-West errors. Three columns: credit only,
# equity only, and both plus rate/inflation controls.
#
# The **no-dummies 1929-2015** table is the faithful match to the published QJE
# Table I. Our replication reproduces it almost coefficient-for-coefficient:
#
# | | Published QJE | Our replication |
# |---|---|---|
# | col 1 $\Delta s_t$ | -1.997 | -1.958 |
# | col 2 $r^{SP}_t$ | 0.081 | 0.075 |
# | col 3 $\Delta s_t$ | -2.061 | -2.124 |
# | col 3 $r^{SP}_t$ | 0.029 | 0.022 |
#
# Column 3 is the paper's headline: with both predictors in, **credit stays
# strongly significant while the equity coefficient collapses** — the credit
# market forecasts the economy better than the stock market.

# %%
import replicate_table_1 as t1

df = t1.build_panel()
res = t1.run_regression(
    df,
    ["d_credit_spread", "sp_return", "d_treasury_3mo",
     "d_treasury_10yr", "CPI_inflation", "gdp_pc_growth"],
    t1.REP_START, t1.REP_END,
)
print(res.summary())

# %% [markdown]
# The generated LaTeX tables are written to `_output/` by
# `doit replicate_table_1`. `table_1_replication_nodummies.tex` matches the
# published paper; the `dummies` variants match the working-paper spec, and the
# `extended` variants carry the sample through 2023 (the Shiller endpoint).

# %%
print((OUTPUT_DIR / "table_1_replication_nodummies.tex").read_text())

# %% [markdown]
# ## Table II & Figure II — the sentiment two-step  *(pending)*
#
# Table II is the analytical core: a *first stage* forecasting the change in the
# credit spread from lagged sentiment measures (the log credit spread, the log
# high-yield issuance share $\ln[HYS]$, and $\ln[P/E10]$), then a *second stage*
# regressing growth on the fitted spread change. Figure II visualizes the fitted
# sentiment index against realized growth.
#
# This work lives on the `6-replicate-table-ii-2` branch and is not yet on
# `main`. Once #6 merges, this section imports `replicate_table_2` and renders
# Table II + Figure II, completing the replication spine.

# %% [markdown]
# ## Summary
#
# On the results available on `main`, our pipeline reproduces the published QJE
# Figure I and Table I closely — including the credit-beats-equities result in
# Table I column 3. Table II/Figure II follow once #6 lands.
