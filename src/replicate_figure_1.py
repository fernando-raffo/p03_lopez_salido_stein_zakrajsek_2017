"""Replicate Figure I of Lopez-Salido, Stein & Zakrajsek (2017):
the Baa-Treasury credit spread over the 1929-2015 replication sample.

Reads the cleaned monthly FRED series produced by process_fred_data_monthly.py
and shades NBER recessions using the hist_recession_indicator column.
"""
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from settings import config

sns.set()

PROCESSED_DATA_DIR = Path(config("PROCESSED_DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
START = config("REPLICATION_START_DATE")
END = config("REPLICATION_END_DATE")

df = pd.read_parquet(PROCESSED_DATA_DIR / "fred_final_series_monthly.parquet")
df = df.loc[(df.index >= START) & (df.index <= END)]

spread = df["BAA_Treasury_spread"].dropna()

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(spread.index, spread.values, color="C0", lw=1.1)

# Shade NBER recessions directly from the data
rec = df["hist_recession_indicator"].fillna(0).astype(int)
in_rec = rec.eq(1)
if in_rec.any():
    starts = df.index[in_rec & ~in_rec.shift(1, fill_value=False)]
    ends = df.index[in_rec & ~in_rec.shift(-1, fill_value=False)]
    for s, e in zip(starts, ends):
        ax.axvspan(s, e, color="grey", alpha=0.25, lw=0)

ax.set_title("Figure I: Baa-Treasury Credit Spread, 1929-2015")
ax.set_ylabel("Percentage points")
ax.set_xlabel("")
ax.margins(x=0.01)
fig.tight_layout()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out = OUTPUT_DIR / "figure_1.png"
fig.savefig(out, dpi=200)
print(f"saved {out}  (n={len(spread)}, "
      f"{spread.index.min().date()}..{spread.index.max().date()})")
