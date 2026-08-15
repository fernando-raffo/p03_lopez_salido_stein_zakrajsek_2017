"""Summary statistics table and figure for the underlying data (issue #32).

Builds a descriptive-statistics table and a companion chart for the key series
used in the replication, so the reader can gauge scale, dispersion, and
co-movement of the underlying data before seeing the regressions. It reuses the
project's own panel builder so column names always match the analysis.

Outputs (written to OUTPUT_DIR):
    table_summary_stats.tex   -- LaTeX tabular, \\input-ed by reports/report.tex
    figure_summary_stats.pdf  -- standardized (z-scored) paths of the headline series
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from settings import config
from replicate_table_2 import build_panel

OUTPUT_DIR = Path(config("OUTPUT_DIR"))

LABELS = {
    "BAA_Treasury_spread": "Baa--Treasury spread (pp)",
    "AAA_Treasury_spread": "Aaa--Treasury spread (pp)",
    "spread": "Credit spread (pp)",
    "s_hat": "Fitted credit spread (pp)",
    "d_y": "Real GDP growth (\\%)",
    "gdp_growth": "Real GDP growth (\\%)",
    "r_sp": "S\\&P 500 return (\\%)",
    "hy_share": "High-yield issuance share",
    "ln_hy_share": "Log high-yield share",
    "ln_pe10": "Log Shiller P/E10",
    "pe10": "Shiller P/E10",
    "i3m": "3m Treasury yield (\\%)",
    "i10y": "10y Treasury yield (\\%)",
    "pi": "Inflation (\\%)",
    "inflation": "Inflation (\\%)",
}


def _label(c):
    return LABELS.get(c, str(c).replace("_", r"\_"))


def main():
    df = build_panel()
    num = df.select_dtypes(include="number").copy()

    keys = [c for c in LABELS if c in num.columns]
    cols = keys if len(keys) >= 3 else list(num.columns)
    sub = num[cols]

    # ---- summary-statistics table ----
    stats = sub.describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    stats = stats.rename(columns={"25%": "p25", "50%": "median", "75%": "p75"})
    stats = stats.rename(index={c: _label(c) for c in stats.index})
    stats.index.name = "Variable"
    stats["count"] = stats["count"].astype(int)
    float_format_func = lambda x: "{:.2f}".format(x)
    latex = stats.to_latex(float_format=float_format_func, escape=False, na_rep="")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "table_summary_stats.tex").write_text(latex)
    print("wrote", OUTPUT_DIR / "table_summary_stats.tex")

    # ---- companion chart: standardized headline series over time ----
    preferred = ["BAA_Treasury_spread", "spread", "s_hat", "hy_share",
                 "ln_hy_share", "d_y", "gdp_growth"]
    plot_cols = [c for c in preferred if c in sub.columns][:3]
    if not plot_cols:
        plot_cols = list(sub.columns)[:3]
    z = (sub[plot_cols] - sub[plot_cols].mean()) / sub[plot_cols].std()
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for c in plot_cols:
        label = LABELS.get(c, c).replace("\\%", "%").replace("\\&", "&")
        ax.plot(z.index, z[c], label=label)
    ax.axhline(0, lw=0.6, color="0.6")
    ax.set_xlabel("Year")
    ax.set_ylabel("Standardized (z-score)")
    ax.legend(fontsize=7, ncol=len(plot_cols), loc="upper center")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_summary_stats.pdf")
    print("wrote", OUTPUT_DIR / "figure_summary_stats.pdf")


if __name__ == "__main__":
    main()
