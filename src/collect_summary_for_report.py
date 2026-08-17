"""Collect Fernando's per-slug summary-statistics outputs into one include file
the LaTeX report \\input's. Excludes its own output file to avoid recursion, and
wraps each wide table in \\resizebox so it never overflows the margin."""

from pathlib import Path

try:
    from settings import config

    OUTPUT_DIR = Path(config("OUTPUT_DIR"))
except Exception:
    OUTPUT_DIR = Path("_output")

INCLUDE_NAME = "summary_statistics_include.tex"
SLUG_TITLES = {
    "cape": "Cyclically-adjusted P/E (CAPE) and forward equity returns",
    "credit_spreads": "Credit spreads (Baa/Aaa--Treasury)",
    "gdp_growth": "GDP growth",
    "hy_share": "High-yield issuance share",
}
DEFAULT_FIGURE_WIDTH = r"0.85\textwidth"
FIGURE_WIDTH_OVERRIDES = {"cape": r"0.65\textwidth"}


def _san(s):
    s = s.replace("_", " ")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#")]:
        s = s.replace(a, b)
    return s


def _title(slug):
    return SLUG_TITLES.get(slug, _san(slug).title())


def main():
    tex = sorted(
        f for f in OUTPUT_DIR.glob("summary_statistics_*.tex") if f.name != INCLUDE_NAME
    )
    pdf = sorted(OUTPUT_DIR.glob("summary_statistics_*.pdf"))
    out = []
    for f in tex:
        slug = f.stem.replace("summary_statistics_", "")
        out += [
            r"\begin{table}[H]\centering",
            r"\caption{\label{tab:summary-%s}Summary statistics: %s.}"
            % (slug.replace("_", "-"), _title(slug)),
            r"\resizebox{\textwidth}{!}{%",
            r"\input{../_output/%s}" % f.name,
            r"}",
            r"\end{table}",
        ]
    for f in pdf:
        slug = f.stem.replace("summary_statistics_", "")
        width = FIGURE_WIDTH_OVERRIDES.get(slug, DEFAULT_FIGURE_WIDTH)
        out += [
            r"\begin{figure}[H]\centering",
            r"\includegraphics[width=%s]{%s}" % (width, f.name),
            r"\caption{\label{fig:summary-%s}%s.}"
            % (slug.replace("_", "-"), _title(slug)),
            r"\end{figure}",
        ]
    if not out:
        out = ["% (no summary_statistics_* files found in _output)"]
    (OUTPUT_DIR / INCLUDE_NAME).write_text("\n".join(out) + "\n")
    print(
        "wrote",
        OUTPUT_DIR / INCLUDE_NAME,
        "-",
        len(tex),
        "tables,",
        len(pdf),
        "figures",
    )


if __name__ == "__main__":
    main()
