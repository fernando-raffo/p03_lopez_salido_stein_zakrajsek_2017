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


def _san(s):
    s = s.replace("_", " ")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#")]:
        s = s.replace(a, b)
    return s


def main():
    tex = sorted(
        f for f in OUTPUT_DIR.glob("summary_statistics_*.tex") if f.name != INCLUDE_NAME
    )
    pdf = sorted(OUTPUT_DIR.glob("summary_statistics_*.pdf"))
    out = []
    for f in tex:
        slug = _san(f.stem.replace("summary_statistics_", ""))
        out += [
            r"\begin{table}[H]\centering",
            r"\caption{Summary statistics: %s.}" % slug,
            r"\resizebox{\textwidth}{!}{%",
            r"\input{../_output/%s}" % f.name,
            r"}",
            r"\end{table}",
        ]
    for f in pdf:
        slug = _san(f.stem.replace("summary_statistics_", ""))
        out += [
            r"\begin{figure}[H]\centering",
            r"\includegraphics[width=0.85\textwidth]{%s}" % f.name,
            r"\caption{Summary statistics: %s.}" % slug,
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
