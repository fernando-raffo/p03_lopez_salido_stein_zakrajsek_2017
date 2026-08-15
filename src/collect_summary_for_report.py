"""Collect Fernando's per-slug summary-statistics outputs into one include file
that the LaTeX report can \\input. Robust to whatever slugs his summary script
emits: scans OUTPUT_DIR for summary_statistics_*.tex / *.pdf and wraps each in a
captioned float. Writes _output/summary_statistics_include.tex (empty-safe)."""
from pathlib import Path
try:
    from settings import config
    OUTPUT_DIR = Path(config("OUTPUT_DIR"))
except Exception:
    OUTPUT_DIR = Path("_output")


def _san(s):
    s = s.replace("_", " ")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#")]:
        s = s.replace(a, b)
    return s


def main():
    tex = sorted(OUTPUT_DIR.glob("summary_statistics_*.tex"))
    pdf = sorted(OUTPUT_DIR.glob("summary_statistics_*.pdf"))
    out = []
    for f in tex:
        slug = _san(f.stem.replace("summary_statistics_", ""))
        out += [r"\begin{table}[H]\centering",
                r"\caption{Summary statistics: %s.}" % slug,
                r"\input{../_output/%s}" % f.name,
                r"\end{table}"]
    for f in pdf:
        slug = _san(f.stem.replace("summary_statistics_", ""))
        out += [r"\begin{figure}[H]\centering",
                r"\includegraphics[width=0.85\textwidth]{%s}" % f.name,
                r"\caption{Summary statistics: %s.}" % slug,
                r"\end{figure}"]
    if not out:
        out = ["% (no summary_statistics_* files found in _output)"]
    (OUTPUT_DIR / "summary_statistics_include.tex").write_text("\n".join(out) + "\n")
    print("wrote", OUTPUT_DIR / "summary_statistics_include.tex", "-", len(tex), "tables,", len(pdf), "figures")


if __name__ == "__main__":
    main()
