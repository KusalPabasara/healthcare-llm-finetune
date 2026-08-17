"""Compile a Markdown doc to PDF via XeLaTeX.

    python scripts/build_pdf.py docs/TEAM_GUIDE.md

The Markdown carries its own title block and contents list, because GitHub
renders it directly and needs them. The LaTeX template provides a title page
and a generated table of contents, so both are stripped before compiling —
otherwise the PDF shows each twice.

Requires xelatex, pandoc, and the Linux Libertine fonts.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent.parent / "docs" / "template.tex"


def strip_front_matter(text: str) -> str:
    """Remove the H1 + metadata table + inline contents list."""
    # Leading H1 and the metadata table that follows it.
    text = re.sub(
        r"\A#\s+.*?\n\n\|\s*\|\s*\|\n\|[-|]+\|\n(?:\|.*\n)+\n---\n\n",
        "",
        text,
        count=1,
    )
    # The hand-written contents list; the template generates its own.
    text = re.sub(r"###\s+Contents\n\n(?:\d+\.\s+\[.*\n)+\n---\n\n", "", text, count=1)

    # Section-break rules. They separate sections usefully in Markdown, but in
    # print the section headings already carry a rule and a second one reads
    # as an artifact.
    text = re.sub(r"\n---\n", "\n", text)
    return text


def widen_tables(text: str) -> str:
    """Convert pipe tables to grid tables so long cells wrap.

    pandoc renders a pipe table as plain `l` columns regardless of the dashes
    in the separator row, and `l` never wraps -- a long cell simply runs past
    the margin. Grid tables are the only syntax that makes pandoc emit
    proportional `p{}` columns, so wide tables are rewritten into grid form
    with widths proportional to their longest cell.
    """
    lines = text.splitlines()
    out, i = [], 0

    while i < len(lines):
        sep = re.fullmatch(r"\s*\|(?:\s*:?-+:?\s*\|)+\s*", lines[i] or "")
        header = lines[i - 1] if i else ""
        if sep and header.strip().startswith("|"):
            rows = [header]
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(lines[j])
                j += 1

            cells = [[c.strip() for c in r.split("|")[1:-1]] for r in rows]
            ncol = len(cells[0])
            cells = [c for c in cells if len(c) == ncol]
            if ncol < 2:
                out.append(lines[i])
                i += 1
                continue

            # Width each column by its longest cell, then scale to the measure.
            longest = [max(len(r[c]) for r in cells) for c in range(ncol)]
            budget = 96 - (3 * ncol)
            total = sum(longest) or 1
            widths = [max(9, round(w / total * budget)) for w in longest]

            def rule(ch):
                return "+" + "+".join(ch * (w + 2) for w in widths) + "+"

            def row(vals):
                # Wrap each cell to its column width, emitting continuation
                # lines so the grid stays valid.
                import textwrap

                wrapped = [
                    textwrap.wrap(v, w) or [""] for v, w in zip(vals, widths)
                ]
                height = max(len(x) for x in wrapped)
                lines_out = []
                for k in range(height):
                    parts = []
                    for c in range(ncol):
                        piece = wrapped[c][k] if k < len(wrapped[c]) else ""
                        parts.append(" " + piece.ljust(widths[c]) + " ")
                    lines_out.append("|" + "|".join(parts) + "|")
                return lines_out

            if out and out[-1] == header:
                out.pop()
            out.append(rule("-"))
            out.extend(row(cells[0]))
            out.append(rule("="))
            for r in cells[1:]:
                out.extend(row(r))
                out.append(rule("-"))

            out.append("")
            i = j
            continue

        out.append(lines[i])
        i += 1

    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="defaults to source with .pdf")
    ap.add_argument("--keep-tex", action="store_true", help="also write the .tex")
    args = ap.parse_args()

    if not args.source.exists():
        sys.exit(f"error: {args.source} not found")
    if not TEMPLATE.exists():
        sys.exit(f"error: template missing at {TEMPLATE}")
    for tool in ("pandoc", "xelatex"):
        if not shutil.which(tool):
            sys.exit(f"error: {tool} not installed")

    out = args.output or args.source.with_suffix(".pdf")
    body = widen_tables(strip_front_matter(args.source.read_text()))

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
        fh.write(body)
        tmp = Path(fh.name)

    cmd = [
        "pandoc", str(tmp),
        # grid tables (from widen_tables) need the pandoc reader, not gfm
        "-f", "markdown",
        "--template", str(TEMPLATE),
        "--pdf-engine", "xelatex",
        "--toc", "--toc-depth", "2",
        "-o", str(out),
    ]

    print(f"compiling {args.source.name} -> {out.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    tmp.unlink(missing_ok=True)

    if result.returncode != 0:
        # LaTeX errors are buried in a wall of log output; surface the lines
        # that actually say what went wrong.
        err = (result.stdout + result.stderr).splitlines()
        for line in err:
            if line.startswith("!") or "Error" in line or line.startswith("l."):
                print("  " + line)
        sys.exit("compilation failed")

    if args.keep_tex:
        tex = out.with_suffix(".tex")
        subprocess.run(
            ["pandoc", str(args.source), "-f", "gfm", "--template", str(TEMPLATE),
             "--toc", "--toc-depth", "2", "-o", str(tex)],
            capture_output=True,
        )
        print(f"  wrote {tex.name}")

    pages = subprocess.run(
        ["pdfinfo", str(out)], capture_output=True, text=True
    ).stdout
    n = next((l.split(":")[1].strip() for l in pages.splitlines()
              if l.startswith("Pages:")), "?")
    size = out.stat().st_size / 1024
    print(f"  {n} pages, {size:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
