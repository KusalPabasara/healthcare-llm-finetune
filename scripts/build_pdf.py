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
    body = strip_front_matter(args.source.read_text())

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
        fh.write(body)
        tmp = Path(fh.name)

    cmd = [
        "pandoc", str(tmp),
        "-f", "gfm",
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
