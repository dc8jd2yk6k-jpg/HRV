"""A small document model rendered to Markdown and to LaTeX (-> PDF).

Text in every block uses one inline markup that both outputs understand:
``**bold**``, ``*italic*``, ```code```, ``[text](target)`` where a target
starting with ``#`` links to a section. Unicode symbols (−, ×, ≈, ≥, …) are
passed through to LaTeX, which maps them via ``\\DeclareUnicodeCharacter``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# ----------------------------------------------------------------------------
# Blocks


@dataclass
class Heading:
    text: str
    level: int = 1  # 1 = section, 2 = subsection
    numbered: bool = True

    @property
    def slug(self) -> str:
        return slugify(self.text)


@dataclass
class Para:
    text: str


@dataclass
class ListBlock:
    items: list  # str, or (str, [sub-item str, ...])
    ordered: bool = False


@dataclass
class Table:
    header: list[str]
    rows: list[list[str]]
    colspec: str = ""  # LaTeX column spec; default: first column l, rest r
    small: bool = False  # many columns: use a smaller font in LaTeX


@dataclass
class Figure:
    stem: str  # path relative to the output dir, without extension
    caption: str
    width: float = 1.0  # fraction of the text width in LaTeX

    @property
    def label(self) -> str:
        return "fig:" + Path(self.stem).name


@dataclass
class Code:
    text: str
    lang: str = "bash"


@dataclass
class Box:
    title: str
    text: str


@dataclass
class Document:
    title: str
    subtitle: str = ""
    note: str = ""
    blocks: list = field(default_factory=list)

    def add(self, *blocks) -> None:
        self.blocks.extend(blocks)


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9 -]", "", text.lower())
    return re.sub(r"\s+", "-", s.strip())


def _numbering(blocks) -> dict[int, str]:
    """Section numbers ("2", "2.1") for numbered headings, by block index."""
    nums, sec, sub = {}, 0, 0
    for i, b in enumerate(blocks):
        if isinstance(b, Heading) and b.numbered:
            if b.level == 1:
                sec, sub = sec + 1, 0
                nums[i] = f"{sec}"
            else:
                sub += 1
                nums[i] = f"{sec}.{sub}"
    return nums


# ----------------------------------------------------------------------------
# Markdown


def to_markdown(doc: Document) -> str:
    out = [f"# {doc.title}" + (f" — {doc.subtitle}" if doc.subtitle else ""), ""]
    if doc.note:
        out += [f"_{doc.note}_", ""]
    nums = _numbering(doc.blocks)
    fig_no = 0
    for i, b in enumerate(doc.blocks):
        if isinstance(b, Heading):
            prefix = f"{nums[i]}. " if b.level == 1 and i in nums else (f"{nums[i]} " if i in nums else "")
            out += ["#" * (b.level + 1) + " " + prefix + b.text, ""]
        elif isinstance(b, Para):
            out += [b.text, ""]
        elif isinstance(b, ListBlock):
            for k, item in enumerate(b.items, 1):
                text, subs = (item, []) if isinstance(item, str) else item
                out.append(f"{k}. {text}" if b.ordered else f"* {text}")
                out += [f"   * {s}" for s in subs]
            out.append("")
        elif isinstance(b, Table):
            out.append("| " + " | ".join(b.header) + " |")
            out.append("|" + "|".join("---" for _ in b.header) + "|")
            out += ["| " + " | ".join(r) + " |" for r in b.rows]
            out.append("")
        elif isinstance(b, Figure):
            fig_no += 1
            out += [f"![{Path(b.stem).name}]({b.stem}.png)", "", f"*Figure {fig_no}. {b.caption}*", ""]
        elif isinstance(b, Code):
            out += [f"```{b.lang}", b.text, "```", ""]
        elif isinstance(b, Box):
            out += [f"> **{b.title}** {b.text}", ""]
    for i, line in enumerate(out):  # internal links point at GitHub's heading anchors
        out[i] = re.sub(r"\]\(#([^)]+)\)", lambda m: f"](#{_md_anchor(doc, m.group(1))})", line)
    return "\n".join(out).rstrip() + "\n"


def _md_anchor(doc: Document, slug: str) -> str:
    nums = _numbering(doc.blocks)
    for i, b in enumerate(doc.blocks):
        if isinstance(b, Heading) and b.slug == slug:
            prefix = f"{nums[i]}. " if b.level == 1 and i in nums else (f"{nums[i]} " if i in nums else "")
            return slugify(prefix + b.text)
    return slug


# ----------------------------------------------------------------------------
# LaTeX

_TEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
}
_INLINE = re.compile(r"`([^`]+)`|\*\*(.+?)\*\*|\*(.+?)\*|\[([^\]]+)\]\(([^)]+)\)")


def tex_escape(text: str, code: bool = False) -> str:
    s = "".join(_TEX_SPECIAL.get(ch, ch) for ch in text)
    if code:
        # Let long paths and identifiers break after separators.
        s = re.sub(r"(/|\\_|\.)", r"\1\\allowbreak{}", s)
    else:
        s = re.sub(r'"([^"]*)"', r"``\1''", s)
        s = re.sub(r"(?<![A-Za-z])'([^']+)'(?![A-Za-z])", r"`\1'", s)
    return s


def tex_inline(text: str) -> str:
    out, pos = [], 0
    for m in _INLINE.finditer(text):
        out.append(tex_escape(text[pos : m.start()]))
        code, bold, ital, ltext, target = m.groups()
        if code is not None:
            out.append(r"\texttt{" + tex_escape(code, code=True) + "}")
        elif bold is not None:
            out.append(r"\textbf{" + tex_inline(bold) + "}")
        elif ital is not None:
            out.append(r"\emph{" + tex_inline(ital) + "}")
        elif target.startswith("#"):
            out.append(r"\hyperref[sec:" + target[1:] + "]{" + tex_inline(ltext) + "}")
        else:
            out.append(r"\href{" + target.replace("%", r"\%").replace("#", r"\#") + "}{" + tex_inline(ltext) + "}")
        pos = m.end()
    out.append(tex_escape(text[pos:]))
    return "".join(out)


PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{textcomp}
\usepackage{amsmath}
\usepackage[protrusion=true,expansion=true]{microtype}
\usepackage[a4paper,margin=2.2cm,top=2.4cm,bottom=2.4cm]{geometry}
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage[table]{xcolor}
\usepackage[font=small,labelfont={bf,sf},format=hang,skip=6pt]{caption}
\usepackage{enumitem}
\usepackage{fancyhdr}
\usepackage[explicit]{titlesec}
\usepackage[most]{tcolorbox}
\usepackage{fvextra}
\usepackage{parskip}
\usepackage{needspace}
\usepackage{xurl}
\usepackage[hidelinks]{hyperref}

\definecolor{strap}{HTML}{2A78D6}
\definecolor{watch}{HTML}{EB6834}
\definecolor{ink}{HTML}{0B0B0B}
\definecolor{inksoft}{HTML}{52514E}
\definecolor{rule}{HTML}{C3C2B7}
\definecolor{wash}{HTML}{F0EFEC}
\definecolor{link}{HTML}{1C5CAB}
\hypersetup{colorlinks=true,linkcolor=link,urlcolor=link,citecolor=link}

\DeclareUnicodeCharacter{2212}{\textminus}
\DeclareUnicodeCharacter{2248}{\ensuremath{\approx}}
\DeclareUnicodeCharacter{2265}{\ensuremath{\geq}}
\DeclareUnicodeCharacter{2264}{\ensuremath{\leq}}
\DeclareUnicodeCharacter{2192}{\ensuremath{\rightarrow}}
\DeclareUnicodeCharacter{03C1}{\ensuremath{\rho}}
\DeclareUnicodeCharacter{03B1}{\ensuremath{\alpha}}
\DeclareUnicodeCharacter{2026}{\ldots}

\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\newcolumntype{R}{>{\raggedleft\arraybackslash}X}
\renewcommand{\arraystretch}{1.18}
\setlength{\parskip}{5pt plus 1pt}
\setlist{nosep,leftmargin=1.4em,topsep=2pt}
\setlist[enumerate]{label=\arabic*.}

\titleformat{\section}{\sffamily\Large\bfseries\color{ink}}{\thesection}{0.6em}{#1}
\titleformat{name=\section,numberless}{\sffamily\Large\bfseries\color{ink}}{}{0em}{#1}
\titleformat{\subsection}{\sffamily\large\bfseries\color{ink}}{\thesubsection}{0.6em}{#1}
\titleformat{name=\subsection,numberless}{\sffamily\large\bfseries\color{ink}}{}{0em}{#1}
\titlespacing*{\section}{0pt}{14pt}{4pt}
\titlespacing*{\subsection}{0pt}{10pt}{2pt}

\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0.4pt}
\fancyhead[L]{\sffamily\small\color{inksoft}@@SHORTTITLE@@}
\fancyhead[R]{\sffamily\small\color{inksoft}\thepage}

\fvset{breaklines=true,breakanywhere=true,fontsize=\small}
\newtcolorbox{keybox}[1]{enhanced,breakable,colback=wash,colframe=rule,boxrule=0.5pt,arc=2pt,
  left=8pt,right=8pt,top=6pt,bottom=6pt,fonttitle=\sffamily\bfseries,coltitle=ink,colbacktitle=wash,
  title={#1},attach title to upper={\ }}
\newtcolorbox{codebox}{enhanced,breakable,colback=wash!50,colframe=rule,boxrule=0.4pt,arc=1pt,
  left=6pt,right=6pt,top=4pt,bottom=4pt}
"""


def to_latex(doc: Document, short_title: str = "") -> str:
    out = [PREAMBLE.replace("@@SHORTTITLE@@", tex_escape(short_title or doc.title))]
    out.append(r"\hypersetup{pdftitle={" + tex_escape(doc.title + (" — " + doc.subtitle if doc.subtitle else "")) + "}}")
    out.append(r"\begin{document}")
    out.append(r"\thispagestyle{plain}")
    out.append(r"\begin{center}")
    out.append(r"{\sffamily\bfseries\LARGE " + tex_inline(doc.title) + r"\par}")
    if doc.subtitle:
        out.append(r"\vspace{3pt}{\sffamily\large\color{inksoft} " + tex_inline(doc.subtitle) + r"\par}")
    if doc.note:
        out.append(r"\vspace{6pt}{\small\itshape\color{inksoft} " + tex_inline(doc.note) + r"\par}")
    out.append(r"\end{center}")
    out.append(r"\vspace{2pt}")

    for b in doc.blocks:
        if isinstance(b, Heading):
            cmd = "section" if b.level == 1 else "subsection"
            # Keep a heading together with the start of what follows it.
            out.append(r"\Needspace{%d\baselineskip}" % (12 if b.level == 1 else 10))
            if b.numbered:
                out.append(rf"\{cmd}{{{tex_inline(b.text)}}}\label{{sec:{b.slug}}}")
            else:
                out.append(rf"\phantomsection\{cmd}*{{{tex_inline(b.text)}}}\label{{sec:{b.slug}}}")
                out.append(rf"\addcontentsline{{toc}}{{{cmd}}}{{{tex_inline(b.text)}}}")
        elif isinstance(b, Para):
            out.append(tex_inline(b.text) + "\n")
        elif isinstance(b, ListBlock):
            env = "enumerate" if b.ordered else "itemize"
            out.append(rf"\begin{{{env}}}")
            for item in b.items:
                text, subs = (item, []) if isinstance(item, str) else item
                out.append(r"\item " + tex_inline(text))
                if subs:
                    out.append(r"\begin{itemize}")
                    out += [r"\item " + tex_inline(s) for s in subs]
                    out.append(r"\end{itemize}")
            out.append(rf"\end{{{env}}}")
        elif isinstance(b, Table):
            out.append(_tex_table(b))
        elif isinstance(b, Figure):
            out.append(r"\begin{figure}[H]")
            out.append(r"\centering")
            out.append(rf"\includegraphics[width={b.width:g}\linewidth]{{{b.stem}.pdf}}")
            out.append(r"\caption{" + tex_inline(b.caption) + "}")
            out.append(rf"\label{{{b.label}}}")
            out.append(r"\end{figure}")
        elif isinstance(b, Code):
            out += [r"\begin{codebox}", r"\begin{Verbatim}", b.text, r"\end{Verbatim}", r"\end{codebox}"]
        elif isinstance(b, Box):
            out += [r"\begin{keybox}{" + tex_inline(b.title) + "}", tex_inline(b.text), r"\end{keybox}"]
    out.append(r"\end{document}")
    return "\n".join(out) + "\n"


def _tex_table(t: Table) -> str:
    n = len(t.header)
    spec = t.colspec or ("l" + "r" * (n - 1))
    tabx = any(ch in spec for ch in "XYCR")
    lines = [r"\begin{center}", r"\footnotesize" if t.small else r"\small"]
    if t.small:
        lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(rf"\begin{{tabularx}}{{\linewidth}}{{{spec}}}" if tabx else rf"\begin{{tabular}}{{{spec}}}")
    lines.append(r"\toprule")
    if any(h.strip() for h in t.header):
        lines.append(" & ".join(r"\textbf{" + tex_inline(h) + "}" for h in t.header) + r" \\")
        lines.append(r"\midrule")
    for r in t.rows:
        lines.append(" & ".join(tex_inline(c) for c in r) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}" if tabx else r"\end{tabular}")
    lines.append(r"\end{center}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# PDF


def latex_engine() -> str | None:
    for exe in ("latexmk", "pdflatex"):
        if shutil.which(exe):
            return exe
    return None


def build_pdf(tex_path: Path) -> Path:
    """Compile *tex_path* in its own directory and remove the auxiliary files."""
    tex_path = Path(tex_path)
    engine = latex_engine()
    if engine is None:
        raise RuntimeError("no LaTeX engine found (install TeX Live: pdflatex/latexmk)")
    cwd, name = tex_path.parent, tex_path.name
    flags = ["-interaction=nonstopmode", "-halt-on-error"]
    if engine == "latexmk":
        cmds = [["latexmk", "-pdf", "-quiet", *flags, name]]
    else:
        cmds = [["pdflatex", *flags, name]] * 2
    for cmd in cmds:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if proc.returncode != 0:
            log = tex_path.with_suffix(".log")
            tail = log.read_text(errors="replace")[-3000:] if log.exists() else proc.stdout[-3000:]
            raise RuntimeError(f"LaTeX failed for {tex_path}:\n{tail}")
    for ext in (".aux", ".log", ".out", ".fls", ".fdb_latexmk", ".toc"):
        tex_path.with_suffix(ext).unlink(missing_ok=True)
    return tex_path.with_suffix(".pdf")
