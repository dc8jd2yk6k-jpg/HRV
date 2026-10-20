import shutil

import pytest

from hrvcompare.doc import Box, Code, Document, Figure, Heading, ListBlock, Para, Table, build_pdf, tex_inline, to_latex, to_markdown


def test_tex_inline_escapes_and_converts_markup():
    assert tex_inline("50% of a_b & c") == r"50\% of a\_b \& c"
    assert tex_inline("**bold *it* end**") == r"\textbf{bold \emph{it} end}"
    assert tex_inline("see `tables/x_y.csv`") == r"see \texttt{tables/\allowbreak{}x\_\allowbreak{}y.\allowbreak{}csv}"
    assert tex_inline("[Reproduce](#reproduce)") == r"\hyperref[sec:reproduce]{Reproduce}"
    assert tex_inline('a "quoted" word') == "a ``quoted'' word"
    assert tex_inline("the watch's 'normal range'") == "the watch's `normal range'"


def _doc():
    d = Document("Title", "Sub", "Note with [link](#results)")
    d.add(
        Heading("Results"),
        Para("Value −8.0 ms ≈ 2% ≥ 80%"),
        ListBlock(["one", ("two", ["two-a"])], ordered=True),
        Table(["a", "b"], [["1", "2"]]),
        Figure("figures/x", "A caption"),
        Code("echo hi"),
        Box("Bottom line.", "text"),
        Heading("Appendix", numbered=False),
    )
    return d


def test_markdown_numbering_and_figures():
    md = to_markdown(_doc())
    assert "## 1. Results" in md and "## Appendix" in md
    assert "![x](figures/x.png)" in md and "*Figure 1. A caption*" in md
    assert "](#1-results)" in md  # internal link follows GitHub's anchor for the numbered heading


def test_latex_structure():
    tex = to_latex(_doc())
    assert tex.count(r"\begin{document}") == 1 and tex.rstrip().endswith(r"\end{document}")
    assert r"\section{Results}\label{sec:results}" in tex
    assert r"\section*{Appendix}" in tex
    assert r"\includegraphics[width=1\linewidth]{figures/x.pdf}" in tex
    assert r"\begin{tabular}{lr}" in tex


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no LaTeX engine")
def test_pdf_builds(tmp_path):
    d = _doc()
    d.blocks = [b for b in d.blocks if not isinstance(b, Figure)]
    tex = tmp_path / "t.tex"
    tex.write_text(to_latex(d))
    pdf = build_pdf(tex)
    assert pdf.exists() and pdf.read_bytes()[:4] == b"%PDF"
    assert not (tmp_path / "t.aux").exists()
