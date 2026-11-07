"""Rebuild the raw TrailSupply policies: uv run python scripts/build_sample_documents.py.

Uses only canonical source sentences. PDF dates, DOCX properties and ZIP timestamps
are fixed so repeated runs with the locked dependencies produce identical bytes.
Bundled Liberation Sans fonts (SIL OFL 1.1) make PDF generation independent of the OS.
"""

import argparse
from datetime import datetime
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from citarilo.corpus import CorpusDocument, load_corpus

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/trailsupply-raw"
FONTS = Path(__file__).resolve().parent / "fonts"
DISCLAIMER = "Fiktivní testovací dokument – TrailSupply neexistuje"


def build_pdf(document: CorpusDocument) -> None:
    pdfmetrics.registerFont(TTFont("Liberation", str(FONTS / "LiberationSans-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("Liberation-Bold", str(FONTS / "LiberationSans-Bold.ttf")))
    body = ParagraphStyle(
        "Policy",
        fontName="Liberation",
        fontSize=11,
        leading=17,
        spaceAfter=13,
        alignment=TA_LEFT,
        splitLongWords=False,
        bulletFontName="Liberation",
        bulletFontSize=11,
        bulletIndent=-12,
    )
    title = ParagraphStyle(
        "PolicyTitle",
        parent=body,
        fontName="Liberation-Bold",
        fontSize=20,
        leading=25,
        spaceAfter=25,
        keepWithNext=True,
    )
    heading = ParagraphStyle(
        "PolicySection",
        parent=body,
        fontName="Liberation-Bold",
        fontSize=14,
        leading=19,
        spaceBefore=12,
        spaceAfter=11,
        keepWithNext=True,
    )

    def furniture(canvas, doc):
        canvas.setFillColor(colors.HexColor("#173e35"))
        canvas.setFont("Liberation-Bold", 12)
        canvas.drawString(54, A4[1] - 34, "TrailSupply")
        canvas.setStrokeColor(colors.HexColor("#b7c7c1"))
        canvas.line(54, A4[1] - 45, A4[0] - 54, A4[1] - 45)
        canvas.setFillColor(colors.HexColor("#52635d"))
        canvas.setFont("Liberation", 9)
        canvas.drawString(54, 34, DISCLAIMER)
        canvas.drawRightString(A4[0] - 54, 34, f"Strana {doc.page}")

    story = [Paragraph(escape(document.title), title)]
    for index, section in enumerate(document.sections):
        if index == 3:
            story.append(PageBreak())
        story.append(Paragraph(escape(section.heading), heading))
        if index == 1:
            for sentence in section.sentences:
                story.append(Paragraph(escape(sentence.text), body, bulletText="•"))
        else:
            story.append(Paragraph(" ".join(escape(s.text) for s in section.sentences), body))
        story.append(Spacer(1, 5))
    SimpleDocTemplate(
        str(OUTPUT / f"{document.doc_id}.pdf"),
        pagesize=A4,
        leftMargin=66,
        rightMargin=60,
        topMargin=76,
        bottomMargin=65,
        title=document.title,
        author="TrailSupply",
        invariant=1,
    ).build(story, onFirstPage=furniture, onLaterPages=furniture)


def build_docx(document: CorpusDocument) -> None:
    doc = Document()
    page = doc.sections[0]
    page.page_width, page.page_height = Inches(8.5), Inches(11)
    page.top_margin = page.bottom_margin = Inches(0.8)
    page.left_margin = page.right_margin = Inches(0.85)
    for name, size in (("Normal", 11), ("Heading 1", 21), ("Heading 2", 14)):
        style = doc.styles[name]
        style.font.name = "Liberation Sans"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_after = Pt(10)
        style.paragraph_format.line_spacing = 1.2
    doc.styles["Heading 2"].paragraph_format.space_before = Pt(14)
    header = page.header.paragraphs[0]
    run = header.add_run("TrailSupply")
    run.bold = True
    run.font.name = "Liberation Sans"
    run.font.size = Pt(11)
    footer = page.footer.paragraphs[0]
    run = footer.add_run(DISCLAIMER + "  |  ")
    run.font.size = Pt(8)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    doc.add_heading(document.title, level=1)
    for index, section in enumerate(document.sections):
        doc.add_heading(section.heading, level=2)
        if index == 1:
            for sentence in section.sentences:
                doc.add_paragraph(sentence.text, style="List Bullet")
        else:
            doc.add_paragraph(" ".join(s.text for s in section.sentences))
    doc.core_properties.title = document.title
    doc.core_properties.author = "TrailSupply"
    doc.core_properties.created = doc.core_properties.modified = datetime(2026, 3, 1)
    stream = BytesIO()
    doc.save(stream)
    stream.seek(0)
    with ZipFile(stream) as source, ZipFile(OUTPUT / f"{document.doc_id}.docx", "w") as target:
        for member in source.infolist():
            info = ZipInfo(member.filename, date_time=(2026, 3, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            target.writestr(info, source.read(member))


def build_markdown(document: CorpusDocument) -> None:
    lines = [f"# {document.title}", "", f"> {DISCLAIMER}", ""]
    for section in document.sections:
        lines.extend([f"## {section.heading}", "", " ".join(s.text for s in section.sentences), ""])
    (OUTPUT / f"{document.doc_id}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    OUTPUT = parser.parse_args().out
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for document in load_corpus(ROOT / "data/trailsupply").documents:
        if document.doc_id in {"shipping", "returns", "gear-rental"}:
            build_pdf(document)
        elif document.doc_id in {"payments", "warranty"}:
            build_docx(document)
        else:
            build_markdown(document)


if __name__ == "__main__":
    main()
