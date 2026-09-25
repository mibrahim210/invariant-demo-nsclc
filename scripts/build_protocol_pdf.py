"""Render docs/study_protocol.md to docs/study_protocol.pdf with reproducible bytes.

    uv run --with reportlab==4.4.4 python scripts/build_protocol_pdf.py
    uv run --with reportlab==4.4.4 python scripts/build_protocol_pdf.py --check

Supports the Markdown subset the protocol uses: #/## headings, paragraphs, "- " bullets,
**bold** and `code`. --check rebuilds in memory and fails if the committed PDF differs.
"""
import argparse
import html
import io
import re
import sys
from pathlib import Path

from reportlab import rl_config
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

rl_config.invariant = 1  # fixed IDs and timestamps -> identical bytes on every build

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "study_protocol.md"
OUT = ROOT / "docs" / "study_protocol.pdf"

base = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13.5,
                      alignment=TA_LEFT, spaceAfter=6)
H1 = ParagraphStyle("h1", parent=base["Title"], fontName="Helvetica-Bold", fontSize=16, leading=20,
                    alignment=TA_LEFT, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=15,
                    spaceBefore=8, spaceAfter=4)
META = ParagraphStyle("meta", parent=BODY, fontSize=8.5, textColor="#555555", spaceAfter=10)


def inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)


def story_from_markdown(md: str) -> list:
    story, bullets, para = [], [], []

    def flush():
        if para:
            style = META if len(story) == 1 else BODY
            story.append(Paragraph(inline(" ".join(para)), style))
            para.clear()
        if bullets:
            story.append(ListFlowable([ListItem(Paragraph(inline(b), BODY), leftIndent=12) for b in bullets],
                                      bulletType="bullet", start="\u2022", leftIndent=12, bulletFontSize=8))
            story.append(Spacer(1, 3))
            bullets.clear()

    for line in md.splitlines():
        if line.startswith("# "):
            flush(); story.append(Paragraph(inline(line[2:]), H1))
        elif line.startswith("## "):
            flush(); story.append(Paragraph(inline(line[3:]), H2))
        elif line.startswith("- "):
            if para:
                flush()
            bullets.append(line[2:])
        elif not line.strip():
            flush()
        else:
            para.append(line.strip())
    flush()
    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillGray(0.4)
    canvas.drawString(18 * mm, 10 * mm, "invariant-demo-nsclc | Study protocol v2.0 | synthetic data only")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build() -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=18 * mm, title="Study protocol - synthetic NSCLC-inspired imaging phenotype model",
                            author="Invariant project team", creator="scripts/build_protocol_pdf.py")
    doc.build(story_from_markdown(SRC.read_text(encoding="utf-8")), onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    pdf = build()
    if args.check:
        if not OUT.exists() or OUT.read_bytes() != pdf:
            sys.exit("MISMATCH: docs/study_protocol.pdf does not match docs/study_protocol.md; rebuild it.")
        print("OK: PDF matches its Markdown source.")
    else:
        OUT.write_bytes(pdf)
        print(f"Wrote {OUT.relative_to(ROOT)} ({len(pdf)} bytes)")


if __name__ == "__main__":
    main()
