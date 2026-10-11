"""Office files, web pages, data and code as documents, and choosing passages.

The archives here are built in the test, so each case states exactly what the
reader is given, including the hostile ones.
"""

import json
import zipfile

import pytest

from model.document import DOC_END_TOKEN, DocumentError, read_document
from model.document_formats import MAX_MEMBER_BYTES, handles
from model.retrieval import OMISSION, bm25_scores, select_relevant, split_passages
from model.serve import _focus, _read_attachments

W_NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
A_NS = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
S_NS = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
R_NS = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'


def _zip(path, members):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return str(path)


def _docx(tmp_path, body):
    return _zip(tmp_path / "a.docx", {"word/document.xml": f"<w:document {W_NS}><w:body>{body}</w:body></w:document>"})


def _p(text, style=None):
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{ppr}<w:r><w:t>{text}</w:t></w:r></w:p>"


# ── Office ────────────────────────────────────────────────────────────────

def test_docx_keeps_headings_paragraphs_and_tables(tmp_path):
    table = (
        "<w:tbl><w:tr><w:tc>" + _p("Region") + "</w:tc><w:tc>" + _p("Sales") + "</w:tc></w:tr>"
        "<w:tr><w:tc>" + _p("North") + "</w:tc><w:tc>" + _p("42") + "</w:tc></w:tr></w:tbl>"
    )
    doc = read_document(_docx(tmp_path, _p("Results", "Heading1") + _p("Sales rose.") + table))
    text = doc.pages[0].text
    assert "# Results" in text and "Sales rose." in text
    assert "| North | 42 |" in text


def test_pptx_reads_one_page_per_slide_in_slide_order(tmp_path):
    def slide(text):
        return f"<p:sld {A_NS} xmlns:p='p'><a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:sld>"

    path = _zip(tmp_path / "deck.pptx", {
        "ppt/slides/slide10.xml": slide("tenth"),
        "ppt/slides/slide2.xml": slide("second"),
        "ppt/slides/slide1.xml": slide("first"),
    })
    doc = read_document(path)
    assert [p.text for p in doc.pages] == ["first", "second", "tenth"]
    assert len(read_document(path, max_pages=2)) == 2
    assert read_document(path, max_pages=2).truncated


def test_xlsx_resolves_shared_strings_and_sheet_names(tmp_path):
    sheet = (
        f"<worksheet {S_NS}><sheetData>"
        '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
        '<row r="2"><c r="A2" t="s"><v>2</v></c><c r="C2"><v>3.5</v></c></row>'
        "</sheetData></worksheet>"
    )
    path = _zip(tmp_path / "book.xlsx", {
        "xl/sharedStrings.xml": f"<sst {S_NS}><si><t>item</t></si><si><t>cost</t></si><si><t>tea</t></si></sst>",
        "xl/workbook.xml": f'<workbook {S_NS} {R_NS}><sheets><sheet name="Budget" r:id="rId1"/></sheets></workbook>',
        "xl/worksheets/sheet1.xml": sheet,
    })
    text = read_document(path).pages[0].text
    assert text.startswith("## Budget")
    assert "| item | cost |" in text
    assert "| tea |  | 3.5 |" in text


def test_a_doctype_is_refused(tmp_path):
    bomb = (
        '<?xml version="1.0"?><!DOCTYPE l [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;">]>'
        f"<w:document {W_NS}><w:body>{_p('&b;')}</w:body></w:document>"
    )
    path = _zip(tmp_path / "bomb.docx", {"word/document.xml": bomb})
    with pytest.raises(DocumentError, match="DOCTYPE"):
        read_document(path)


def test_a_member_that_inflates_too_far_is_refused(tmp_path):
    path = _zip(tmp_path / "big.docx", {"word/document.xml": b"0" * (MAX_MEMBER_BYTES + 1)})
    with pytest.raises(DocumentError, match="inflates"):
        read_document(path)


def test_a_file_that_is_not_an_archive_is_refused(tmp_path):
    path = tmp_path / "fake.docx"
    path.write_text("not a zip")
    with pytest.raises(DocumentError, match="not a readable office file"):
        read_document(str(path))


def test_the_character_budget_applies(tmp_path):
    doc = read_document(_docx(tmp_path, _p("x" * 500)), max_chars=100)
    assert doc.truncated and len(doc.to_text()) <= 140


# ── Text formats ──────────────────────────────────────────────────────────

def test_html_keeps_visible_text_only(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><title>Guide</title><style>p{}</style></head><body>"
        "<h2>Setup</h2><script>alert(1)</script><p>Run the installer.</p>"
        "<ul><li>one</li><li>two</li></ul></body></html>"
    )
    doc = read_document(str(path))
    assert doc.title == "Guide"
    text = doc.pages[0].text
    assert "## Setup" in text and "Run the installer." in text and "- one" in text
    assert "alert" not in text and "p{}" not in text


def test_csv_becomes_a_table(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text('name,score\n"Lee, A",9\n')
    assert "| Lee, A | 9 |" in read_document(str(path)).pages[0].text


def test_json_is_pretty_printed(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"a": {"b": 1}}))
    assert '"b": 1' in read_document(str(path)).pages[0].text


def test_code_is_fenced_with_its_language_and_name(tmp_path):
    path = tmp_path / "main.rs"
    path.write_text("fn main() {}\n")
    text = read_document(str(path)).pages[0].text
    assert text.startswith("```rust main.rs") and text.endswith("```")


def test_handles_names_the_new_types():
    for name in ("a.docx", "b.PPTX", "c.xlsx", "d.csv", "e.html", "f.json", "g.py", "h.go"):
        assert handles(name)
    assert not handles("x.exe")


# ── Retrieval ─────────────────────────────────────────────────────────────

def _long_document():
    filler = [f"Chapter {i}. The committee discussed routine matters and minutes." for i in range(60)]
    filler[47] = "Chapter 47. The reactor coolant pump failed on 3 March, halting output."
    return "\n\n".join(filler)


def test_passages_respect_the_size():
    passages = split_passages("a" * 3000 + "\n\nshort", size=1000)
    assert all(len(p) <= 1000 for p in passages)
    assert passages[-1] == "short"


def test_bm25_ranks_the_matching_passage_first():
    passages = split_passages(_long_document(), size=100)
    scores = bm25_scores(passages, "when did the coolant pump fail")
    best = max(range(len(passages)), key=scores.__getitem__)
    assert "coolant pump" in passages[best]


def test_selection_finds_the_passage_a_cut_would_lose():
    text = _long_document()
    budget = 600
    assert "coolant" not in text[:budget]
    excerpt, selected = select_relevant(text, "when did the coolant pump fail", budget)
    assert selected and "3 March" in excerpt
    assert len(excerpt) <= budget
    assert OMISSION in excerpt


def test_selection_keeps_document_order():
    text = "\n\n".join(f"alpha {i} beta" if i in (2, 9) else f"filler {i}" for i in range(12))
    excerpt, _ = select_relevant(text, "alpha beta", 200)
    assert excerpt.index("alpha 2") < excerpt.index("alpha 9")


def test_no_shared_words_falls_back_to_the_opening():
    excerpt, selected = select_relevant(_long_document(), "zzz qqq", 200)
    assert not selected and excerpt == _long_document()[:200]


def test_focus_closes_the_document_marker():
    text = "<doc>\n<page>1\n" + _long_document() + "\n" + DOC_END_TOKEN
    focused, excerpted = _focus(text, "coolant pump", 500)
    assert excerpted and focused.rstrip().endswith(DOC_END_TOKEN)


class _Config:
    max_seq_len = 200  # 600 characters of attachment budget


class _Gen:
    class model:
        config = _Config()


def test_an_attachment_is_excerpted_for_the_question(tmp_path):
    path = tmp_path / "report.txt"
    path.write_text(_long_document())
    _, docs = _read_attachments(
        _Gen(), [{"kind": "document", "path": str(path)}], query="coolant pump failure"
    )
    assert "3 March" in docs[0]
    assert "[document excerpted" in docs[0]


def test_without_a_question_an_attachment_is_cut_as_before(tmp_path):
    path = tmp_path / "report.txt"
    path.write_text(_long_document())
    _, docs = _read_attachments(_Gen(), [{"kind": "document", "path": str(path)}])
    assert "3 March" not in docs[0]
    assert "[document truncated" in docs[0]
