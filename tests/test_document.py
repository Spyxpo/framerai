"""Document ingestion tests: reading order, page identity, and optional readers.

A PDF's content stream lists text in whatever order the producer emitted it, so
a two-column page arrives interleaved and reads as nonsense unless the order is
recovered. These tests pin that recovery, the page markers that make "on page
four" answerable, and the two places the module is deliberately optional: the
PDF reader and the page rasteriser both fail with a reason a caller can act on
rather than returning something empty that reads as a blank page.
"""

import json

import pytest

from model.data import iter_document_records, iter_text_records
from model.document import (
    DOC_END_TOKEN,
    DOC_TOKEN,
    PAGE_TOKEN,
    Document,
    DocumentError,
    Page,
    TextSpan,
    available_raster_backends,
    detect_columns,
    needs_raster,
    order_spans,
    rasterize_page,
    read_document,
    register_raster_backend,
    spans_to_text,
)
from model.tokenizer import FramerTokenizer

PAGE_WIDTH = 612.0


def two_column_spans():
    """Spans as a producer emits them: across the columns, not down them."""
    return [
        TextSpan("left one", 50.0, 700.0, 10.0),
        TextSpan("right one", 330.0, 700.0, 10.0),
        TextSpan("left two", 50.0, 680.0, 10.0),
        TextSpan("right two", 330.0, 680.0, 10.0),
    ]


def test_two_column_page_reads_down_each_column():
    text = spans_to_text(two_column_spans(), PAGE_WIDTH)
    assert text.splitlines() == ["left one", "left two", "right one", "right two"]


def test_single_column_page_reports_no_boundary():
    spans = [TextSpan(f"line {i}", 50.0, 700.0 - 20.0 * i, 10.0) for i in range(6)]
    assert detect_columns(spans, PAGE_WIDTH) == []
    assert spans_to_text(spans, PAGE_WIDTH).splitlines()[0] == "line 0"


def test_column_boundary_falls_between_the_columns():
    boundaries = detect_columns(two_column_spans(), PAGE_WIDTH)
    assert len(boundaries) == 1
    assert 100.0 < boundaries[0] < 330.0


def test_spans_on_one_line_join_left_to_right():
    spans = [
        TextSpan("world", 120.0, 700.0, 10.0),
        TextSpan("hello", 50.0, 700.0, 10.0),
    ]
    assert spans_to_text(spans, PAGE_WIDTH) == "hello world"


def test_order_is_stable_for_empty_input():
    assert order_spans([], PAGE_WIDTH) == []
    assert spans_to_text([], PAGE_WIDTH) == ""


def test_page_markers_carry_page_numbers():
    doc = Document(path="x.pdf", pages=[Page(1, "alpha"), Page(2, "beta")])
    text = doc.to_text()
    assert text.startswith(DOC_TOKEN)
    assert text.endswith(DOC_END_TOKEN)
    assert f"{PAGE_TOKEN}1" in text and f"{PAGE_TOKEN}2" in text
    assert doc.to_text(page_markers=False) == "alpha\n\nbeta"


def test_max_pages_truncates_without_losing_the_end_marker():
    doc = Document(path="x.pdf", pages=[Page(i, f"page {i}") for i in range(1, 6)])
    text = doc.to_text(max_pages=2)
    assert f"{PAGE_TOKEN}3" not in text
    assert text.endswith(DOC_END_TOKEN)


def test_a_page_with_no_text_layer_is_a_scan():
    scanned, typed = Page(1, ""), Page(2, "a page of real extracted text here")
    assert needs_raster(scanned) and not needs_raster(typed)
    doc = Document(path="x.pdf", pages=[scanned, typed])
    assert [p.number for p in doc.scanned_pages] == [1]


def test_document_markers_survive_the_tokenizer_without_shifting_the_vocabulary():
    tok = FramerTokenizer(vocab_size=400)
    assert tok.first_merge_id == 287
    for marker in (DOC_TOKEN, DOC_END_TOKEN, PAGE_TOKEN):
        assert marker in tok.reserved_tokens
    ids = tok.encode(f"{DOC_TOKEN}{PAGE_TOKEN}1 hi{DOC_END_TOKEN}", add_special=False)
    assert tok.decode(ids) == f"{DOC_TOKEN}{PAGE_TOKEN}1 hi{DOC_END_TOKEN}"


def test_text_file_reads_as_a_single_page(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("just some text")
    doc = read_document(str(path))
    assert len(doc) == 1 and doc.pages[0].text == "just some text"


def test_unsupported_type_names_itself():
    with pytest.raises(DocumentError, match="unsupported document type"):
        read_document("archive.zip")


def test_missing_pdf_reader_is_reported_not_crashed(tmp_path, monkeypatch):
    import model.document as document

    def refuse():
        raise DocumentError("Reading PDFs needs the 'pypdf' package.")

    monkeypatch.setattr(document, "_load_pdf_reader", refuse)
    with pytest.raises(DocumentError, match="pypdf"):
        document.read_pdf(str(tmp_path / "absent.pdf"))


def test_rasterising_without_a_backend_explains_itself():
    import model.document as document

    monkey = dict(document._RASTER_BACKENDS)
    document._RASTER_BACKENDS.clear()
    try:
        with pytest.raises(DocumentError, match="no page raster backend"):
            rasterize_page("x.pdf", 1)
    finally:
        document._RASTER_BACKENDS.update(monkey)


def test_a_registered_raster_backend_is_used():
    import model.document as document

    calls = []
    register_raster_backend("stub", lambda path, page, dpi: calls.append((path, page, dpi)) or "img")
    try:
        assert "stub" in available_raster_backends()
        assert rasterize_page("x.pdf", 3, dpi=150, backend="stub") == "img"
        assert calls == [("x.pdf", 3, 150)]
        with pytest.raises(DocumentError, match="not registered"):
            rasterize_page("x.pdf", 1, backend="absent")
    finally:
        document._RASTER_BACKENDS.pop("stub", None)


def test_document_records_reach_the_training_corpus(tmp_path):
    (tmp_path / "doc.txt").write_text("the corpus text")
    (tmp_path / "corpus.jsonl").write_text(
        json.dumps({"document": "doc.txt"}) + "\n"
        + json.dumps({"document": "doc.txt", "text": "given verbatim"}) + "\n"
    )

    from_documents = list(iter_document_records(str(tmp_path)))
    assert len(from_documents) == 1, "a record carrying its own text must not be read twice"
    assert "the corpus text" in from_documents[0]

    everything = list(iter_text_records(str(tmp_path)))
    assert "given verbatim" in everything
    assert any("the corpus text" in text for text in everything)


def test_an_unreadable_document_does_not_stop_the_corpus(tmp_path, capsys):
    (tmp_path / "good.txt").write_text("readable content")
    (tmp_path / "corpus.jsonl").write_text(
        json.dumps({"document": "broken.zip"}) + "\n"
        + json.dumps({"document": "absent.pdf"}) + "\n"
        + json.dumps({"document": "good.txt"}) + "\n"
    )
    (tmp_path / "broken.zip").write_bytes(b"not a document")

    texts = list(iter_document_records(str(tmp_path)))
    assert len(texts) == 1 and "readable content" in texts[0]
    assert "skipping" in capsys.readouterr().out


class _StubConfig:
    image_size = 8


class _StubGen:
    model = type("M", (), {"config": _StubConfig()})()


def test_attachments_contribute_document_text(tmp_path):
    from model.serve import _read_attachments

    doc = tmp_path / "brief.txt"
    doc.write_text("the attached content")

    image, documents = _read_attachments(
        _StubGen(), [{"kind": "document", "path": str(doc)}]
    )
    assert image is None
    assert len(documents) == 1 and "the attached content" in documents[0]


def test_an_unreadable_attachment_does_not_lose_the_turn(tmp_path):
    from model.serve import _read_attachments

    good = tmp_path / "good.txt"
    good.write_text("readable")
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a document")

    _, documents = _read_attachments(
        _StubGen(),
        [
            {"kind": "document", "path": str(bad)},
            {"kind": "document", "path": str(good)},
        ],
    )
    assert len(documents) == 2
    assert "could not be read" in documents[0]
    assert "readable" in documents[1]


def test_attachments_without_paths_are_skipped():
    from model.serve import _read_attachments

    image, documents = _read_attachments(_StubGen(), [{"kind": "image"}, {}])
    assert image is None and documents == []
    assert _read_attachments(_StubGen(), None) == (None, [])
