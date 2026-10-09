"""Document ingestion tests: reading order, page identity, and optional readers.

A PDF's content stream lists text in whatever order the producer emitted it, so
a two-column page arrives interleaved and reads as nonsense unless the order is
recovered. These tests pin that recovery, the page markers that make "on page
four" answerable, and the two places the module is deliberately optional: the
PDF reader and the page rasteriser both fail with a reason a caller can act on
rather than returning something empty that reads as a blank page.
"""

import json
import zlib

import pytest

from model.data import iter_document_records, iter_text_records
from model.document import (
    DOC_END_TOKEN,
    DOC_TOKEN,
    DOCUMENT_CONTENT_LIMIT,
    PAGE_CONTENT_LIMIT,
    PAGE_TOKEN,
    PAGE_TREE_ENTRY_LIMIT,
    Document,
    DocumentError,
    Page,
    TextSpan,
    _page_tree_limit,
    available_raster_backends,
    detect_columns,
    needs_raster,
    order_spans,
    rasterize_page,
    read_document,
    read_pdf,
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


# ── Limits on how much of a long document is read (Issue #392) ────────────
#
# A document is read on the worker's only thread, and tokenizing what comes out
# of it costs far more than parsing it, so a limit has to stop the reading and
# not trim the result afterwards. The PDF reader is an optional extra, so these
# tests stand in for it with pages that record when they are opened. What a
# limit saved is then an exact set of page numbers, and no test depends on time.


class _StubPage:
    mediabox = type("Box", (), {"width": 612.0, "height": 792.0})()

    def __init__(self, index, text, opened):
        self.index, self.text, self.opened = index, text, opened

    def extract_text(self, visitor_text=None, **operand_visitors):
        self.opened.add(self.index)
        if visitor_text is not None:
            visitor_text(self.text, None, [1, 0, 0, 1, 50.0, 700.0], {}, 10.0)
            return ""
        return self.text


def _stub_pdf(monkeypatch, texts):
    """Make read_pdf see a PDF with these page texts. Returns the pages it opened."""
    import model.document as document

    opened = set()
    pages = [_StubPage(i, text, opened) for i, text in enumerate(texts)]

    class Reader:
        metadata = {}

        def __init__(self, path):
            self.pages = pages

    monkeypatch.setattr(document, "_load_pdf_reader", lambda: Reader)
    return opened


def _rendered_limit(max_chars):
    """Most characters to_text may write for a budget: the budget plus its own markers."""
    return max_chars + len(DOC_TOKEN) + len(DOC_END_TOKEN) + 1


def test_pages_past_the_character_budget_are_never_opened(monkeypatch):
    opened = _stub_pdf(monkeypatch, ["x" * 100] * 10)

    doc = read_pdf("stub.pdf", max_chars=350)

    assert doc.truncated
    assert opened == {0, 1, 2, 3}, "reading has to stop when the budget is spent, not after"
    assert len(doc) == 4
    # The page that crosses the line is cut to fit, not dropped and not kept whole.
    assert 0 < len(doc.pages[3].text) < 100
    assert len(doc.to_text()) <= _rendered_limit(350)


def test_blank_pages_spend_the_budget_too(monkeypatch):
    opened = _stub_pdf(monkeypatch, [""] * 1000)

    doc = read_pdf("stub.pdf", max_chars=100)

    assert doc.truncated
    assert len(opened) < 20, "a run of blank pages must not be free to read"
    assert len(doc) == len(opened)


def test_the_page_limit_stops_reading_and_says_so(monkeypatch):
    opened = _stub_pdf(monkeypatch, [f"page {i}" for i in range(5)])

    doc = read_pdf("stub.pdf", max_pages=2)

    assert [p.text for p in doc.pages] == ["page 0", "page 1"]
    assert opened == {0, 1}
    assert doc.truncated


def test_a_document_inside_its_limits_is_not_marked_truncated(monkeypatch):
    _stub_pdf(monkeypatch, ["alpha", "beta"])
    for limits in (
        {},
        {"max_pages": 2},
        {"max_chars": 10_000},
        {"max_pages": 2, "max_chars": 10_000},
    ):
        doc = read_pdf("stub.pdf", **limits)
        assert [p.text for p in doc.pages] == ["alpha", "beta"], limits
        assert not doc.truncated, limits


def test_a_budget_spent_exactly_on_the_last_page_is_not_truncation(monkeypatch):
    _stub_pdf(monkeypatch, ["0123456789", "0123456789"])
    exact = 2 * (10 + len(PAGE_TOKEN) + 1 + 2)  # two ten-character pages and their markers

    whole = read_pdf("stub.pdf", max_chars=exact)
    assert not whole.truncated
    assert [len(p.text) for p in whole.pages] == [10, 10]

    one_short = read_pdf("stub.pdf", max_chars=exact - 1)
    assert one_short.truncated
    assert [len(p.text) for p in one_short.pages] == [10, 9]


def test_a_text_file_is_cut_to_the_budget(tmp_path):
    path = tmp_path / "long.txt"
    path.write_text("y" * 5000)

    doc = read_document(str(path), max_chars=200)

    assert doc.truncated
    assert 0 < len(doc.pages[0].text) < 200
    assert len(doc.to_text()) <= _rendered_limit(200)


def test_a_text_file_is_read_only_as_far_as_the_budget(tmp_path, monkeypatch):
    import model.document as document

    path = tmp_path / "long.txt"
    path.write_text("y" * 5000)
    sizes = []

    class Spy:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.handle.close()

        def read(self, size=-1):
            sizes.append(size)
            return self.handle.read(size)

    real_open = open
    monkeypatch.setattr(document, "open", lambda *a, **k: Spy(real_open(*a, **k)), raising=False)

    read_document(str(path), max_chars=200)

    assert len(sizes) == 1 and 0 < sizes[0] <= 200, "the rest of the file must not be read"


def test_a_text_file_inside_the_budget_is_read_whole(tmp_path):
    path = tmp_path / "short.txt"
    path.write_text("a short note")

    doc = read_document(str(path), max_chars=200)

    assert doc.pages[0].text == "a short note"
    assert not doc.truncated


class _WindowConfig:
    image_size = 8

    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len


class _WindowGen:
    def __init__(self, max_seq_len):
        self.model = type("M", (), {"config": _WindowConfig(max_seq_len)})()


def test_the_character_budget_follows_the_models_window():
    from model.serve import ATTACHMENT_CHARS_PER_TOKEN, _attachment_chars

    assert _attachment_chars(_WindowConfig(1000)) == 1000 * ATTACHMENT_CHARS_PER_TOKEN
    assert _attachment_chars(_WindowConfig(0)) is None
    assert _attachment_chars(_StubConfig()) is None, "a config with no window has nothing to size against"


def test_a_document_that_fits_the_window_is_read_whole(tmp_path):
    from model.serve import _read_attachments

    note = tmp_path / "note.txt"
    note.write_text("a short note")

    _, documents = _read_attachments(_WindowGen(100), [{"kind": "document", "path": str(note)}])

    assert len(documents) == 1 and "a short note" in documents[0]
    assert "truncated" not in documents[0] and "skipped" not in documents[0]


def test_documents_share_one_character_budget(tmp_path):
    from model.serve import ATTACHMENT_CHARS_PER_TOKEN, _read_attachments

    window = 100
    budget = window * ATTACHMENT_CHARS_PER_TOKEN
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    first.write_text("a" * (budget * 4))
    second.write_text("a note that no longer fits")

    _, documents = _read_attachments(
        _WindowGen(window),
        [{"kind": "document", "path": str(first)}, {"kind": "document", "path": str(second)}],
    )

    assert len(documents) == 2
    text, _, note = documents[0].partition("\n[document truncated")
    assert note, "text that was cut has to say so"
    assert 0 < len(text) <= _rendered_limit(budget)
    assert "skipped" in documents[1] and "a note" not in documents[1]


def test_without_a_window_the_text_is_not_cut(tmp_path):
    from model.serve import _read_attachments

    path = tmp_path / "long.txt"
    path.write_text("b" * 50_000)

    _, documents = _read_attachments(_StubGen(), [{"kind": "document", "path": str(path)}])

    assert len(documents[0]) > 50_000
    assert "truncated" not in documents[0]


def test_the_page_ceiling_is_shared_across_a_requests_documents(monkeypatch):
    import model.serve as serve

    opened = _stub_pdf(monkeypatch, [f"page {i}" for i in range(10)])
    monkeypatch.setattr(serve, "MAX_ATTACHMENT_PAGES", 3)
    ref = {"kind": "document", "path": "long.pdf"}

    _, documents = serve._read_attachments(_StubGen(), [ref, ref])

    assert opened == {0, 1, 2}, "pages past the ceiling are never opened, in this document or the next"
    assert f"{PAGE_TOKEN}3" in documents[0] and f"{PAGE_TOKEN}4" not in documents[0]
    assert "truncated" in documents[0]
    assert "skipped" in documents[1]


# ---------------------------------------------------------------------------
# Issue #443: touching ``reader.pages`` at all makes pypdf flatten the whole
# page tree before max_pages ever runs, so a request for a handful of pages
# used to cost the same as a file's entire declared page count regardless.
# Every test below needs pypdf's own page-tree flattening, which the stub
# reader above cannot exercise, so each calls pytest.importorskip for itself
# rather than at module level - these are the only tests in this file that
# are not optional the way the rest of the module is, and the existing tests
# above must keep running without pypdf installed, exactly as before.
# ---------------------------------------------------------------------------

def _real_pdf(tmp_path, pages, name="real.pdf"):
    """A real PDF with this many real pages, built fast - never a bomb-sized fixture."""
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    path = tmp_path / name
    with open(path, "wb") as handle:
        writer.write(handle)
    return str(path)


def test_443_the_page_tree_limit_never_exceeds_the_flat_ceiling():
    assert _page_tree_limit(100_000) == PAGE_TREE_ENTRY_LIMIT
    assert _page_tree_limit(1_000_000) == PAGE_TREE_ENTRY_LIMIT, "a larger pypdf default must not raise it"
    assert _page_tree_limit(5) == 5, "a lower current default (e.g. one a test has scoped down) is kept"


def test_443_a_page_tree_bigger_than_the_limit_is_a_document_error(tmp_path):
    pypdf = pytest.importorskip("pypdf", minversion="6.18")
    path = _real_pdf(tmp_path, pages=10)

    # Scopes pypdf's own default down to 5 entries for this call only, so a
    # real but tiny (10-page) file stands in for a bomb-sized one -
    # deterministic and fast, rather than actually building and reading a
    # file with more than PAGE_TREE_ENTRY_LIMIT real pages.
    with pypdf.apply_configuration(page_tree_maximum_entries=5):
        with pytest.raises(DocumentError, match="could not be read"):
            read_pdf(path, max_pages=3, max_chars=500)


def test_443_the_page_tree_limit_does_not_leak_into_the_next_read(tmp_path):
    pypdf = pytest.importorskip("pypdf", minversion="6.18")
    path = _real_pdf(tmp_path, pages=10)

    with pypdf.apply_configuration(page_tree_maximum_entries=5):
        with pytest.raises(DocumentError):
            read_pdf(path, max_pages=3)

    # Back outside the scope, the exact same file and request must succeed:
    # the override above must not have changed anything beyond its own call.
    assert pypdf.get_configuration().page_tree_maximum_entries != 5
    doc = read_pdf(path, max_pages=3, max_chars=500)
    assert len(doc.pages) == 3
    assert doc.truncated


def test_443_a_legitimately_large_document_is_not_rejected_for_a_small_max_pages(tmp_path):
    # Unlike the bomb-sized files Issue #443 is about, a genuine report or
    # book can run to hundreds of pages, and asking for just its first few
    # must still work. (The first fix attempted here scaled the tree limit
    # down with max_pages itself, and that rejected exactly this case - see
    # PAGE_TREE_ENTRY_LIMIT's own comment for why it is a flat ceiling instead.)
    path = _real_pdf(tmp_path, pages=500)
    doc = read_pdf(path, max_pages=3, max_chars=500)
    assert len(doc.pages) == 3
    assert doc.truncated


def test_443_max_pages_still_truncates_a_real_pdf_correctly(tmp_path):
    path = _real_pdf(tmp_path, pages=5)
    doc = read_pdf(path, max_pages=2)
    assert len(doc.pages) == 2
    assert doc.truncated


def test_443_a_real_pdf_within_every_limit_is_read_whole(tmp_path):
    path = _real_pdf(tmp_path, pages=5)

    doc = read_pdf(path, max_pages=2000, max_chars=None)
    assert len(doc.pages) == 5
    assert not doc.truncated

    doc_no_limits = read_pdf(path)
    assert len(doc_no_limits.pages) == 5
    assert not doc_no_limits.truncated


def test_443_read_document_reports_an_oversized_page_tree_the_same_way(tmp_path):
    # /api/generate/document's own op handler (model/serve.py) calls
    # read_document, not read_pdf directly - confirm the DocumentError survives
    # that one extra layer of dispatch, unchanged from how it already handled
    # a file that fails to open at all.
    pypdf = pytest.importorskip("pypdf", minversion="6.18")
    path = _real_pdf(tmp_path, pages=10)

    with pypdf.apply_configuration(page_tree_maximum_entries=5):
        with pytest.raises(DocumentError, match="could not be read"):
            read_document(path, max_pages=3)


def test_443_an_oversized_attachment_does_not_lose_the_rest_of_the_turn(tmp_path):
    # Mirrors test_an_unreadable_attachment_does_not_lose_the_turn above, for
    # Issue #443's own failure mode: a page-tree-oversized PDF attachment must
    # degrade to a note, not raise out of _read_attachments and fail the whole
    # chat turn, and a good attachment listed after it must still be read.
    pypdf = pytest.importorskip("pypdf", minversion="6.18")
    from model.serve import _read_attachments

    bad = _real_pdf(tmp_path, pages=10, name="bad.pdf")
    good = tmp_path / "good.txt"
    good.write_text("readable")

    with pypdf.apply_configuration(page_tree_maximum_entries=5):
        _, documents = _read_attachments(
            _StubGen(),
            [
                {"kind": "document", "path": bad},
                {"kind": "document", "path": str(good)},
            ],
        )
    assert len(documents) == 2
    assert "could not be read" in documents[0]
    assert "readable" in documents[1]


# ---------------------------------------------------------------------------
# Issue #451: parsing a page costs one to two seconds per decoded megabyte of
# its content streams, and nothing bounded that before. A Flate stream
# compresses about a thousandfold, pages may share one, a /Contents array may
# repeat one and a form XObject is parsed again at every invocation, so a 14 KB
# PDF kept a worker parsing until its request timed out.
#
# These tests count the bytes pypdf actually parses rather than timing it, so
# what a limit saves is an exact number. They shrink the limits where a test
# would otherwise have to parse megabytes; the shipped values are used wherever
# the bound holds without parsing at all. The files are hand-written, so no PDF
# library is needed to build them, only to read them.
# ---------------------------------------------------------------------------

KiB = 1024


def _noop(size):
    """``size`` bytes of bare operators: as costly to parse as text, yielding none."""
    return b"q Q\n" * (size // 4)


def _text(size):
    """Exactly ``size`` bytes of ordinary text-showing content."""
    lines = bytearray()
    n = 0
    while len(lines) < size:
        lines += b"BT /F1 11 Tf 72 %d Td (Line %d of an ordinary page.) Tj ET\n" % (700 - (n % 60) * 11, n)
        n += 1
    return bytes(lines[: size - 1]).rsplit(b"\n", 1)[0].ljust(size, b" ")


def _content_pdf(tmp_path, pages, content, *, repeat=1, xobject=None, distinct=False, name="content.pdf"):
    """A PDF whose pages draw ``content``: one shared stream unless ``distinct``.

    ``repeat`` makes each page's /Contents an array naming the stream that many
    times; ``xobject`` adds that content as form /X0 to every page's resources.
    """
    objs = []

    def add(body):
        objs.append(body)
        return len(objs)

    def stream(data, extra=b""):
        packed = zlib.compress(data, 9)
        return add(b"<< /Length %d /Filter /FlateDecode%s >>\nstream\n" % (len(packed), extra) + packed + b"\nendstream")

    catalog, tree = add(b""), add(b"")
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    form = None
    if xobject is not None:
        form = stream(
            xobject,
            b" /Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 %d 0 R >> >>" % font,
        )
    shared = None if distinct else stream(content)
    kids = []
    for _ in range(pages):
        ref = b"%d 0 R" % (stream(content) if distinct else shared)
        contents = ref if repeat == 1 else b"[" + b" ".join([ref] * repeat) + b"]"
        xobjects = b" /XObject << /X0 %d 0 R >>" % form if form else b""
        resources = b"<< /Font << /F1 %d 0 R >>%s >>" % (font, xobjects)
        kids.append(add(
            b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] /Resources %s /Contents %s >>"
            % (tree, resources, contents)
        ))
    objs[catalog - 1] = b"<< /Type /Catalog /Pages %d 0 R >>" % tree
    objs[tree - 1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(b"%d 0 R" % k for k in kids), pages)

    body = bytearray(b"%PDF-1.7\n")
    offsets = []
    for number, obj in enumerate(objs, start=1):
        offsets.append(len(body))
        body += b"%d 0 obj\n" % number + obj + b"\nendobj\n"
    xref = len(body)
    body += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    body += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    body += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, catalog, xref)
    path = tmp_path / name
    path.write_bytes(bytes(body))
    return str(path)


@pytest.fixture
def parsed(monkeypatch):
    """Bytes of content pypdf parses during a test.

    Past ``ceiling`` the spy refuses to parse, so a reader without the limits
    fails the test at once instead of parsing for minutes first.
    """
    pytest.importorskip("pypdf", minversion="6.18")
    from pypdf.generic import ContentStream

    original = getattr(ContentStream, "_parse_content_stream", None)
    if original is None:
        pytest.skip("this pypdf parses content streams some other way")
    record = {"bytes": 0, "ceiling": None}

    def spy(self, stream):
        record["bytes"] += stream.getbuffer().nbytes
        if record["ceiling"] is not None and record["bytes"] > record["ceiling"]:
            raise RuntimeError("parsed past what the test allows")
        return original(self, stream)

    monkeypatch.setattr(ContentStream, "_parse_content_stream", spy)
    return record


def _limits(monkeypatch, page, document):
    import model.document as document_module

    monkeypatch.setattr(document_module, "PAGE_CONTENT_LIMIT", page)
    monkeypatch.setattr(document_module, "DOCUMENT_CONTENT_LIMIT", document)


def test_451_a_page_over_the_page_limit_is_never_parsed(tmp_path, parsed):
    # A 4 KB file whose one page decodes to more than the shipped page limit.
    path = _content_pdf(tmp_path, 1, _noop(PAGE_CONTENT_LIMIT + 1024 * KiB))
    parsed["ceiling"] = 0

    doc = read_pdf(path)

    assert parsed["bytes"] == 0, "the oversized page must be measured, not parsed"
    assert len(doc) == 1 and doc.pages[0].text == "", "the page is kept, empty, as a damaged one is"
    assert doc.truncated


def test_451_a_form_xobject_over_the_page_limit_is_never_parsed(tmp_path, parsed):
    page = b"q /X0 Do Q\n"
    path = _content_pdf(tmp_path, 1, page, xobject=_noop(PAGE_CONTENT_LIMIT + 1024 * KiB))
    parsed["ceiling"] = len(page)

    doc = read_pdf(path)

    assert parsed["bytes"] == len(page), "only the page's own few bytes are parsed"
    assert doc.truncated


def test_451_many_small_pages_cannot_add_up_past_the_document_budget(tmp_path, parsed, monkeypatch):
    # The audit's file, scaled down: 80 pages sharing one stream, each page well
    # inside the page limit. Only the document budget can stop it.
    _limits(monkeypatch, page=64 * KiB, document=256 * KiB)
    path = _content_pdf(tmp_path, 80, _noop(48 * KiB))
    parsed["ceiling"] = 256 * KiB

    doc = read_pdf(path)

    assert parsed["bytes"] <= 256 * KiB
    assert len(doc) == 5, "five 48 KiB pages fit in 256 KiB, the sixth is never opened"
    assert doc.truncated


def test_451_a_contents_array_cannot_repeat_a_stream_past_the_page_limit(tmp_path, parsed, monkeypatch):
    # 20 x 16 KiB is far under pypdf's own cap on a concatenated array, so
    # pypdf alone would parse all of it.
    _limits(monkeypatch, page=64 * KiB, document=DOCUMENT_CONTENT_LIMIT)
    path = _content_pdf(tmp_path, 1, _noop(16 * KiB), repeat=20)
    parsed["ceiling"] = 0

    doc = read_pdf(path)

    assert parsed["bytes"] == 0
    assert doc.truncated


def test_451_a_form_xobject_is_charged_at_every_invocation(tmp_path, parsed, monkeypatch):
    # A tiny page invoking one small form hundreds of times; pypdf allows 5000.
    _limits(monkeypatch, page=64 * KiB, document=DOCUMENT_CONTENT_LIMIT)
    page = b"q /X0 Do Q\n" * 500
    path = _content_pdf(tmp_path, 1, page, xobject=_noop(16 * KiB))
    parsed["ceiling"] = 64 * KiB

    doc = read_pdf(path)

    assert parsed["bytes"] <= 64 * KiB
    assert doc.truncated


def test_451_a_page_without_text_is_parsed_once(tmp_path, parsed):
    # Reading used to run a second extraction for a page whose first yielded no
    # positioned text, parsing the same content twice.
    content = _noop(20 * KiB)
    path = _content_pdf(tmp_path, 1, content)

    doc = read_pdf(path)

    assert parsed["bytes"] == len(content)
    assert not doc.truncated


def test_451_a_page_exactly_at_the_page_limit_is_read_whole(tmp_path, parsed, monkeypatch):
    _limits(monkeypatch, page=64 * KiB, document=DOCUMENT_CONTENT_LIMIT)

    at_limit = read_pdf(_content_pdf(tmp_path, 1, _text(64 * KiB), name="at.pdf"))
    assert "Line 0 of an ordinary page." in at_limit.pages[0].text
    assert not at_limit.truncated

    over = read_pdf(_content_pdf(tmp_path, 1, _text(64 * KiB + 1), name="over.pdf"))
    assert over.pages[0].text == ""
    assert over.truncated


def test_451_pages_that_exactly_fill_the_document_budget_are_all_read(tmp_path, parsed, monkeypatch):
    _limits(monkeypatch, page=64 * KiB, document=256 * KiB)

    four = read_pdf(_content_pdf(tmp_path, 4, _text(64 * KiB), distinct=True, name="four.pdf"))
    assert len(four) == 4 and all(page.text for page in four.pages)
    assert not four.truncated

    five = read_pdf(_content_pdf(tmp_path, 5, _text(64 * KiB), distinct=True, name="five.pdf"))
    assert len(five) == 4, "the fifth page does not fit and is never opened"
    assert five.truncated


def test_451_an_ordinary_document_reads_whole_within_the_shipped_limits(tmp_path, parsed):
    content = _text(10 * KiB)
    pages = 30
    path = _content_pdf(tmp_path, pages, content, distinct=True)

    doc = read_pdf(path)

    assert len(doc) == pages and not doc.truncated
    assert all("Line 0 of an ordinary page." in page.text for page in doc.pages)
    assert parsed["bytes"] == pages * len(content), "each page parsed exactly once"


def test_451_the_page_tree_limit_still_stops_a_read_before_any_content_is_parsed(tmp_path, parsed):
    pypdf = pytest.importorskip("pypdf", minversion="6.18")
    path = _content_pdf(tmp_path, 10, _text(8 * KiB), distinct=True)
    parsed["ceiling"] = 0

    with pypdf.apply_configuration(page_tree_maximum_entries=5):
        with pytest.raises(DocumentError, match="could not be read"):
            read_pdf(path)
    assert parsed["bytes"] == 0


def test_451_the_document_op_answers_a_bounded_read(tmp_path, parsed, monkeypatch):
    # model/serve.py's "document" op, which /api/generate/document reaches.
    from model.serve import handle

    _limits(monkeypatch, page=64 * KiB, document=256 * KiB)
    path = _content_pdf(tmp_path, 80, _noop(48 * KiB))
    parsed["ceiling"] = 256 * KiB

    result = handle(None, "document", {"document_path": path, "out_dir": str(tmp_path)})

    assert "error" not in result
    assert result["pages"] == 5
    assert parsed["bytes"] <= 256 * KiB


def test_451_a_bounded_attachment_does_not_lose_the_rest_of_the_turn(tmp_path, parsed, monkeypatch):
    from model.serve import _read_attachments

    _limits(monkeypatch, page=64 * KiB, document=256 * KiB)
    bomb = _content_pdf(tmp_path, 80, _noop(48 * KiB), name="bomb.pdf")
    good = tmp_path / "good.txt"
    good.write_text("readable")
    parsed["ceiling"] = 256 * KiB

    _, documents = _read_attachments(
        _StubGen(),
        [{"kind": "document", "path": bomb}, {"kind": "document", "path": str(good)}],
    )

    assert len(documents) == 2
    assert "[document truncated: the rest was not read]" in documents[0]
    assert "readable" in documents[1]
    assert parsed["bytes"] <= 256 * KiB
