"""Document ingestion: read a PDF as ordered pages of text and optional rasters.

A document is the most common carrier of long-form content, and until now the
model had no route to one. This module turns a file into :class:`Document` -
an ordered list of :class:`Page`, each with the text it carries and, where the
text layer is empty, a raster the vision tower can read instead.

Three problems get their own answer here:

- **Reading order.** A content stream lists text in the order a producer
  happened to emit it, which for a two-column page interleaves the columns.
  :func:`order_spans` recovers the order a reader would use, by finding the
  vertical gaps that separate columns and sorting within each.
- **Page identity.** Pages are joined with an explicit marker so a page number
  survives into the sequence and an answer can cite one, rather than the whole
  document flattening into an unattributable wall of text.
- **Scanned pages.** A page with no text layer is not empty, it is an image.
  :func:`needs_raster` says so, and a pluggable raster backend renders it.

The PDF reader and the raster backends are imported lazily and are optional
extras, exactly like ``opencv-python`` and ``sounddevice`` in the cognition
layer: the module imports and its pure functions run without either.
"""

from contextlib import nullcontext
from dataclasses import dataclass, field

# Markers written between pages. These live in the tokenizer's fixed-capacity
# reserved block, so adding them shifts no existing token id.
DOC_TOKEN = "<doc>"
DOC_END_TOKEN = "<doc_end>"
PAGE_TOKEN = "<page>"

# What ``to_text`` writes itself. A caller that feeds document text to
# generation passes this as ``allowed_special``, so the page boundaries survive
# encoding while anything the document merely contains stays literal.
DOC_MARKERS = frozenset({DOC_TOKEN, DOC_END_TOKEN, PAGE_TOKEN})

# A page whose text layer holds fewer characters than this is treated as a
# scan: the glyphs are pixels, not text, and only a raster will read them.
MIN_TEXT_LAYER_CHARS = 24

# Fraction of the page width a vertical whitespace run must span before it is
# taken for a column separator rather than word spacing.
COLUMN_GAP_RATIO = 0.045

# Fraction of the median line height within which two spans are the same line.
LINE_TOLERANCE_RATIO = 0.6

# Touching ``reader.pages`` at all makes pypdf flatten the whole ``/Pages``
# tree to learn the page count, before `max_pages` below ever runs - so a
# request for 3 pages of a file whose tree has 100,000 entries paid for all
# 100,000 regardless (Issue #443). Pypdf's own page_tree_maximum_entries
# configuration is what that flatten is bounded by, default 100,000; read_pdf
# lowers it for its own call, so the flatten - and an attacker's cost per
# request - stays well short of that.
#
# This is deliberately a flat ceiling, not scaled to the caller's own
# max_pages: a request for 3 pages of a genuine 500-page report still has to
# flatten all 500 entries first, the same as a request for every page of it
# does, so scaling down with max_pages would reject that legitimate document
# (confirmed - it does, with a naive max_pages-scaled bound). Ten times
# MAX_DOCUMENT_PAGES/MAX_ATTACHMENT_PAGES - the largest page count either
# caller's own request validation ever allows - comfortably covers any
# document either would legitimately reference, however many of its pages are
# actually requested, while remaining a fifth of pypdf's own default.
PAGE_TREE_ENTRY_LIMIT = 20_000

# Reading a page means parsing its content streams, which costs one to two
# seconds per decoded megabyte whatever they hold (pypdf 6.19, measured for lines
# of text, kerned text, numbers and bare operators alike), and no limit above
# bounds it (Issue #451): max_pages counts pages, max_chars counts text only once
# a page has been parsed, PAGE_TREE_ENTRY_LIMIT bounds the tree, and pypdf itself
# decodes up to 75 MB per stream. Content compresses about a thousandfold, pages
# may share one stream, a /Contents array may repeat one, and a form XObject is
# parsed again at every invocation, so a 14 KB file kept a worker parsing until
# its request timed out.
#
# read_pdf therefore measures each content stream before pypdf parses it and
# charges the decoded size to two budgets. A page that needs more than
# PAGE_CONTENT_LIMIT is left empty, as a damaged page is; once
# DOCUMENT_CONTENT_LIMIT is spent no further page is opened, as with max_chars.
# Either way Document.truncated says so. Text costs as much to parse as an attack
# does, so these bound worker time rather than detect malice: at most about 5 s
# for one page and 20 s for a document, while a page of ordinary text is a few
# tens of kilobytes of content, so documents of a few hundred pages read whole.
PAGE_CONTENT_LIMIT = 2 * 1024 * 1024
DOCUMENT_CONTENT_LIMIT = 8 * 1024 * 1024

# The decoders a content stream may use whose output pypdf can cap.
_CONTENT_DECODER_CAPS = (
    "zlib_maximum_output_length",
    "lzw_maximum_output_length",
    "run_length_maximum_output_length",
)


class DocumentError(RuntimeError):
    """A document could not be read, with the reason a caller can act on."""


@dataclass
class TextSpan:
    """One run of text with the position the content stream placed it at."""

    text: str
    x: float
    y: float
    size: float = 0.0


@dataclass
class Page:
    """One page: its text in reading order, and how it was obtained."""

    number: int
    text: str = ""
    width: float = 0.0
    height: float = 0.0
    raster_path: str | None = None

    @property
    def is_scanned(self) -> bool:
        """True when the text layer is too thin to be the page's real content."""
        return len(self.text.strip()) < MIN_TEXT_LAYER_CHARS


@dataclass
class Document:
    """An ordered set of pages read from one file."""

    path: str
    pages: list[Page] = field(default_factory=list)
    title: str = ""
    # True when a page or character limit left part of the file unread, so a
    # caller can say the text is partial instead of passing it off as the whole.
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.pages)

    @property
    def scanned_pages(self) -> list[Page]:
        """Pages whose content is pixels, so a raster is the only way to read them."""
        return [p for p in self.pages if p.is_scanned]

    def to_text(self, page_markers: bool = True, max_pages: int | None = None) -> str:
        """Flatten to one string, keeping page identity unless asked not to.

        With ``page_markers`` the pages are separated by ``<page>`` so the model
        sees where one ends, which is what makes "on page 4" answerable.
        """
        pages = self.pages if max_pages is None else self.pages[:max_pages]
        if not page_markers:
            return "\n\n".join(p.text for p in pages if p.text.strip())

        parts = [DOC_TOKEN]
        for page in pages:
            parts.append(f"{PAGE_TOKEN}{page.number}")
            if page.text.strip():
                parts.append(page.text)
        parts.append(DOC_END_TOKEN)
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Reading order
# ---------------------------------------------------------------------------

def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def detect_columns(spans: list[TextSpan], page_width: float) -> list[float]:
    """Find the x positions that separate columns.

    Projects every span onto the x axis and looks for runs of empty space wide
    enough to be a column gutter rather than the space between two words. The
    margins are excluded, so a page with generous margins and one column
    reports no boundary.

    Returns the boundary x positions, ascending. An empty list means one column.
    """
    if page_width <= 0 or len(spans) < 4:
        return []

    bins = 100
    occupied = [False] * bins
    for span in spans:
        # A span's width is not reported, so approximate it from its text
        # length at the font size that drew it.
        width = max(len(span.text) * span.size * 0.5, span.size)
        start = max(0, min(bins - 1, int(bins * span.x / page_width)))
        end = max(0, min(bins - 1, int(bins * (span.x + width) / page_width)))
        for i in range(start, end + 1):
            occupied[i] = True

    if not any(occupied):
        return []

    first = occupied.index(True)
    last = bins - 1 - occupied[::-1].index(True)
    min_run = max(1, int(bins * COLUMN_GAP_RATIO))

    boundaries: list[float] = []
    run_start = None
    for i in range(first, last + 1):
        if not occupied[i]:
            if run_start is None:
                run_start = i
            continue
        if run_start is not None:
            if i - run_start >= min_run:
                centre = (run_start + i) / 2.0
                boundaries.append(page_width * centre / bins)
            run_start = None
    return boundaries


def _column_of(x: float, boundaries: list[float]) -> int:
    column = 0
    for boundary in boundaries:
        if x >= boundary:
            column += 1
    return column


def order_spans(spans: list[TextSpan], page_width: float) -> list[TextSpan]:
    """Sort spans into the order a reader would read them.

    Column first, then down the page, then left to right within a line. PDF
    user space puts the origin at the bottom left, so a larger ``y`` is higher
    on the page and sorts earlier.
    """
    if not spans:
        return []

    boundaries = detect_columns(spans, page_width)
    sizes = [s.size for s in spans if s.size > 0]
    tolerance = _median(sizes) * LINE_TOLERANCE_RATIO if sizes else 0.0

    def line_key(span: TextSpan) -> float:
        if tolerance <= 0:
            return -span.y
        # Quantise y so spans a hair apart are read as one line rather than
        # as many lines that then sort by rounding noise.
        return -round(span.y / tolerance)

    return sorted(spans, key=lambda s: (_column_of(s.x, boundaries), line_key(s), s.x))


def spans_to_text(spans: list[TextSpan], page_width: float) -> str:
    """Join spans into page text, in reading order, one line per line."""
    ordered = order_spans(spans, page_width)
    if not ordered:
        return ""

    boundaries = detect_columns(ordered, page_width)
    sizes = [s.size for s in ordered if s.size > 0]
    tolerance = _median(sizes) * LINE_TOLERANCE_RATIO if sizes else 0.0

    lines: list[str] = []
    current: list[str] = []
    previous: TextSpan | None = None
    for span in ordered:
        text = span.text.strip()
        if not text:
            continue
        same_line = (
            previous is not None
            and tolerance > 0
            and abs(span.y - previous.y) <= tolerance
            and _column_of(span.x, boundaries) == _column_of(previous.x, boundaries)
        )
        if current and not same_line:
            lines.append(" ".join(current))
            current = []
        current.append(text)
        previous = span
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reading a file
# ---------------------------------------------------------------------------

def _load_pdf_reader():
    """Import the PDF reader lazily, with an actionable message when absent."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - exercised by the message only
        raise DocumentError(
            "Reading PDFs needs the 'pypdf' package. Install it with "
            "'pip install pypdf' (it is listed as an optional extra in "
            "requirements.txt)."
        ) from exc
    return PdfReader


class _ContentLimitReached(Exception):
    """A content budget ran out while a page was being read (Issue #451)."""


def _decode_cap(pypdf_module, cap: int):
    """Scope pypdf's content decoders to stop after ``cap`` bytes of output.

    Never above a cap already in force; pypdf reads 0 as no cap at all. Without
    pypdf's configuration API (before 6.18) its own built-in caps stay.
    """
    if pypdf_module is None or not hasattr(pypdf_module, "apply_configuration"):
        return nullcontext()
    current = pypdf_module.get_configuration()
    caps = {}
    for name in _CONTENT_DECODER_CAPS:
        value = getattr(current, name, None)
        if isinstance(value, int):
            caps[name] = cap if value <= 0 else min(value, cap)
    return pypdf_module.apply_configuration(**caps) if caps else nullcontext()


def _decoded_length(stream, limit: int, pypdf_module) -> int | None:
    """Decoded size of a content stream, or None when it is larger than ``limit``.

    Decoding stops one byte past the limit, so measuring a bomb never decodes it.
    A stream that fits stays decoded in pypdf's cache for the parse that follows.
    """
    get_data = getattr(stream, "get_data", None)
    if get_data is None:
        return 0
    limit_error = getattr(getattr(pypdf_module, "errors", None), "LimitReachedError", ())
    try:
        with _decode_cap(pypdf_module, limit + 1):
            data = get_data()
    except limit_error:
        return None
    except Exception:  # noqa: BLE001 - a damaged stream fails again when parsed, as before
        return 0
    return len(data) if len(data) <= limit else None


class _ContentBudget:
    """Decoded content a read_pdf call may still hand pypdf to parse (Issue #451)."""

    def __init__(self, pypdf_module):
        self._pypdf = pypdf_module
        self.document_left = DOCUMENT_CONTENT_LIMIT
        self.page_left = PAGE_CONTENT_LIMIT
        # "page" or "document" once a limit has stopped reading.
        self.reached = None

    def start_page(self) -> None:
        self.page_left = PAGE_CONTENT_LIMIT
        if self.reached == "page":
            self.reached = None

    def check(self) -> None:
        if self.reached:
            raise _ContentLimitReached

    def charge(self, stream) -> None:
        """Charge one stream about to be parsed, or raise when it does not fit.

        A stream too large to fit costs all the room that was left, so a run of
        oversized pages spends the document budget too.
        """
        self.check()
        room = min(self.page_left, self.document_left)
        size = _decoded_length(stream, room, self._pypdf)
        used = room if size is None else size
        self.page_left -= used
        self.document_left -= used
        if size is None:
            self.reached = "document" if self.document_left <= 0 else "page"
            raise _ContentLimitReached


def _resolved(obj, key):
    """``obj[key]`` resolved, or None when absent or ``obj`` is not a pypdf dictionary."""
    try:
        value = obj[key]
    except Exception:  # noqa: BLE001 - absent, malformed, or no dictionary at all
        return None
    return value.get_object() if hasattr(value, "get_object") else value


def _page_content_streams(page):
    """The streams of a page's own ``/Contents``: one stream, or each in an array."""
    contents = _resolved(page, "/Contents")
    if contents is None:
        return
    if hasattr(contents, "get_data"):
        yield contents
        return
    try:
        items = iter(contents)
    except TypeError:
        return
    for item in items:
        yield item.get_object() if hasattr(item, "get_object") else item


def _page_resources(page):
    """A page's ``/Resources``, inherited from the page tree as pypdf reads them."""
    get_inherited = getattr(page, "get_inherited", None)
    if get_inherited is None:
        return _resolved(page, "/Resources")
    try:
        resources = get_inherited("/Resources", None)
    except Exception:  # noqa: BLE001 - malformed resources hold no forms
        return None
    return resources.get_object() if hasattr(resources, "get_object") else resources


def _form_xobject(resources, operands):
    """The stream a ``Do`` makes pypdf parse, or None for an image or an unknown name."""
    if not operands or resources is None:
        return None
    xobjects = _resolved(resources, "/XObject")
    form = _resolved(xobjects, operands[0]) if xobjects is not None else None
    if form is None or not hasattr(form, "get_data") or _resolved(form, "/Subtype") == "/Image":
        return None
    return form


def _page_text(page, budget: _ContentBudget) -> tuple[list[TextSpan], str]:
    """Positioned spans and plain text of one page, from one extraction pass.

    The plain text is what pypdf returns from that same pass, so a page without
    positioned spans is not parsed a second time to get it. Each form XObject a
    ``Do`` is about to parse is charged to ``budget`` first, at every invocation.
    pypdf swallows errors raised inside a form, so a spent budget raises again at
    the next operator until extraction has unwound. What was read before then is
    kept, the way a page cut by max_chars keeps what fits.
    """
    spans: list[TextSpan] = []
    resources = [_page_resources(page)]

    def visitor(text, cm, tm, font_dict, font_size):
        if not text or not text.strip():
            return
        try:
            x, y = float(tm[4]), float(tm[5])
        except (TypeError, IndexError, ValueError):
            return
        size = float(font_size or 0.0)
        spans.append(TextSpan(text=text, x=x, y=y, size=size))

    def before(operator, operands, cm, tm):
        budget.check()
        if operator == b"Do":
            form = _form_xobject(resources[-1], operands)
            if form is not None:
                budget.charge(form)
            resources.append(_resolved(form, "/Resources") if form is not None else None)

    def after(operator, operands, cm, tm):
        if operator == b"Do" and len(resources) > 1:
            resources.pop()

    try:
        plain = page.extract_text(
            visitor_text=visitor, visitor_operand_before=before, visitor_operand_after=after
        )
    except _ContentLimitReached:
        return spans, ""
    except Exception:  # noqa: BLE001 - a damaged page must not fail the document
        return [], ""
    return spans, plain or ""


def _page_overhead(number: int) -> int:
    """Characters ``to_text`` spends around a page's text, whatever the page holds.

    The marker and the two line breaks, one after the marker and one after the
    text. A blank page writes only the first, so this never undercounts.
    """
    return len(PAGE_TOKEN) + len(str(number)) + 2


def _page_tree_limit(pypdf_default: int) -> int:
    """How many ``/Pages`` tree entries a read_pdf call lets pypdf flatten.

    Never above pypdf's own current default (whatever that is, e.g. in a test
    that has already lowered it), and never above PAGE_TREE_ENTRY_LIMIT - a
    caller cannot raise it by deploying a newer pypdf with a larger one.
    """
    return min(PAGE_TREE_ENTRY_LIMIT, pypdf_default)


def read_pdf(
    path: str, max_pages: int | None = None, max_chars: int | None = None
) -> Document:
    """Read ``path`` into a :class:`Document` with pages in reading order.

    Pages that raise are kept as empty pages rather than dropped, so page
    numbers stay aligned with the file and a later raster pass can fill them in.

    ``max_pages`` and ``max_chars`` bound how much of a long file is read. Reading
    stops at the page limit, or once the pages read fill ``max_chars`` of
    :meth:`Document.to_text` output, page markers included so a run of blank pages
    is not free. The page that crosses the line is cut to fit and no page after
    it is opened, so what a limit saves is the work of reading what it leaves
    out. Either way ``Document.truncated`` records that the file held more.

    Reading also stops short, before either limit above ever runs, when the
    file's page *tree* itself is larger than ``PAGE_TREE_ENTRY_LIMIT`` (Issue
    #443): touching ``reader.pages`` at all, even for page 0, makes pypdf
    flatten the whole tree first, so a request for a handful of pages used to
    still pay for a file's entire declared page count. That raises
    :class:`DocumentError`, the same as any other unreadable file.

    What parsing a page costs is bounded too (Issue #451): every content stream
    is measured before pypdf parses it, a page's own and each form XObject's at
    each invocation, against ``PAGE_CONTENT_LIMIT`` and
    ``DOCUMENT_CONTENT_LIMIT``. A page over the first is kept empty, the second
    stops the read, and either sets ``Document.truncated``.
    """
    reader_cls = _load_pdf_reader()
    try:
        reader = reader_cls(path)
    except Exception as exc:  # noqa: BLE001 - surface as one document error
        raise DocumentError(f"could not open '{path}': {exc}") from exc

    title = ""
    try:
        metadata = reader.metadata or {}
        title = str(metadata.get("/Title", "") or "")
    except Exception:  # noqa: BLE001 - metadata is optional
        title = ""

    document = Document(path=path, title=title)
    room = max_chars

    # Scopes pypdf's own page-tree-walk ceiling down (see PAGE_TREE_ENTRY_LIMIT
    # above), and converts pypdf's own resource/parse errors into
    # DocumentError, same as a file that fails to open at all, above. Real
    # production use always has pypdf importable here - _load_pdf_reader()
    # above only succeeds once it already is - but a test that replaces
    # _load_pdf_reader() with a stub reader never needs it to be, so this
    # stays as optional as the rest of this module when it genuinely is not.
    # The configuration API itself only exists from pypdf 6.18; an older pypdf
    # keeps its own built-in ceiling and still gets the error conversion below.
    try:
        import pypdf as _pypdf
    except ImportError:
        _pypdf = None

    page_tree_scope = nullcontext()
    resource_errors = ()
    if _pypdf is not None:
        resource_errors = _pypdf.errors.PyPdfError
        if hasattr(_pypdf, "apply_configuration") and hasattr(_pypdf, "get_configuration"):
            page_tree_limit = _page_tree_limit(_pypdf.get_configuration().page_tree_maximum_entries)
            page_tree_scope = _pypdf.apply_configuration(page_tree_maximum_entries=page_tree_limit)

    budget = _ContentBudget(_pypdf)
    try:
        with page_tree_scope:
            for index, page in enumerate(reader.pages):
                if (max_pages is not None and index >= max_pages) or (room is not None and room <= 0):
                    document.truncated = True
                    break
                try:
                    box = page.mediabox
                    width, height = float(box.width), float(box.height)
                except Exception:  # noqa: BLE001 - fall back to a common page size
                    width, height = 612.0, 792.0

                budget.start_page()
                spans, plain = [], ""
                try:
                    for stream in _page_content_streams(page):
                        budget.charge(stream)
                except _ContentLimitReached:
                    pass
                else:
                    spans, plain = _page_text(page, budget)
                if budget.reached:
                    document.truncated = True
                    if budget.reached == "document" and not spans and not plain.strip():
                        break

                text = spans_to_text(spans, width) if spans else plain
                text = text.strip()
                if room is not None:
                    room -= _page_overhead(index + 1)
                    if len(text) > max(room, 0):
                        text, document.truncated = text[: max(room, 0)], True
                    room -= len(text)
                document.pages.append(Page(number=index + 1, text=text, width=width, height=height))
                if budget.reached == "document":
                    break
    except resource_errors as exc:
        # LimitReachedError (the tree is bigger than page_tree_limit allows) and
        # any other resource/parse error pypdf itself raises while walking it.
        raise DocumentError(f"'{path}' could not be read: {exc}") from exc
    return document


def read_document(
    path: str, max_pages: int | None = None, max_chars: int | None = None
) -> Document:
    """Read any supported document. PDF today; the dispatch point for more.

    ``max_chars`` stops a long file being read past that much text; see
    :func:`read_pdf`. A plain-text file is one page, so ``max_pages`` does not
    apply to it, and ``max_chars`` is what bounds it.
    """
    lowered = path.lower()
    if lowered.endswith(".pdf"):
        return read_pdf(path, max_pages=max_pages, max_chars=max_chars)
    if lowered.endswith((".txt", ".md")):
        room = None if max_chars is None else max(max_chars - _page_overhead(1), 0)
        with open(path, encoding="utf-8", errors="replace") as handle:
            # One character past the limit is enough to know there was more, and
            # the rest of the file is never read.
            text = handle.read() if room is None else handle.read(room + 1)
        truncated = room is not None and len(text) > room
        if truncated:
            text = text[:room]
        return Document(
            path=path, pages=[Page(number=1, text=text.strip())], truncated=truncated
        )
    raise DocumentError(f"unsupported document type: '{path}'")


def needs_raster(page: Page) -> bool:
    """True when a page carries no usable text layer and must be seen instead."""
    return page.is_scanned


# ---------------------------------------------------------------------------
# Page rasterisation (optional, pluggable)
# ---------------------------------------------------------------------------

_RASTER_BACKENDS: dict = {}


def register_raster_backend(name: str, fn) -> None:
    """Register a callable ``fn(path, page_number, dpi) -> PIL.Image``.

    Rasterisation is pluggable and optional on purpose. Every capable renderer
    carries a heavier licence than this project, so none is a dependency and
    the choice stays with the deployment.
    """
    _RASTER_BACKENDS[name] = fn


def available_raster_backends() -> list[str]:
    """Names of the raster backends registered in this process."""
    return sorted(_RASTER_BACKENDS)


def rasterize_page(path: str, page_number: int, dpi: int = 200, backend: str | None = None):
    """Render one page to a PIL image using a registered backend.

    Raises :class:`DocumentError` naming the situation when no backend is
    registered, rather than returning something empty that reads as a blank page.
    """
    if backend is not None:
        fn = _RASTER_BACKENDS.get(backend)
        if fn is None:
            raise DocumentError(
                f"raster backend '{backend}' is not registered; "
                f"available: {available_raster_backends() or 'none'}"
            )
        return fn(path, page_number, dpi)

    if not _RASTER_BACKENDS:
        raise DocumentError(
            "no page raster backend is registered, so a scanned page cannot be "
            "rendered. Register one with document.register_raster_backend(), or "
            "install an optional renderer. The text layer path needs none."
        )
    name = available_raster_backends()[0]
    return _RASTER_BACKENDS[name](path, page_number, dpi)
