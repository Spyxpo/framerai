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

from dataclasses import dataclass, field

# Markers written between pages. These live in the tokenizer's fixed-capacity
# reserved block, so adding them shifts no existing token id.
DOC_TOKEN = "<doc>"
DOC_END_TOKEN = "<doc_end>"
PAGE_TOKEN = "<page>"

# A page whose text layer holds fewer characters than this is treated as a
# scan: the glyphs are pixels, not text, and only a raster will read them.
MIN_TEXT_LAYER_CHARS = 24

# Fraction of the page width a vertical whitespace run must span before it is
# taken for a column separator rather than word spacing.
COLUMN_GAP_RATIO = 0.045

# Fraction of the median line height within which two spans are the same line.
LINE_TOLERANCE_RATIO = 0.6


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


def _page_spans(page) -> list[TextSpan]:
    """Collect positioned text spans from one page via the extraction visitor."""
    spans: list[TextSpan] = []

    def visitor(text, cm, tm, font_dict, font_size):
        if not text or not text.strip():
            return
        try:
            x, y = float(tm[4]), float(tm[5])
        except (TypeError, IndexError, ValueError):
            return
        size = float(font_size or 0.0)
        spans.append(TextSpan(text=text, x=x, y=y, size=size))

    try:
        page.extract_text(visitor_text=visitor)
    except Exception:  # noqa: BLE001 - a damaged page must not fail the document
        return []
    return spans


def read_pdf(path: str, max_pages: int | None = None) -> Document:
    """Read ``path`` into a :class:`Document` with pages in reading order.

    Pages that raise are kept as empty pages rather than dropped, so page
    numbers stay aligned with the file and a later raster pass can fill them in.
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
    for index, page in enumerate(reader.pages):
        if max_pages is not None and index >= max_pages:
            break
        try:
            box = page.mediabox
            width, height = float(box.width), float(box.height)
        except Exception:  # noqa: BLE001 - fall back to a common page size
            width, height = 612.0, 792.0

        spans = _page_spans(page)
        if spans:
            text = spans_to_text(spans, width)
        else:
            try:
                text = page.extract_text() or ""
            except Exception:  # noqa: BLE001 - a damaged page yields no text
                text = ""
        document.pages.append(
            Page(number=index + 1, text=text.strip(), width=width, height=height)
        )
    return document


def read_document(path: str, max_pages: int | None = None) -> Document:
    """Read any supported document. PDF today; the dispatch point for more."""
    lowered = path.lower()
    if lowered.endswith(".pdf"):
        return read_pdf(path, max_pages=max_pages)
    if lowered.endswith((".txt", ".md")):
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        return Document(path=path, pages=[Page(number=1, text=text.strip())])
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
