"""Readers for office files, web pages, data and source code.

PDF has its own reader in :mod:`model.document`; everything else people attach
lands here. Each reader returns the same :class:`~model.document.Document`, so
the page markers, the character budget and the truncation flag behave the same
whatever the file was.

Office files (``.docx``, ``.pptx``, ``.xlsx``) are zip archives of XML, and are
read with the standard library alone. Two things guard that path, because the
bytes are untrusted:

- **Decompressed size.** Each member's declared size is checked before it is
  read, and the read itself is capped, so a small archive that inflates to
  gigabytes is refused rather than unpacked.
- **Entity expansion.** A member that declares a DOCTYPE is refused. Office XML
  never needs one, and a DOCTYPE is how an entity bomb is written.
"""

import csv
import io
import json
import re
import zipfile
from html.parser import HTMLParser
from xml.etree import ElementTree

from .document import Document, DocumentError, Page, _page_overhead

# Ceiling on what one archive member may inflate to, and on all members read.
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024

# Spreadsheets can hold millions of cells; the model reads a window's worth.
MAX_SHEET_ROWS = 5000
MAX_SHEET_COLS = 64

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

CODE_EXTENSIONS = {
    ".py": "python", ".js": "javascript", ".mjs": "javascript", ".ts": "typescript",
    ".tsx": "tsx", ".jsx": "jsx", ".java": "java", ".go": "go", ".rs": "rust",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp", ".cs": "csharp",
    ".rb": "ruby", ".php": "php", ".sh": "shell", ".sql": "sql", ".css": "css",
    ".swift": "swift", ".kt": "kotlin", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
}

OFFICE_EXTENSIONS = (".docx", ".pptx", ".xlsx")
TEXT_EXTENSIONS = (".csv", ".html", ".htm", ".json") + tuple(CODE_EXTENSIONS)


def handles(path: str) -> bool:
    """True when this module reads ``path``."""
    return path.lower().endswith(OFFICE_EXTENSIONS + TEXT_EXTENSIONS)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

def _fit(pages: list[Page], max_pages, max_chars, path: str, title: str = "") -> Document:
    """Apply the page and character limits the PDF reader applies."""
    truncated = False
    if max_pages is not None and len(pages) > max_pages:
        pages, truncated = pages[:max_pages], True
    if max_chars is not None:
        room = max_chars
        kept = []
        for page in pages:
            room -= _page_overhead(page.number)
            if room <= 0:
                truncated = True
                break
            if len(page.text) > room:
                kept.append(Page(number=page.number, text=page.text[:room]))
                truncated = True
                break
            kept.append(page)
            room -= len(page.text)
        pages = kept
    return Document(path=path, pages=pages, title=title, truncated=truncated)


# ---------------------------------------------------------------------------
# Office archives
# ---------------------------------------------------------------------------

class _Archive:
    """A zip archive whose members are read under size limits."""

    def __init__(self, path: str):
        try:
            self.zip = zipfile.ZipFile(path)
        except (zipfile.BadZipFile, OSError) as exc:
            raise DocumentError(f"'{path}' is not a readable office file: {exc}") from exc
        self.path = path
        self.read_total = 0

    def names(self) -> list[str]:
        return self.zip.namelist()

    def read(self, name: str) -> bytes:
        try:
            info = self.zip.getinfo(name)
        except KeyError as exc:
            raise DocumentError(f"'{self.path}' is missing {name}") from exc
        if info.file_size > MAX_MEMBER_BYTES:
            raise DocumentError(f"'{self.path}': {name} inflates past {MAX_MEMBER_BYTES} bytes")
        if self.read_total + info.file_size > MAX_ARCHIVE_BYTES:
            raise DocumentError(f"'{self.path}' inflates past {MAX_ARCHIVE_BYTES} bytes")
        with self.zip.open(info) as handle:
            # The declared size can lie; the read is capped regardless.
            data = handle.read(MAX_MEMBER_BYTES + 1)
        if len(data) > MAX_MEMBER_BYTES:
            raise DocumentError(f"'{self.path}': {name} inflates past {MAX_MEMBER_BYTES} bytes")
        self.read_total += len(data)
        return data

    def xml(self, name: str):
        data = self.read(name)
        if b"<!DOCTYPE" in data[:4096].upper() or b"<!ENTITY" in data.upper():
            raise DocumentError(f"'{self.path}': {name} declares a DOCTYPE, which office XML never needs")
        try:
            return ElementTree.fromstring(data)
        except ElementTree.ParseError as exc:
            raise DocumentError(f"'{self.path}': {name} is not valid XML: {exc}") from exc

    def close(self):
        self.zip.close()


def _natural(name: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def _docx_paragraph(p) -> str:
    parts = []
    for node in p.iter():
        if node.tag == W + "t" and node.text:
            parts.append(node.text)
        elif node.tag == W + "tab":
            parts.append("\t")
        elif node.tag in (W + "br", W + "cr"):
            parts.append("\n")
    text = "".join(parts)
    style = p.find(f"{W}pPr/{W}pStyle")
    level = style.get(W + "val", "") if style is not None else ""
    match = re.match(r"(?i)heading(\d)", level)
    if match and text.strip():
        return "#" * int(match.group(1)) + " " + text
    if p.find(f"{W}pPr/{W}numPr") is not None and text.strip():
        return "- " + text
    return text


def _docx_table(tbl) -> str:
    rows = []
    for tr in tbl.iter(W + "tr"):
        cells = []
        for tc in tr.iter(W + "tc"):
            cells.append(" ".join(_docx_paragraph(p) for p in tc.iter(W + "p")).strip())
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def read_docx(path, max_pages=None, max_chars=None) -> Document:
    archive = _Archive(path)
    try:
        root = archive.xml("word/document.xml")
    finally:
        archive.close()
    body = root.find(W + "body")
    blocks = []
    for child in list(body) if body is not None else []:
        if child.tag == W + "p":
            blocks.append(_docx_paragraph(child))
        elif child.tag == W + "tbl":
            blocks.append(_docx_table(child))
    text = "\n".join(b for b in blocks if b.strip())
    # A word-processor file has no fixed pages; it is one flow of text.
    return _fit([Page(number=1, text=text)], max_pages, max_chars, path)


def read_pptx(path, max_pages=None, max_chars=None) -> Document:
    archive = _Archive(path)
    try:
        slides = sorted(
            (n for n in archive.names() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
            key=_natural,
        )
        pages = []
        for number, name in enumerate(slides, start=1):
            if max_pages is not None and number > max_pages:
                break
            root = archive.xml(name)
            lines = []
            for para in root.iter(A + "p"):
                line = "".join(t.text or "" for t in para.iter(A + "t"))
                if line.strip():
                    lines.append(line)
            pages.append(Page(number=number, text="\n".join(lines)))
    finally:
        archive.close()
    document = _fit(pages, max_pages, max_chars, path)
    document.truncated = document.truncated or (max_pages is not None and len(slides) > max_pages)
    return document


def _column_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref or "")
    if not letters:
        return 0
    index = 0
    for ch in letters.group(0):
        index = index * 26 + (ord(ch) - 64)
    return index - 1


def read_xlsx(path, max_pages=None, max_chars=None) -> Document:
    archive = _Archive(path)
    try:
        shared = []
        if "xl/sharedStrings.xml" in archive.names():
            for si in archive.xml("xl/sharedStrings.xml").iter(S + "si"):
                shared.append("".join(t.text or "" for t in si.iter(S + "t")))

        names = {}
        if "xl/workbook.xml" in archive.names():
            for sheet in archive.xml("xl/workbook.xml").iter(S + "sheet"):
                rel = sheet.get(R + "id", "")
                names[rel] = sheet.get("name", "")
        sheet_files = sorted(
            (n for n in archive.names() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)),
            key=_natural,
        )
        ordered_names = list(names.values())

        pages, cut = [], False
        for number, name in enumerate(sheet_files, start=1):
            if max_pages is not None and number > max_pages:
                cut = True
                break
            root = archive.xml(name)
            rows = []
            for r_index, row in enumerate(root.iter(S + "row")):
                if r_index >= MAX_SHEET_ROWS:
                    cut = True
                    break
                cells = {}
                for c in row.iter(S + "c"):
                    col = _column_index(c.get("r", ""))
                    if col >= MAX_SHEET_COLS:
                        continue
                    kind = c.get("t")
                    if kind == "inlineStr":
                        value = "".join(t.text or "" for t in c.iter(S + "t"))
                    else:
                        v = c.find(S + "v")
                        value = v.text if v is not None and v.text is not None else ""
                        if kind == "s" and value.isdigit() and int(value) < len(shared):
                            value = shared[int(value)]
                    cells[col] = value
                if cells and any(str(v).strip() for v in cells.values()):
                    width = max(cells) + 1
                    rows.append("| " + " | ".join(str(cells.get(i, "")) for i in range(width)) + " |")
            title = ordered_names[number - 1] if number - 1 < len(ordered_names) else f"Sheet{number}"
            pages.append(Page(number=number, text=f"## {title}\n" + "\n".join(rows)))
    finally:
        archive.close()
    document = _fit(pages, max_pages, max_chars, path)
    document.truncated = document.truncated or cut
    return document


# ---------------------------------------------------------------------------
# Text formats
# ---------------------------------------------------------------------------

class _HTMLText(HTMLParser):
    """Visible text of a page, with block elements on their own lines."""

    SKIP = {"script", "style", "noscript", "template", "svg", "head"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
             "section", "article", "header", "footer", "pre", "blockquote", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip, self.title, self._in_title = [], 0, "", False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        if tag == "title":
            self._in_title = True
        if tag in self.BLOCK:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("- ")
        if re.fullmatch(r"h[1-6]", tag):
            self.parts.append("#" * int(tag[1]) + " ")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        if tag == "title":
            self._in_title = False
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if not self.skip:
            self.parts.append(data)

    def text(self) -> str:
        joined = "".join(self.parts)
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in joined.splitlines()]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _read_text(path, limit):
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read() if limit is None else handle.read(limit + 1)


def read_text_format(path, max_pages=None, max_chars=None) -> Document:
    lowered = path.lower()
    # Read a little past the budget: markup and fences are dropped or added,
    # and the budget applies to what the model sees.
    limit = None if max_chars is None else max_chars * 4
    raw = _read_text(path, limit)
    clipped = limit is not None and len(raw) > limit
    title = ""

    if lowered.endswith((".html", ".htm")):
        parser = _HTMLText()
        parser.feed(raw)
        text, title = parser.text(), parser.title.strip()
    elif lowered.endswith(".csv"):
        rows = []
        for i, row in enumerate(csv.reader(io.StringIO(raw))):
            if i >= MAX_SHEET_ROWS:
                clipped = True
                break
            rows.append("| " + " | ".join(row[:MAX_SHEET_COLS]) + " |")
        text = "\n".join(rows)
    elif lowered.endswith(".json"):
        try:
            text = json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
        except ValueError:
            text = raw
    else:
        ext = lowered[lowered.rfind("."):]
        name = path.replace("\\", "/").rsplit("/", 1)[-1]
        text = f"```{CODE_EXTENSIONS.get(ext, '')} {name}\n{raw.rstrip()}\n```"

    document = _fit([Page(number=1, text=text.strip())], max_pages, max_chars, path, title=title)
    document.truncated = document.truncated or clipped
    return document


def read(path, max_pages=None, max_chars=None) -> Document:
    lowered = path.lower()
    if lowered.endswith(".docx"):
        return read_docx(path, max_pages, max_chars)
    if lowered.endswith(".pptx"):
        return read_pptx(path, max_pages, max_chars)
    if lowered.endswith(".xlsx"):
        return read_xlsx(path, max_pages, max_chars)
    return read_text_format(path, max_pages, max_chars)
