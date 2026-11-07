"""Offline conversion of untrusted local documents to the citable corpus format."""

import os
import re
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from itertools import pairwise
from pathlib import Path
from zipfile import ZipFile

import pdfplumber
from defusedxml import ElementTree
from docx import Document

from citarilo.corpus import MAX_FILE_BYTES, MAX_LINE_LENGTH, CorpusError, load_corpus

MAX_SOURCE_BYTES = 20 * 1024 * 1024
MAX_DOCX_EXPANDED_BYTES = 40 * 1024 * 1024
MAX_PDF_PAGES = 100
LONG_SENTENCE_CHARS = 600
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+\{#[^}]+\})?\s*$")
_BULLET = re.compile(r"^[-*+•]\s+")
_NUMBERED = re.compile(r"^(\d+)[.)]\s+([^\W\d_][^\s]*)", re.UNICODE)
_MONTH = re.compile(
    r"(?:leden|ledna|únor|února|březen|března|duben|dubna|květen|května|červen|června|"
    r"červenec|července|srpen|srpna|září|říjen|října|listopad|listopadu|prosinec|prosince)",
    re.I,
)
_PAGE = re.compile(r"^(?:(?:strana|page)\s+)?\d+(?:\s*(?:/|z|of)\s*\d+)?$", re.I)
_ABBREVIATIONS = re.compile(
    r"\b(?:např|tj|tzn|tzv|č|atd|apod|resp|cca|odst|písm|str|tel|ing|mgr|dr|"
    r"s\s*\.\s*r\s*\.\s*o|a\s*\.\s*s)\.",
    re.I,
)
_PROTECTED = re.compile(
    r"https?://[^\s<>\"'“”‘’«»]+|"
    r"\b\d{1,2}\.\s*\d{1,2}\.\s*\d{4}\b|\b\d+[.,]\d+\b"
)
_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}")


class DocumentImportError(ValueError):
    """An actionable local import failure, safe to display to the operator."""


@dataclass(frozen=True)
class ImportReport:
    documents: int
    sections: int
    sentences: int
    warnings: tuple[str, ...]


def slugify(text: str) -> str:
    """Produce an ASCII kebab ID; collisions are handled by the document builder."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-") or "section"


def split_sentences(text: str) -> list[str]:
    """Split Czech prose deterministically, protecting dates, ordinals and abbreviations.

    Abbreviations stay with following text; at paragraph/list boundaries the caller
    flushes explicitly. Ambiguous abbreviation-at-sentence-end cases need operator
    review, rather than a statistical model or a downloaded language resource.
    """
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()
    protected: set[int] = set()
    for pattern in (_PROTECTED, _ABBREVIATIONS):
        for match in pattern.finditer(text):
            end = match.end()
            if pattern is _PROTECTED and match[0].startswith("http"):
                end -= len(match[0]) - len(match[0].rstrip(".!?;,:)]}"))
            protected.update(range(match.start(), end))
    # Do not scan every suffix of long non-email tokens with the email regex.
    for token in re.finditer(r"\S+", text):
        if "@" in token[0] and len(token[0]) <= 320:
            for email in _EMAIL.finditer(token[0]):
                protected.update(range(token.start() + email.start(), token.start() + email.end()))
    result: list[str] = []
    start = 0
    for match in re.finditer(r"[.!?]+[“”’\"'»)\]}]*(?=\s|$)", text):
        position = match.start()
        if position in protected:
            continue
        if text[position] == ".":
            ordinal = re.search(r"(?:^|\s)(\d+)$", text[:position])
            following = text[match.end() :].lstrip()
            next_word = following.split(maxsplit=1)[0].rstrip(".,;:!?") if following else ""
            if ordinal and (following[:1].islower() or _MONTH.fullmatch(next_word)):
                continue
        result.append(text[start : match.end()].strip())
        start = match.end()
    if tail := text[start:].strip():
        result.append(tail)
    return result


def _check_docx(raw: bytes) -> None:
    """Bound ZIP expansion and reject active content before handing OOXML to a parser."""
    with ZipFile(BytesIO(raw)) as archive:
        members = archive.infolist()
        if len(members) > 2000 or sum(m.file_size for m in members) > MAX_DOCX_EXPANDED_BYTES:
            raise DocumentImportError("DOCX archive exceeds the expanded size/member limit")
        if "word/document.xml" not in archive.namelist():
            raise DocumentImportError("Not a Word DOCX document")
        for member in members:
            name = member.filename.lower()
            if any(token in name for token in ("vbaproject", "macros", "embeddings/")):
                raise DocumentImportError("DOCX macros and embedded objects are not supported")
            if name.endswith((".xml", ".rels")):
                content = archive.read(member)
                root = ElementTree.fromstring(content)
                if any(e.get("TargetMode") == "External" for e in root.iter()):
                    raise DocumentImportError("DOCX external relationships are not supported")
                if b"macroenabled" in content.lower():
                    raise DocumentImportError("Macro-enabled DOCX content is not supported")


def _extract_pdf(raw: bytes) -> str:
    """Read text and typography in one pass, removing repeated page furniture."""
    lines: list[tuple[str, int, str | None]] = []
    margin_counts: Counter[str] = Counter()
    with pdfplumber.open(BytesIO(raw)) as pdf:
        if len(pdf.pages) > MAX_PDF_PAGES:
            raise DocumentImportError(f"PDF exceeds {MAX_PDF_PAGES} pages")
        for number, page in enumerate(pdf.pages, 1):
            if page.find_tables():
                raise DocumentImportError(
                    f"PDF page {number}: tables are unsupported; convert to text or Markdown first"
                )
            body = [
                c
                for c in page.chars
                if c["bottom"] > 55 and c["top"] < page.height - 55 and c["text"].strip()
            ]
            if not body and (page.images or page.lines or page.rects or page.curves):
                raise DocumentImportError(
                    f"PDF page {number}: no extractable text in the body; "
                    "apply OCR or convert to text first"
                )
            if not any(c["text"].strip() for c in page.chars):
                page.close()
                continue
            body_size = Counter(round(c["size"], 1) for c in body or page.chars).most_common(1)[0][
                0
            ]
            # extract_text_lines groups by top, which reorders large inline fonts.
            # Use pdfplumber's clustering with the baseline from the text matrix.
            rows = pdfplumber.utils.cluster_objects(page.chars, lambda c: -c["matrix"][5], 2)
            margins: set[str] = set()
            previous_body: tuple[float, float] | None = None
            for chars in rows:
                row = sorted(
                    pdfplumber.utils.extract_words(chars, y_tolerance=page.height),
                    key=lambda w: w["x0"],
                )
                if not row:
                    continue
                line = " ".join(w["text"] for w in row)
                if (
                    max(c["bottom"] for c in chars) <= 55
                    or min(c["top"] for c in chars) >= page.height - 55
                ):
                    # Footer labels and page counters often occupy separate columns.
                    parts: list[list[dict]] = [[]]
                    for word in row:
                        if parts[-1] and word["x0"] - parts[-1][-1]["x1"] > 25:
                            parts.append([])
                        parts[-1].append(word)
                    for part in parts:
                        text = " ".join(w["text"] for w in part)
                        key = re.sub(r"\d+", "#", text) if _PAGE.fullmatch(text) else text
                        margins.add(key)
                        lines.append((text, 0, key))
                else:
                    # Word gaps stay below ~1 em even when justified; wider gaps mean
                    # columns or a borderless table, which would interleave text.
                    gap_limit = body_size * 1.5
                    bounds = (row[0]["x0"], row[-1]["x1"])
                    if any(b["x0"] - a["x1"] > gap_limit for a, b in pairwise(row)) or (
                        previous_body is not None
                        and (
                            bounds[0] - previous_body[1] > gap_limit
                            or previous_body[0] - bounds[1] > gap_limit
                        )
                    ):
                        raise DocumentImportError(
                            f"PDF page {number}: multi-column layouts are unsupported; "
                            "convert to text or Markdown first"
                        )
                    previous_body = bounds
                    size = min(c["size"] for c in chars if c["text"].strip())
                    level = (1 if size > body_size + 5 else 2) if size > body_size + 1 else 0
                    lines.append((line, level, None))
            margin_counts.update(margins)
            page.close()
    repeated = {line for line, count in margin_counts.items() if count >= 2}
    return "\n".join(
        "#" * level + " " + line if level else line
        for line, level, margin in lines
        if line and margin not in repeated
    )


def _extract_docx(raw: bytes) -> tuple[str, set[int]]:
    """Read paragraph styles only after the OOXML archive passes safety checks."""
    _check_docx(raw)
    document = Document(BytesIO(raw))
    unsupported = {
        "ins": ("tracked changes", "accept or reject all changes"),
        "del": ("tracked changes", "accept or reject all changes"),
        "sdt": ("content controls", "convert to plain paragraphs"),
        "txbxContent": ("text boxes", "convert to plain paragraphs"),
        "textbox": ("text boxes", "convert to plain paragraphs"),
        "tbl": ("tables", "convert tables to text or Markdown"),
        "AlternateContent": ("AlternateContent", "convert to plain paragraphs"),
    }
    # Inspect descendants before python-docx's top-level paragraph API can omit them.
    for element in document.element.body.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag in {"moveFrom", "moveTo"} or tag.endswith("PrChange"):
            tag = "ins"
        if tag in unsupported:
            kind, action = unsupported[tag]
            raise DocumentImportError(f"DOCX {kind} are not supported; {action} first")
    source = [p for p in document.paragraphs if p.text.strip()]
    title = next((p for p in source if p.style is not None and p.style.name == "Title"), None)
    if title is None:
        title = next(
            (p for p in source if p.style is not None and p.style.name == "Heading 1"), None
        )
    paragraphs: list[str] = []
    list_lines: set[int] = set()
    line_index = 0
    if title is not None:
        paragraphs.append("# " + title.text.strip())
        line_index = title.text.strip().count("\n") + 2
    for paragraph in source:
        if paragraph is title:
            continue
        text = paragraph.text.strip()
        style = paragraph.style
        name = style.name if style is not None else ""
        if name in {"Title", "Heading 1"}:
            text = "## " + text
        elif heading := re.fullmatch(r"Heading ([2-6])", name):
            text = "#" * int(heading[1]) + " " + text
        # OOXML numbering properties generate list labels outside paragraph.text.
        properties = paragraph._p.pPr
        if properties is None or properties.numPr is None:
            properties = style.element.pPr if style is not None else None
        if properties is not None and properties.numPr is not None:
            list_lines.add(line_index)
        paragraphs.append(text)
        line_index += text.count("\n") + 2
    # Invariant: every text run in the body must reach the output. Any structure the
    # paragraph API skips (custom XML, simple fields, ...) is rejected, not dropped.
    expected = "".join(e.text or "" for e in document.element.body.iter() if e.tag.endswith("}t"))
    if re.sub(r"\s", "", expected) != re.sub(r"\s", "", "".join(p.text for p in source)):
        raise DocumentImportError(
            "DOCX contains text outside plain paragraphs; convert to plain paragraphs first"
        )
    return "\n\n".join(paragraphs), list_lines


def _extract(path: Path) -> tuple[str, set[int] | None]:
    extension = path.suffix.lower()
    if extension not in (".pdf", ".docx", ".md"):
        raise DocumentImportError(
            f"{path}: unsupported type {extension or '(none)'}; use PDF, DOCX or MD"
        )
    if not path.is_file():
        raise DocumentImportError(f"{path}: source must be a regular local file")
    with path.open("rb") as source:
        raw = source.read(MAX_SOURCE_BYTES + 1)
    if len(raw) > MAX_SOURCE_BYTES:
        raise DocumentImportError(f"{path}: file exceeds {MAX_SOURCE_BYTES} bytes (20 MiB)")
    list_lines = None
    try:
        if extension == ".pdf":
            if not raw.startswith(b"%PDF-"):
                raise DocumentImportError("Invalid PDF signature")
            text = _extract_pdf(raw)
        elif extension == ".docx":
            text, list_lines = _extract_docx(raw)
        else:
            text = raw.decode("utf-8")
    except DocumentImportError:
        raise
    except Exception as exc:
        raise DocumentImportError(
            f"{path}: cannot extract text from {extension}; check the document"
        ) from exc
    if not any(c.isalpha() for c in text):
        raise DocumentImportError(
            f"{path}: no extractable text; scanned PDFs require OCR before import"
        )
    return unicodedata.normalize("NFC", text), list_lines


def _numbered_list_lines(lines: list[str]) -> set[int]:
    """Recognize only consecutive, uppercase numbered items; dates remain prose."""
    result: set[int] = set()
    block: list[tuple[int, int]] = []

    def finish() -> None:
        if len(block) >= 2:
            result.update(index for index, _ in block)
        block.clear()

    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        item = _NUMBERED.match(line)
        if item and item[2][0].isupper() and not _MONTH.fullmatch(item[2].rstrip(".,;:")):
            number = int(item[1])
            if block and number != block[-1][1] + 1:
                finish()
            block.append((index, number))
        else:
            finish()
    finish()
    return result


def normalize(
    text: str, *, fallback_title: str, list_lines: set[int] | None = None
) -> tuple[str, tuple[str, ...]]:
    """Normalize headings and paragraphs, preserving evidence text and source order."""
    title = ""
    preamble: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    paragraph: list[str] = []
    warnings: list[str] = []
    before: list[str] = []

    def flush() -> None:
        if not paragraph:
            return
        joined = "\n".join(paragraph)
        joined = re.sub(r"(?<=\w)\u00ad\n(?=\w)", "", joined).replace("\u00ad", "")

        def keep_hyphen(match: re.Match) -> str:
            word = match[1] + "-" + match[2]
            warnings.append(f"hard-hyphen line join retained for review: {word}")
            return match[1] + "-"

        joined = re.sub(r"(\w+)-\n(?=(\w+))", keep_hyphen, joined)
        sentences = split_sentences(joined)
        if sections:
            sections[-1][1].extend(sentences)
        else:
            before.extend(sentences)
        paragraph.clear()

    lines = text.splitlines()
    numbered_items = _numbered_list_lines(lines) if list_lines is None else set()
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            flush()
            continue
        heading = _HEADING.fullmatch(line)
        if heading:
            flush()
            label = heading[2].strip()
            if len(heading[1]) == 1 and not title:
                title = label
            else:
                sections.append((label, []))
        elif line.startswith("> ") and not sections:
            flush()
            preamble.append(line[2:])
        else:
            bullet = _BULLET.match(line) if list_lines is None else None
            if bullet or index in numbered_items:
                flush()
                line = line[bullet.end() :] if bullet else re.sub(r"^\d+[.)]\s+", "", line)
            elif list_lines is not None and index in list_lines:
                flush()  # OOXML list labels are metadata; visible prefixes are evidence.
            # Strip Markdown presentation while keeping link text and destinations.
            line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", line)
            line = line.replace("\\.", ".").replace("\\_", "_")
            line = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m[1] or m[2], line)
            paragraph.append(line)
    flush()
    if before:
        warnings.append("text before the first section heading retained in 'Introduction'")
        sections.insert(0, ("Introduction", before))
    if not sections or not any(sentences for _, sentences in sections):
        raise DocumentImportError(
            "No extractable body text; scanned PDFs require OCR before import"
        )
    if not title:
        title = fallback_title
        warnings.append("no H1 title; used source filename")
    output = [f"# {title}", "", *(f"> {line}" for line in preamble), ""]
    used: set[str] = set()
    for heading, sentences in sections:
        if not sentences:
            warnings.append(f"empty heading omitted: {heading}")
            continue
        base = slugify(heading)
        section_id = base
        suffix = 2
        while section_id in used:
            section_id = f"{base}-{suffix}"
            suffix += 1
        used.add(section_id)
        output.append(f"## {heading} {{#{section_id}}}")
        for sentence in sentences:
            if len(sentence) > LONG_SENTENCE_CHARS:
                warnings.append(f"very long sentence ({len(sentence)} characters) in {heading}")
            if len(sentence) > MAX_LINE_LENGTH:
                raise DocumentImportError(
                    f"Sentence exceeds canonical limit of {MAX_LINE_LENGTH} characters"
                )
            # Escape a literal prefix from DOCX with the corpus's own list marker.
            output.append("- " + sentence if sentence.startswith("- ") else sentence)
        output.append("")
    return "\n".join(output), tuple(warnings)


def import_documents(sources: list[Path], out: Path, *, force: bool = False) -> ImportReport:
    """Validate the whole batch before publishing any files; never call a provider."""
    if not sources:
        raise DocumentImportError("At least one source document is required")
    names: set[str] = set()
    outputs: list[tuple[str, str]] = []
    warnings: list[str] = []
    for source in sources:
        name = slugify(source.stem) + ".md"
        if name in names:
            raise DocumentImportError(f"Duplicate output filename: {name}")
        names.add(name)
        destination = out / name
        if destination.resolve() == source.resolve():
            raise DocumentImportError(f"{source}: source and output must differ")
        if os.path.lexists(destination) and not force:
            raise DocumentImportError(
                f"{destination}: output already exists; use --force to replace it"
            )
        extracted, list_lines = _extract(source)
        text, notices = normalize(extracted, fallback_title=source.stem, list_lines=list_lines)
        if len(text.encode("utf-8")) > MAX_FILE_BYTES:
            raise DocumentImportError(f"{source}: normalized output exceeds {MAX_FILE_BYTES} bytes")
        warnings.extend(f"{source.name}: {notice}" for notice in notices)
        outputs.append((name, text))
    with tempfile.TemporaryDirectory(prefix="citarilo-import-") as directory:
        staging = Path(directory)
        for name, text in outputs:
            (staging / name).write_text(text, encoding="utf-8")
        try:
            corpus = load_corpus(staging)
        except CorpusError as exc:
            raise DocumentImportError(
                f"Cannot normalize unsupported document structure: {exc}"
            ) from exc
        out.mkdir(parents=True, exist_ok=True)
        for name, text in outputs:
            # Stage on the target filesystem for atomic publication and no symlink writes.
            fd, temporary = tempfile.mkstemp(prefix=".import-", dir=out)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(text)
                if force:
                    os.replace(temporary, out / name)
                else:
                    os.link(temporary, out / name)
            finally:
                Path(temporary).unlink(missing_ok=True)
    return ImportReport(
        len(corpus.documents),
        len(list(corpus.sections())),
        len(list(corpus.sentences())),
        tuple(warnings),
    )
