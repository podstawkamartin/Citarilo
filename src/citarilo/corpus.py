"""Parse the restricted Markdown corpus into immutable, citable records.

The hash is SHA-256 of UTF-8 JSON with ``ensure_ascii=False``, ``sort_keys=True``
and separators ``(',', ':')`` (no trailing newline). Its exact structure is::

    {"schema_version": 1, "chunking_version": CHUNKING_VERSION, "documents": [
        {"doc_id": str, "title": str, "sections": [
            {"section_id": str, "heading": str, "sentences": [str, ...]}, ...
        ]}, ...
    ]}

Documents follow sorted filenames; sections and sentence texts follow source order.
All parsed text is NFC; titles, headings and sentences have surrounding whitespace
removed. Paths, preambles, line numbers, blank lines and list markers are excluded.
Sentence IDs derive from document/section IDs and the position in these arrays.
Increment CHUNKING_VERSION when changing parsing or citation boundary semantics.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from langchain_core.documents import Document

CHUNKING_VERSION = "1"
MAX_FILE_BYTES = 256 * 1024
MAX_LINE_LENGTH = 4096

_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_TITLE = re.compile(r"# ([^#\s].*?)\s*\Z")
_SECTION = re.compile(r"## (.+?) \{#([a-z0-9]+(?:-[a-z0-9]+)*)\}\s*\Z")
_HEADING = re.compile(r"^\s*#")
_HTML = re.compile(r"<!--|<![A-Za-z]|<\?|</?[A-Za-z][^>]*>")
_LINK = re.compile(r"\[[^\]]*\]\s*(?:\(|\[)|^\s*\[[^\]]+\]:")
_TABLE = re.compile(r"^\s*\||\|\s*$|^\s*:?-+:?\s*\|")
_TABLE_DELIMITER = re.compile(r"\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?\s*\Z")


class CorpusError(ValueError):
    """All corpus validation failures, each formatted as file:line: message."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = tuple(problems)
        super().__init__("\n".join(problems))


@dataclass(frozen=True, slots=True)
class Sentence:
    """A citable non-empty body line, with its one-based source line number."""

    id: str
    text: str
    line: int
    is_list_item: bool


@dataclass(frozen=True, slots=True)
class Section:
    """A retrieval unit with a document-qualified stable key."""

    key: str
    doc_id: str
    section_id: str
    heading: str
    sentences: tuple[Sentence, ...]


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    """A document with non-citable preamble and ordered sections."""

    doc_id: str
    title: str
    path: Path
    preamble: str
    sections: tuple[Section, ...]


@dataclass(frozen=True, slots=True)
class Corpus:
    """An immutable corpus with constant-time citation lookups."""

    documents: tuple[CorpusDocument, ...]
    corpus_hash: str
    _sentences: Mapping[str, Sentence] = field(init=False, repr=False, compare=False)
    _sections: Mapping[str, Section] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        sections = {section.key: section for section in self.sections()}
        sentences = {sentence.id: sentence for sentence in self.sentences()}
        object.__setattr__(self, "_sections", MappingProxyType(sections))
        object.__setattr__(self, "_sentences", MappingProxyType(sentences))

    def sentence(self, sentence_id: str) -> Sentence:
        """Return a sentence by ID, raising KeyError for an unknown ID."""
        return self._sentences[sentence_id]

    def section(self, key: str) -> Section:
        """Return a section by key, raising KeyError for an unknown key."""
        return self._sections[key]

    def sections(self) -> Iterator[Section]:
        """Iterate over all sections in document and source order."""
        for document in self.documents:
            yield from document.sections

    def sentences(self) -> Iterator[Sentence]:
        """Iterate over all sentences in document and source order."""
        for section in self.sections():
            yield from section.sentences


def _forbidden(line: str, following_line: str) -> Iterator[str]:
    if "```" in line or "~~~" in line:
        yield "fenced code is forbidden"
    if _TABLE.search(line) or ("|" in line and _TABLE_DELIMITER.fullmatch(following_line)):
        yield "tables are forbidden"
    if _HTML.search(line):
        yield "HTML tags are forbidden"
    if "![" in line:
        yield "images are forbidden"
    if _LINK.search(line):
        yield "Markdown links are forbidden"


def _parse_document(path: Path, problems: list[str]) -> CorpusDocument | None:
    def error(line: int, message: str) -> None:
        problems.append(f"{path}:{line}: {message}")

    if not _ID.fullmatch(path.stem):
        error(1, "document ID must be lowercase kebab-case")
    try:
        with path.open("rb") as source:
            raw = source.read(MAX_FILE_BYTES + 1)
    except OSError as exc:
        error(1, f"cannot read document: {exc.strerror}")
        return None
    if len(raw) > MAX_FILE_BYTES:
        error(1, f"file exceeds maximum size of {MAX_FILE_BYTES} bytes")
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        for number, raw_line in enumerate(raw.split(b"\n"), start=1):
            try:
                raw_line.decode("utf-8")
            except UnicodeDecodeError:
                error(number, "invalid UTF-8 encoding")
        text = raw.decode("utf-8", errors="replace")
    lines = unicodedata.normalize("NFC", text).split("\n")
    title = ""
    preamble: list[str] = []
    sections: list[Section] = []
    seen_ids: set[str] = set()
    current: tuple[str, str, int] | None = None
    sentences: list[Sentence] = []
    in_body = False

    def finish_section() -> None:
        if current is None:
            return
        section_id, heading, source_line = current
        if not sentences:
            error(source_line, "section must contain at least one sentence")
        sections.append(
            Section(
                key=f"{path.stem}.{section_id}",
                doc_id=path.stem,
                section_id=section_id,
                heading=heading,
                sentences=tuple(sentences),
            )
        )

    for number, line in enumerate(lines, start=1):
        if "\r" in line:
            error(number, "CR/CRLF line endings are forbidden; use LF")
        if len(line) > MAX_LINE_LENGTH:
            error(number, f"line exceeds maximum length of {MAX_LINE_LENGTH} characters")
        following_line = lines[number] if number < len(lines) else ""
        for message in _forbidden(line, following_line):
            error(number, message)
        stripped = line.strip()
        if number == 1:
            match = _TITLE.fullmatch(line)
            if match is None or re.search(r"\s#+$", stripped):
                error(number, "line 1 must be exactly one H1: '# <title>'")
            else:
                title = match[1].strip()
            continue
        match = _SECTION.fullmatch(line)
        if match is not None and match[1].strip() and not match[1].lstrip().startswith("#"):
            finish_section()
            section_id = match[2]
            if section_id in seen_ids:
                error(number, f"duplicate section ID '{section_id}'")
            seen_ids.add(section_id)
            current = (section_id, match[1].strip(), number)
            sentences = []
            in_body = True
            continue
        if _HEADING.match(line):
            if re.match(r"^\s*###", line):
                error(number, "H3+ headings are forbidden")
            elif re.match(r"^\s*##", line):
                error(number, "H2 must use '## <heading> {#lowercase-kebab-id}'")
                finish_section()
                current = None
                sentences = []
                in_body = True
            else:
                error(number, "H1 is allowed only on line 1")
            continue
        if not in_body:
            if stripped and not line.startswith("> "):
                error(number, "preamble allows only blank lines and '> ' blockquote lines")
            preamble.append(line)
        elif stripped and current is not None:
            is_list_item = stripped.startswith("- ")
            sentence_text = stripped[2:].strip() if is_list_item else stripped
            if stripped == "-" or not sentence_text:
                error(number, "list item must contain sentence text")
                continue
            sentences.append(
                Sentence(
                    id=f"{path.stem}.{current[0]}.{len(sentences) + 1}",
                    text=sentence_text,
                    line=number,
                    is_list_item=is_list_item,
                )
            )
    finish_section()
    if not sections:
        error(1, "document must contain at least one H2 section")
    return CorpusDocument(path.stem, title, path, "\n".join(preamble), tuple(sections))


def _corpus_hash(documents: tuple[CorpusDocument, ...]) -> str:
    canonical = {
        "schema_version": 1,
        "chunking_version": CHUNKING_VERSION,
        "documents": [
            {
                "doc_id": document.doc_id,
                "title": document.title,
                "sections": [
                    {
                        "section_id": section.section_id,
                        "heading": section.heading,
                        "sentences": [sentence.text for sentence in section.sentences],
                    }
                    for section in document.sections
                ],
            }
            for document in documents
        ],
    }
    serialized = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_corpus(data_dir: Path) -> Corpus:
    """Load sorted top-level *.md files, reporting all validation problems together."""
    problems: list[str] = []
    if not data_dir.is_dir():
        raise CorpusError([f"{data_dir}:1: corpus directory does not exist or is not a directory"])
    try:
        paths = sorted(data_dir.glob("*.md"))
    except OSError as exc:
        raise CorpusError([f"{data_dir}:1: cannot list corpus directory: {exc.strerror}"]) from exc
    if not paths:
        raise CorpusError([f"{data_dir}:1: corpus directory contains no Markdown files"])
    documents = tuple(
        document for path in paths if (document := _parse_document(path, problems)) is not None
    )
    if problems:
        raise CorpusError(problems)
    return Corpus(documents, _corpus_hash(documents))


def section_documents(corpus: Corpus) -> list[Document]:
    """Export sections as LangChain documents with JSON-serializable citation metadata."""
    return [
        Document(
            page_content="\n".join(
                [
                    document.title,
                    section.heading,
                    *(sentence.text for sentence in section.sentences),
                ]
            ),
            metadata={
                "section_key": section.key,
                "doc_id": section.doc_id,
                "heading": section.heading,
                "sentence_ids": [sentence.id for sentence in section.sentences],
            },
        )
        for document in corpus.documents
        for section in document.sections
    ]
