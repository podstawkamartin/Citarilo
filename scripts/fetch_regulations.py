#!/usr/bin/env python3
"""Fetch pinned e-Sbírka texts and convert their fragments to citable Markdown."""

from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://e-sbirka.gov.cz/sbr-cache/dokumenty-sbirky/"


@dataclass(frozen=True)
class Regulation:
    number: int
    year: int
    version: str
    title: str
    fragment_count: int


REGULATIONS = (
    Regulation(
        190,
        2022,
        "2024-07-01",
        "o vyhrazených technických elektrických zařízeních a požadavcích "
        "na zajištění jejich bezpečnosti",
        240,
    ),
    Regulation(
        192,
        2022,
        "2024-07-01",
        "o vyhrazených technických tlakových zařízeních a požadavcích "
        "na zajištění jejich bezpečnosti",
        511,
    ),
    Regulation(
        193,
        2022,
        "2024-07-01",
        "o vyhrazených technických zdvihacích zařízeních a požadavcích "
        "na zajištění jejich bezpečnosti",
        304,
    ),
    Regulation(
        378,
        2001,
        "2003-01-01",
        "kterým se stanoví bližší požadavky na bezpečný provoz a používání "
        "strojů, technických zařízení, přístrojů a nářadí",
        170,
    ),
)


def clean(value: str) -> str:
    value = value.replace("\u00a0", " ").replace("[", "(").replace("]", ")")
    value = value.replace("|", "/").replace("`", "").replace("\\", "")
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", value).strip())


class TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.sup_depth = 0
        self.after_sup = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "sup":
            self.sup_depth += 1
        elif tag == "br":
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "sup" and self.sup_depth:
            self.sup_depth -= 1
            self.after_sup = True

    def handle_data(self, data: str) -> None:
        if self.sup_depth:
            return
        if self.after_sup:
            data = re.sub(r"^\s*\)", "", data, count=1)
            self.after_sup = False
        self.parts.append(data)


def text_of(html: str) -> str:
    parser = TextParser()
    parser.feed(html)
    parser.close()
    return clean("".join(parser.parts))


@dataclass
class Cell:
    text: str
    header: bool
    colspan: int
    rowspan: int


class TableParser(TextParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[Cell]] = []
        self.row: list[Cell] | None = None
        self.cell: tuple[bool, int, int] | None = None
        self.cell_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            attributes = dict(attrs)
            self.cell_count += 1
            if self.cell_count > 10000:
                raise ValueError("Table exceeds the cell limit")
            spans = []
            for name in ("colspan", "rowspan"):
                raw = attributes.get(name, "1")
                if raw is None or not re.fullmatch(r"[0-9]+", raw):
                    raise ValueError(f"Invalid {name}")
                value = int(raw)
                if not 1 <= value <= 100:
                    raise ValueError(f"Invalid {name}")
                spans.append(value)
            self.cell = (tag == "th", *spans)
            self.parts = []
        else:
            super().handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self.cell is not None:
            assert self.row is not None
            self.row.append(Cell(clean("".join(self.parts)), *self.cell))
            self.cell = None
            self.parts = []
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None
        else:
            super().handle_endtag(tag)


def table_grid(
    rows: list[list[Cell]],
) -> list[tuple[list[str], list[bool], list[bool], bool]]:
    """Expand spanning cells; mark cells inherited from a previous physical row."""
    active: dict[int, tuple[int, str, bool]] = {}
    result = []
    for row in rows:
        values: list[str] = []
        inherited: list[bool] = []
        spanning: list[bool] = []
        col = 0
        all_headers = bool(row) and all(cell.header for cell in row)
        for cell in row:
            while col in active:
                remaining, value, is_spanning = active[col]
                values.append(value)
                inherited.append(True)
                spanning.append(is_spanning)
                if remaining == 1:
                    del active[col]
                else:
                    active[col] = (remaining - 1, value, is_spanning)
                col += 1
            for position in range(cell.colspan):
                values.append(cell.text)
                inherited.append(False)
                spanning.append(position > 0)
                if cell.rowspan > 1:
                    active[col] = (cell.rowspan - 1, cell.text, position > 0)
                col += 1
        while col in active:
            remaining, value, is_spanning = active[col]
            values.append(value)
            inherited.append(True)
            spanning.append(is_spanning)
            if remaining == 1:
                del active[col]
            else:
                active[col] = (remaining - 1, value, is_spanning)
            col += 1

        while values and not values[-1] and not inherited[-1] and not all_headers:
            values.pop()
            inherited.pop()
            spanning.pop()
        result.append((values, inherited, spanning, all_headers))
    width = max((len(values) for values, _, _, _ in result), default=0)
    for values, inherited, spanning, _ in result:
        padding = width - len(values)
        values.extend([""] * padding)
        inherited.extend([False] * padding)
        spanning.extend([False] * padding)
    return result


def sentence(value: str) -> str:
    value = clean(value)
    if value.endswith((",", ";")):
        return value[:-1] + "."
    return value


def render_table(html: str, context: str, table_title: str) -> list[str]:
    parser = TableParser()
    parser.feed(html)
    parser.close()
    grid = table_grid(parser.rows)
    labels: list[str] = []
    caption = table_title.rstrip(". :")
    logical: list[tuple[list[str], list[str], str]] = []
    for (values, inherited, spanning, is_header), source_row in zip(grid, parser.rows, strict=True):
        if is_header:
            # A full-width header is a caption; otherwise combine nested column labels.
            if len(set(values)) == 1 and len(values) > 1:
                caption = values[0].rstrip(". :")
                labels = []
            elif (
                not labels
                or not any(labels)
                or (all(values) and not any(cell.colspan > 1 for cell in source_row))
            ):
                labels = [v.rstrip(": ") for v in values]
            else:
                labels += [""] * (len(values) - len(labels))
                for i, value in enumerate(values):
                    if value:
                        parent = labels[i]
                        if parent == "Lhůty" and value in ("revize", "zkoušky"):
                            labels[i] = f"Lhůta {value}"
                        elif parent and value != parent:
                            labels[i] = f"{parent} – {value}"
                        else:
                            labels[i] = value.rstrip(": ")
            continue
        if len(source_row) == 1 and source_row[0].colspan > 1 and source_row[0].text:
            logical.append(([source_row[0].text], [], caption))
            continue
        values = [
            "" if is_spanning else value
            for value, is_spanning in zip(values, spanning, strict=True)
        ]
        # Physical continuation in tables with explicit blank first cells.
        if logical and values and not values[0] and not inherited[0]:
            previous, prev_labels, prev_caption = logical[-1]
            previous += [""] * max(0, len(values) - len(previous))
            for i, value in enumerate(values):
                if value:
                    previous[i] = clean(previous[i] + " " + value)
            logical[-1] = (previous, prev_labels, prev_caption)
        else:
            logical.append((values[:], labels[:], caption))
    lines = []
    for values, row_labels, row_caption in logical:
        # A cell spanning all columns is a standalone table note.
        if len(values) > 1 and len(set(values)) == 1:
            lines.append(sentence((row_caption + ": " if row_caption else "") + values[0]))
            continue
        pieces = []
        for i, value in enumerate(values):
            if not value:
                continue
            label = row_labels[i] if i < len(row_labels) else ""
            if (
                not label
                and i == len(values) - 1
                and re.fullmatch(r"[A-Z]{1,3}\d?(?:/[A-Z]{1,3}\d?)?", value)
            ):
                label = "rozlišovací znak"
            if not label and i == 0 and re.fullmatch(r"[a-z]\)", value):
                value = f"písm. {value}"
            if i < len(values) - 1 and not re.fullmatch(r"[IVXLCDM]+\.", value):
                value = value.rstrip(".;, ")
            pieces.append(f"{label}: {value}" if label else value)
        if len(pieces) > 1 and re.fullmatch(r"písm\. [a-z]\)", pieces[0]):
            pieces[:2] = [pieces[0] + " " + pieces[1]]
        prefix = row_caption or (context.rstrip(". :") if not row_labels else "")
        line = "; ".join(pieces)
        if prefix:
            line = prefix + ": " + line
        if line:
            result_line = sentence(line)
            lines.append(
                result_line if result_line.endswith((".", "!", "?")) else result_line + "."
            )
    return lines


def metadata_title(regulation: Regulation) -> str:
    return f"Nařízení vlády č. {regulation.number}/{regulation.year} Sb., {regulation.title}"


def get_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "Citarilo-regulations/1.0"})
    with urlopen(request, timeout=60) as response:
        content = response.read(8 * 1024 * 1024 + 1)
    if len(content) > 8 * 1024 * 1024:
        raise ValueError(f"Response exceeds byte limit: {url}")
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError(f"Unexpected e-Sbírka response: {url}")
    return result


def fetch(regulation: Regulation) -> list[dict]:
    path = f"/sb/{regulation.year}/{regulation.number}/{regulation.version}"
    url = BASE + quote(path, safe="")
    metadata = get_json(url)
    expected = {
        "zkracenaCitace": f"Nařízení vlády č. {regulation.number}/{regulation.year} Sb.",
        "uplnaCitace": metadata_title(regulation),
        "datumUcinnostiZneniOd": regulation.version,
        "typZneni": "AKTUALNI",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Unexpected metadata {key} at {url}: {metadata.get(key)!r}, expected {value!r}"
            )
    fragments: list[dict] = []
    page = 0
    while True:
        payload = get_json(f"{url}/fragmenty?cisloStranky={page}")
        page_url = f"{url}/fragmenty?cisloStranky={page}"
        try:
            pages, records = payload["pocetStranek"], payload["seznam"]
        except KeyError as exc:
            raise ValueError(f"Missing field in response: {page_url}") from exc
        if type(pages) is not int or not page < pages <= 100 or not isinstance(records, list):
            raise ValueError(f"Invalid fragment pagination: {page_url}")
        for record in records:
            if not isinstance(record, dict) or not all(
                k in record for k in ("eli", "kodTypuFragmentu")
            ):
                raise ValueError(f"Missing fragment field in response: {page_url}")
        fragments.extend(records)
        if len(fragments) > 5000:
            raise ValueError(f"Fragment limit exceeded: {page_url}")
        if page + 1 == pages:
            break
        page += 1
    if len(fragments) != regulation.fragment_count:
        raise ValueError(
            f"Unexpected fragment count at {url}: {len(fragments)}; "
            f"expected {regulation.fragment_count}"
        )
    return fragments


def fragment_marker(html: str) -> str:
    match = re.search(r"<var(?:\s[^>]*)?>(.*?)</var>", html, re.DOTALL | re.IGNORECASE)
    return text_of(match.group(1)) if match else ""


def without_marker(text: str, marker: str) -> str:
    return text[len(marker) :].strip() if marker and text.startswith(marker) else text


def line_size(line: str) -> int:
    size = len((line + "\n").encode("utf-8"))
    if size > 6000 or len(line) > 4096:
        raise ValueError("Oversized fragment")
    return size


def pack_annex(
    annex_id: str, lines: list[str], boundaries: list[tuple[int, str]]
) -> list[tuple[str, list[str]]]:
    """Pack consecutive top-level points, retaining the introduction with the first point."""
    groups: list[tuple[str | None, list[str]]] = []
    start = 0
    group_id: str | None = None
    for position, next_id in boundaries:
        if position > start:
            groups.append((group_id, lines[start:position]))
        start, group_id = position, next_id
    if start < len(lines):
        groups.append((group_id, lines[start:]))

    reserved = {ident for _, ident in boundaries}
    used = {annex_id}
    chunks: list[tuple[str, list[str]]] = []
    chunk: list[str] = []
    chunk_id = annex_id
    size = 0
    intro_only = False

    def next_id(group: str) -> str:
        if group not in used:
            used.add(group)
            return group
        number = 2
        while f"{group}-part-{number}" in used | reserved:
            number += 1
        result = f"{group}-part-{number}"
        used.add(result)
        return result

    for group, contents in groups:
        group_size = sum(line_size(line) for line in contents)
        # An introduction without a point has no separate continuation identifier.
        group = group or annex_id
        if chunk and size + group_size > 6000 and not intro_only:
            chunks.append((chunk_id, chunk))
            chunk, size = [], 0
            chunk_id = next_id(group)
        # Split a single oversized point only at fragment boundaries. If the first
        # point cannot fit with its introduction, keep as much with it as possible.
        for line in contents:
            length = line_size(line)
            if chunk and size + length > 6000:
                chunks.append((chunk_id, chunk))
                chunk, size = [], 0
                chunk_id = next_id(group)
            chunk.append(line)
            size += length
        intro_only = group == annex_id and not chunks
    if chunk:
        chunks.append((chunk_id, chunk))
    return chunks


def convert(regulation: Regulation, fragments: list[dict]) -> str:
    """Convert a complete ordered fragment list to the restricted corpus format."""
    # Each entry is a heading, stable ID and a list of complete citable lines.
    sections: list[tuple[str, str, list[str], list[tuple[int, str]]]] = []
    heading = ""
    section_id = ""
    current: list[str] = []
    boundaries: list[tuple[int, str]] = []
    annex = ""
    parent_markers: dict[str, str] = {}
    leadins: dict[str, str] = {}
    table_title = ""
    context = ""
    pending_note = ""

    def section(name: str, ident: str) -> None:
        nonlocal heading, section_id, current, boundaries
        if current:
            sections.append((heading, section_id, current, boundaries))
        heading, section_id, current, boundaries = name, ident, [], []

    def add(value: str, *, continuation: bool = False) -> None:
        nonlocal pending_note
        if not value:
            return
        if pending_note:
            value = pending_note + " " + value
            pending_note = ""
        if continuation and current:
            current[-1] = sentence(current[-1] + " " + value)
        else:
            current.append(sentence(value))

    previous_kind = ""
    previous_eli = ""
    for fragment in fragments:
        kind = fragment.get("kodTypuFragmentu", "")
        eli = fragment.get("eli", "")
        html = fragment.get("xhtml") or ""
        text = text_of(html)
        prior_kind, previous_kind = previous_kind, kind
        prior_eli, previous_eli = previous_eli, eli
        handled = {
            "Paragraf",
            "Hlavicka_priloha",
            "Nadpis",
            "Nadpis_nad",
            "Nadpis_pod",
            "Cast",
            "Bod_Rd",
            "Bod_Dd",
            "Pismeno_Lb",
            "Odstavec_Dc",
            "Pred_Odrazka",
            "Pokracovani_Text",
            "Uni_Odrazka",
            "Poznamka",
            "Tabulka",
        }
        dropped = {
            "Prefix",
            "Prefix_Number",
            "Prefix_Type",
            "Prefix_Date",
            "Prefix_Title",
            "Postfix",
            "PPC",
            "Virtual_Document",
            "Virtual_Norma",
            "Virtual_PPC",
            "Virtual_Postfix",
            "Virtual_Prefix",
            "Virtual_Prilohy",
        }
        if text and kind not in handled and kind not in dropped:
            raise ValueError(f"Unhandled fragment type {kind}: {eli}")
        if kind == "Paragraf":
            match = re.search(r"/par_([0-9]+[a-z]?)", eli)
            if not match:
                raise ValueError(f"Unknown paragraph ID: {eli}")
            annex = ""
            section(text, "p-" + match.group(1))
        elif kind == "Hlavicka_priloha":
            match = re.search(r"/priloha_(\d+)", eli)
            if not match:
                raise ValueError(f"Unknown annex ID: {eli}")
            annex = "a" + match.group(1)
            section(f"Příloha č. {match.group(1)}", annex)
            table_title = context = ""
        elif kind in ("Nadpis", "Nadpis_nad", "Nadpis_pod"):
            if not text or not heading:
                continue
            if not current and (
                kind == "Nadpis_pod"
                and not annex
                or annex
                and "/cast_" not in eli
                and "/bod_" not in eli
                and heading == f"Příloha č. {annex[1:]}"
            ):
                heading += " " + text
            elif annex and kind == "Nadpis" and text.lower().startswith("tabulka"):
                table_title = text
            elif annex and kind in ("Nadpis_pod", "Nadpis_nad"):
                add(text)
        elif kind == "Cast":
            continue
        elif kind in (
            "Bod_Rd",
            "Bod_Dd",
            "Pismeno_Lb",
            "Odstavec_Dc",
            "Pred_Odrazka",
            "Pokracovani_Text",
            "Uni_Odrazka",
        ):
            if not text or not heading:
                continue
            marker = fragment_marker(html)
            parent_markers[eli] = marker
            pieces = eli.split("/")
            if annex and kind in ("Bod_Rd", "Bod_Dd"):
                rest = eli.split("/priloha_", 1)[1].split("/", 1)[-1]
                parts = rest.split("/")
                if (
                    len(parts) == 1
                    and parts[0].startswith("bod_")
                    or len(parts) == 2
                    and parts[0].startswith("cast_")
                    and parts[1].startswith("bod_")
                ):
                    tokens = [
                        p.replace("cast_", "c").replace("bod_", "b").replace(":", "-").lower()
                        for p in parts
                    ]
                    ident = annex + "-" + "-".join(tokens)
                    boundaries.append((len(current), ident))
            prefix = []
            for index, part in enumerate(pieces):
                if part.startswith("odst_") and (kind != "Odstavec_Dc" or not marker):
                    prefix.append("odst. " + part[5:].split(":", 1)[0])
                elif part.startswith("pism_") and (
                    kind != "Pismeno_Lb" or index == len(pieces) - 1
                ):
                    prefix.append("písm. " + part[5:].split(":", 1)[0] + ")")
                elif part.startswith("bod_") and not annex:
                    prefix.append("bod " + part[4:].split(":", 1)[0] + ".")
            if annex:
                ancestors = ["/".join(pieces[:i]) for i in range(1, len(pieces))]
                outlines = [
                    parent_markers[p]
                    for p in ancestors
                    if p in parent_markers
                    and p.rsplit("/", 1)[-1].startswith("bod_")
                    and parent_markers[p]
                ]
                if kind == "Pismeno_Lb" and outlines:
                    prefix.insert(0, outlines[-1])
                if kind in ("Bod_Dd", "Bod_Rd"):
                    prefix = [marker] if marker else []
            if kind == "Odstavec_Dc" and marker:
                prefix = [marker]
            content = without_marker(text, marker)
            if kind == "Pismeno_Lb" and not prefix:
                prefix = ["písm. " + marker]
            parent = eli.rsplit("/", 1)[0]
            if kind == "Odstavec_Dc":
                leadins[eli if marker else parent] = content.rstrip(" :,")
            elif kind in ("Pismeno_Lb", "Bod_Rd", "Bod_Dd", "Pred_Odrazka"):
                leadins[eli if kind != "Pred_Odrazka" else parent] = content.rstrip(" :,")

            is_item = kind in ("Pismeno_Lb", "Uni_Odrazka") or (
                kind in ("Bod_Rd", "Bod_Dd") and not annex
            )
            if kind == "Uni_Odrazka" and annex and outlines:
                prefix.insert(0, outlines[-1])
            line = clean(" ".join([*prefix, content]))
            if is_item:
                parents = ["/".join(pieces[:i]) for i in range(1, len(pieces))]
                # Numbered paragraphs take precedence over unnumbered paragraph text.
                if any(part.startswith("odst_") for part in pieces):
                    parents = [p for p in parents if not p.rsplit("/", 1)[-1].startswith("par_")]
                if annex and any(part.startswith("bod_") for part in pieces):
                    parents = [
                        p for p in parents if not p.rsplit("/", 1)[-1].startswith("priloha_")
                    ]
                introductions = [leadins[p] for p in parents if leadins.get(p)]
                chain = ": ".join(introductions)
                marker_chain = " ".join(prefix)
                if introductions:
                    if len(chain) > 600 or len(marker_chain + chain + content) > 4000:
                        chain = introductions[-1]
                    line = clean(f"{marker_chain} – {chain}: {content}")
            if kind in ("Bod_Rd", "Bod_Dd") or (
                annex and kind == "Odstavec_Dc" and not marker and "/bod_" not in eli
            ):
                context = without_marker(text, marker)
            continuation = kind == "Pokracovani_Text" or (
                kind == "Odstavec_Dc"
                and not marker
                and ("/pism_" in eli or "/bod_" in eli)
                and (
                    prior_kind not in ("Bod_Rd", "Bod_Dd", "Pismeno_Lb")
                    or prior_eli == eli.rsplit("/", 1)[0]
                )
            )
            add(content if continuation else line, continuation=continuation)
            table_title = ""
        elif kind == "Poznamka":
            pending_note = text
        elif kind == "Tabulka":
            for line in render_table(html, context, table_title):
                add(line)
            table_title = ""
    if pending_note:
        add(pending_note)
    if current:
        sections.append((heading, section_id, current, boundaries))

    date = {"2024-07-01": "1. 7. 2024", "2003-01-01": "1. 1. 2003"}[regulation.version]
    output = [
        f"# {metadata_title(regulation)}",
        "",
        f"> Zdroj: e-Sbírka, znění účinné od {date}.",
        "> Úřední dílo podle § 3 písm. a) zákona č. 121/2000 Sb.",
        "> Tabulky příloh jsou převedeny na věty a poznámky pod čarou jsou vynechány.",
    ]
    frequencies = Counter(line for _, _, lines, _ in sections for line in lines)
    used: set[str] = set()
    for title, ident, lines, boundaries in sections:
        lines = [f"{title.strip()}: {line}" if frequencies[line] > 1 else line for line in lines]
        title = title.strip()
        if ident.startswith("a"):
            chunks = pack_annex(ident, lines, boundaries)
        else:
            chunks = []
            chunk: list[str] = []
            chunk_bytes = 0
            for line in lines:
                size = line_size(line)
                if chunk and chunk_bytes + size > 6000:
                    chunks.append((ident if not chunks else f"{ident}-{len(chunks) + 1}", chunk))
                    chunk, chunk_bytes = [], 0
                chunk.append(line)
                chunk_bytes += size
            if chunk:
                chunks.append((ident if not chunks else f"{ident}-{len(chunks) + 1}", chunk))
        for number, (chunk_id, chunk) in enumerate(chunks):
            if chunk_id in used:
                raise ValueError(f"Duplicate section ID {chunk_id}")
            used.add(chunk_id)
            suffix = " (pokračování)" if number else ""
            output.extend(["", f"## {title}{suffix} {{#{chunk_id}}}", *chunk])
    return "\n".join(output) + "\n"


def validate(regulation: Regulation, markdown: str) -> None:
    import tempfile

    from citarilo.corpus import load_corpus

    if len(markdown.encode("utf-8")) > 256 * 1024:
        raise ValueError("Document exceeds the corpus file size limit")
    if any(len(line) > 4096 for line in markdown.splitlines()):
        raise ValueError("Line exceeds the corpus length limit")
    with tempfile.TemporaryDirectory(prefix="citarilo-regulation-") as directory:
        path = Path(directory) / f"nv-{regulation.number}-{regulation.year}.md"
        path.write_text(markdown, encoding="utf-8")
        corpus = load_corpus(Path(directory))
        for item in corpus.sections():
            if sum(len((s.text + "\n").encode("utf-8")) for s in item.sentences) > 6000:
                raise ValueError(f"Section exceeds the byte limit: {item.key}")
        counts = Counter(s.text for s in corpus.sentences())
        duplicates = {line: count for line, count in counts.items() if count > 1}
        for line, count in duplicates.items():
            print(f"Duplicate in {path.name} ({count}): {line}")


def main() -> None:
    destination = ROOT / "data" / "regulations"
    if not destination.is_dir():
        raise FileNotFoundError(destination)
    with tempfile.TemporaryDirectory(prefix=".regulations-", dir=destination) as directory:
        staged = []
        for regulation in REGULATIONS:
            markdown = convert(regulation, fetch(regulation))
            validate(regulation, markdown)
            name = f"nv-{regulation.number}-{regulation.year}.md"
            path = Path(directory) / name
            path.write_text(markdown, encoding="utf-8")
            validate(regulation, path.read_text(encoding="utf-8"))
            staged.append((path, destination / name))
        for source, target in staged:
            os.replace(source, target)
            print(f"Wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
