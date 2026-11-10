"""Real-format round trips, adversarial imports and evidence portability, all offline."""

import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from docx import Document
from docx.oxml import OxmlElement
from lxml.etree import Element
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from test_index import FakeClient
from test_pipeline import rig as rig
from test_pipeline import structured
from test_retrieval_eval import RankedClient

import citarilo.cli as cli
import citarilo.importer as importer
from citarilo.budget import BudgetLedger
from citarilo.config import Settings
from citarilo.corpus import load_corpus, section_documents
from citarilo.evals import (
    EvidenceMappingError,
    evaluate_answers,
    evaluate_retrieval,
    load_cases,
    load_probes,
    resolve_cases,
    resolve_probes,
)
from citarilo.importer import (
    MAX_DOCX_EXPANDED_BYTES,
    MAX_SOURCE_BYTES,
    DocumentImportError,
    import_documents,
    normalize,
    split_sentences,
)
from citarilo.index import IndexMismatchError, UsageTrackingEmbeddings, build_index, load_index

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "examples/trailsupply-raw"


@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("imported")
    report = import_documents(sorted(RAW.iterdir()), directory)
    return directory, load_corpus(directory), report


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Platí od 1. 3. 2026. Další pravidlo.", ["Platí od 1. 3. 2026.", "Další pravidlo."]),
        ("Platí od 1.3.2026. Dále.", ["Platí od 1.3.2026.", "Dále."]),
        ("Otevíráme ve 12:00. Zavřeme v 18:00.", ["Otevíráme ve 12:00.", "Zavřeme v 18:00."]),
        ("Cena je 1 299 Kč. Další stojí 1,50 Kč.", ["Cena je 1 299 Kč.", "Další stojí 1,50 Kč."]),
        ("Je to 3. objednávka. Platí 1. ledna.", ["Je to 3. objednávka.", "Platí 1. ledna."]),
        ("Je to 100. objednávka. Děkujeme.", ["Je to 100. objednávka.", "Děkujeme."]),
        ("Přijďte 1. Března 2026. Platí to.", ["Přijďte 1. Března 2026.", "Platí to."]),
        (
            "Např. boty, tj. zboží č. 5, atd. přijímáme.",
            ["Např. boty, tj. zboží č. 5, atd. přijímáme."],
        ),
        ("Trail s.r.o. přijímá platby. Hotovo!", ["Trail s.r.o. přijímá platby.", "Hotovo!"]),
        (
            "Viz https://firma.example/a.b?q=3. Potvrďte.",
            ["Viz https://firma.example/a.b?q=3.", "Potvrďte."],
        ),
        (
            "Pište na jmeno.prijmeni@firma.example. Děkujeme.",
            ["Pište na jmeno.prijmeni@firma.example.", "Děkujeme."],
        ),
        ("Platí? „Ano!“ Děkujeme.", ["Platí?", "„Ano!“", "Děkujeme."]),
        ("  První.\nDruhá. ", ["První.", "Druhá."]),
        ("Bez koncové tečky", ["Bez koncové tečky"]),
        ("", []),
    ],
)
def test_czech_sentence_boundaries(text, expected):
    assert split_sentences(text) == expected


def test_normalization_headings_lists_hyphenation_and_warnings(tmp_path):
    output, warnings = normalize(
        "# Pravidla\n\n> Neověřená data.\n\nText před sekcí.\n\n"
        "## Příliš žluťoučký\nZboží je nepo\u00ad\nužité. Další věta.\n"
        "- Samostatná položka\n- Jiná položka\n## Příliš žluťoučký\nPoslední věta.\n"
        "## Prilis zlutoucky 2\n" + "x" * 601 + ".\n",
        fallback_title="fallback",
    )
    (tmp_path / "policy.md").write_text(output)
    doc = load_corpus(tmp_path).documents[0]
    assert [s.section_id for s in doc.sections] == [
        "introduction",
        "prilis-zlutoucky",
        "prilis-zlutoucky-2",
        "prilis-zlutoucky-2-2",
    ]
    assert [s.text for s in doc.sections[1].sentences] == [
        "Zboží je nepoužité.",
        "Další věta.",
        "Samostatná položka",
        "Jiná položka",
    ]
    assert len(warnings) == 2
    assert "first section" in warnings[0] and "very long sentence" in warnings[1]
    assert "Neověřená data." in doc.preamble


def test_plain_document_gets_a_title_and_intro_section():
    output, warnings = normalize("Pouze text dokumentu.", fallback_title="Pravidla")
    assert output.startswith("# Pravidla\n") and "{#introduction}" in output
    assert len(warnings) == 2


def test_date_at_the_start_of_a_body_line_is_not_a_numbered_list():
    output, _ = normalize(
        "# Policy\n## Platnost\n1. 3. 2026 začíná platnost.\n2. Další položka.\n3. Poslední.\n",
        fallback_title="Policy",
    )
    assert "\n1. 3. 2026 začíná platnost.\nDalší položka.\nPoslední.\n" in output


@pytest.mark.parametrize("text", ["x" * 4097 + ".", "| A | B |\n| --- | --- |\n| x | y |"])
def test_canonical_size_and_structure_limits(tmp_path, text):
    source = tmp_path / "source.md"
    source.write_text("# Policy\n## Rule\n" + text)
    with pytest.raises(DocumentImportError):
        import_documents([source], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_raw_formats_round_trip_every_sentence_and_section(imported):
    _, actual, report = imported
    expected = load_corpus(ROOT / "data/trailsupply")
    assert (report.documents, report.sections, report.sentences, report.warnings) == (
        6,
        36,
        146,
        (),
    )
    for source, result in zip(expected.documents, actual.documents, strict=True):
        assert (source.doc_id, source.title) == (result.doc_id, result.title)
        for a, b in zip(source.sections, result.sections, strict=True):
            assert a.heading == b.heading
            assert [s.text for s in a.sentences] == [s.text for s in b.sentences]
    assert expected.corpus_hash != actual.corpus_hash
    assert [d.page_content for d in section_documents(expected)] == [
        d.page_content for d in section_documents(actual)
    ]  # Exactly the same embedding/generation inputs; only citation metadata changes.


def test_pdf_extraction_joins_wrapped_words_and_removes_page_furniture(tmp_path):
    source = tmp_path / "policy.pdf"
    canvas = Canvas(str(source), pagesize=(600, 800))
    for number in (1, 2):
        canvas.setFont("Helvetica", 10)
        canvas.drawString(60, 770, "Company header")
        canvas.drawString(60, 30, "Company footer")
        canvas.drawString(500, 30, f"Page {number}")
        if number == 1:
            canvas.setFont("Helvetica", 20)
            canvas.drawString(60, 710, "Policy")
            canvas.setFont("Helvetica", 14)
            canvas.drawString(60, 670, "Condition")
        canvas.setFont("Helvetica", 11)
        for y, text in zip(
            (630, 610, 590),
            ("Items must be un-", "used. Wrapped", "lines stay together."),
            strict=True,
        ):
            canvas.drawString(60, y, text)
        canvas.showPage()
    canvas.save()
    report = import_documents([source], tmp_path / "out")
    document = load_corpus(tmp_path / "out").documents[0]
    assert document.title == "Policy"
    assert [section.heading for section in document.sections] == ["Condition"]
    assert [s.text for s in document.sections[0].sentences] == [
        "Items must be un-used.",
        "Wrapped lines stay together.",
    ] * 2
    assert len(report.warnings) == 2
    assert all("hard-hyphen line join retained for review: un-used" in w for w in report.warnings)


@pytest.mark.parametrize("title_style", ["Title", "Heading 1"])
def test_docx_styles_lists_and_paragraph_boundaries(tmp_path, title_style):
    source = tmp_path / "policy.docx"
    document = Document()
    document.add_paragraph("Policy", style=title_style)
    document.add_paragraph("Condition", style="Heading 2")
    document.add_paragraph("First paragraph without punctuation")
    document.add_paragraph("Second paragraph without punctuation")
    document.add_paragraph("Bullet item", style="List Bullet")
    document.add_paragraph("Numbered item", style="List Number")
    document.add_paragraph("Final sentence.")
    document.save(source)
    report = import_documents([source], tmp_path / "out")
    actual = load_corpus(tmp_path / "out").documents[0]
    assert actual.title == "Policy"
    assert [section.heading for section in actual.sections] == ["Condition"]
    assert [s.text for s in actual.sections[0].sentences] == [
        "First paragraph without punctuation",
        "Second paragraph without punctuation",
        "Bullet item",
        "Numbered item",
        "Final sentence.",
    ]
    assert report.warnings == ()


def test_docx_tables_are_rejected_instead_of_silently_losing_evidence(tmp_path):
    source = tmp_path / "policy.docx"
    document = Document()
    document.add_heading("Policy", level=1)
    document.add_heading("Condition", level=2)
    document.add_paragraph("See the conditions below.")
    document.add_table(rows=1, cols=1).cell(0, 0).text = "Essential condition."
    document.save(source)
    with pytest.raises(DocumentImportError, match="tables are not supported"):
        import_documents([source], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_generated_documents_are_reproducible(tmp_path):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_sample_documents.py"), "--out", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(p.name for p in RAW.iterdir())
    for expected in RAW.iterdir():
        assert (tmp_path / expected.name).read_bytes() == expected.read_bytes(), expected.name


@pytest.mark.parametrize("suffix", [".txt", ".html", ".docm", ".xls", ""])
def test_unsupported_types(tmp_path, suffix):
    source = tmp_path / ("source" + suffix)
    source.write_text("Untrusted input")
    with pytest.raises(DocumentImportError, match="unsupported type"):
        import_documents([source], tmp_path / "out")


@pytest.mark.parametrize("suffix", [".pdf", ".docx", ".md"])
def test_invalid_or_empty_files(tmp_path, suffix):
    source = tmp_path / f"source{suffix}"
    source.write_bytes(b"")
    with pytest.raises(DocumentImportError):
        import_documents([source], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_scanned_pdf_rejected_without_ocr(tmp_path):
    source = tmp_path / "scan.pdf"
    canvas = Canvas(str(source))
    canvas.rect(20, 20, 100, 100, fill=True)
    canvas.showPage()
    canvas.save()
    with pytest.raises(DocumentImportError, match="no extractable text.*OCR"):
        import_documents([source], tmp_path / "out")


def test_pdf_page_limit(tmp_path):
    source = tmp_path / "pages.pdf"
    canvas = Canvas(str(source))
    for _ in range(101):
        canvas.showPage()
    canvas.save()
    with pytest.raises(DocumentImportError, match="exceeds 100 pages"):
        import_documents([source], tmp_path / "out")


def test_source_size_limit_checked_before_conversion(tmp_path):
    source = tmp_path / "large.md"
    with source.open("wb") as stream:
        stream.truncate(MAX_SOURCE_BYTES + 1)
    with pytest.raises(DocumentImportError, match="20 MiB"):
        import_documents([source], tmp_path / "out")


@pytest.mark.parametrize("unsafe", ["macro", "external", "entity", "expansion"])
def test_untrusted_docx_is_rejected(tmp_path, unsafe):
    source = tmp_path / "unsafe.docx"
    with ZipFile(source, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", "<document/>")
        if unsafe == "macro":
            archive.writestr("word/vbaProject.bin", b"untrusted macro")
        elif unsafe == "external":
            archive.writestr(
                "word/_rels/document.xml.rels",
                '<Relationships><Relationship TargetMode="External" Target="https://invalid.example"/></Relationships>',
            )
        elif unsafe == "entity":
            archive.writestr(
                "word/styles.xml",
                '<!DOCTYPE x [<!ENTITY secret SYSTEM "file:///etc/passwd">]><x>&secret;</x>',
            )
        else:
            archive.writestr("large.bin", b"x" * (MAX_DOCX_EXPANDED_BYTES + 1))
    expected = {
        "macro": "macros",
        "external": "external relationships",
        "entity": "cannot extract text",
        "expansion": "expanded size/member limit",
    }[unsafe]
    with pytest.raises(DocumentImportError, match=expected):
        import_documents([source], tmp_path / "out")


def test_batch_failure_does_not_publish_partial_corpus(tmp_path):
    source = tmp_path / "valid.md"
    source.write_text("# Title\n\n## Policy\nText.\n")
    invalid = tmp_path / "bad.md"
    invalid.write_text("# Title\n\n## Policy\n<script>Unsafe</script>\n")
    with pytest.raises(DocumentImportError):
        import_documents([source, invalid], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_overwrite_requires_force_and_does_not_follow_output_symlinks(tmp_path):
    source = tmp_path / "source.md"
    source.write_text("# Title\n\n## Policy\nText.\n")
    out = tmp_path / "out"
    import_documents([source], out)
    destination = out / "source.md"
    before = destination.read_bytes()
    with pytest.raises(DocumentImportError, match="--force"):
        import_documents([source], out)
    assert destination.read_bytes() == before
    destination.unlink()
    victim = tmp_path / "victim"
    victim.write_text("Keep me")
    destination.symlink_to(victim)
    with pytest.raises(DocumentImportError, match="--force"):
        import_documents([source], out)
    import_documents([source], out, force=True)
    assert victim.read_text() == "Keep me"
    assert not destination.is_symlink()


def test_output_collisions_and_in_place_conversion_rejected(tmp_path):
    source = tmp_path / "source.md"
    source.write_text("# Title\n\n## Policy\nText.\n")
    for force in (False, True):
        with pytest.raises(DocumentImportError, match="source and output"):
            import_documents([source], tmp_path, force=force)
    with pytest.raises(DocumentImportError, match="Duplicate output"):
        import_documents([source, source], tmp_path / "out")


def test_import_cli_never_loads_settings(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Import must not read settings, credentials or .env")

    monkeypatch.setattr(cli, "Settings", forbidden)
    cli.main(["import", str(RAW / "orders.md"), "--out", str(tmp_path / "out")])
    assert "Imported 1 documents; 6 sections; 24 sentences; 0 warnings." in capsys.readouterr().out
    with pytest.raises(SystemExit) as failure:
        cli.main(["import", str(RAW / "orders.md"), "--out", str(tmp_path / "out")])
    assert failure.value.code == 1
    assert "--force" in capsys.readouterr().err


def test_all_evaluation_labels_map_to_imported_evidence(imported):
    _, corpus, _ = imported
    canonical = load_corpus(ROOT / "data/trailsupply")
    original = load_cases(ROOT / "evals/cases.jsonl")
    cases = resolve_cases(original, corpus)
    for a, b in zip(original, cases, strict=True):
        assert [canonical.sentence(s).text for s in a.supporting_sentence_ids] == [
            corpus.sentence(s).text for s in b.supporting_sentence_ids
        ]
        assert [canonical.section(s).heading for s in a.expected_section_keys] == [
            corpus.section(s).heading for s in b.expected_section_keys
        ]
    probes = load_probes(ROOT / "evals/verifier_cases.jsonl")
    for a, b in zip(probes, resolve_probes(probes, corpus), strict=True):
        assert [canonical.sentence(s).text for s in a.cited_sentence_ids] == [
            corpus.sentence(s).text for s in b.cited_sentence_ids
        ]
    assert resolve_cases(original, canonical) == original


@pytest.mark.parametrize("duplicate", [False, True])
def test_missing_or_ambiguous_exact_mapping_fails_before_provider(tmp_path, duplicate):
    canonical = load_corpus(ROOT / "data/trailsupply")
    text = canonical.sentence("shipping.standard-prices.3").text if duplicate else "Different text."
    for name in ("one", "two"):
        (tmp_path / f"{name}.md").write_text(f"# Policy\n\n## Costs {{#costs}}\n{text}\n")
    corpus = load_corpus(tmp_path)
    with pytest.raises(EvidenceMappingError, match="expected one exact sentence text match"):
        resolve_cases(load_cases(ROOT / "evals/cases.jsonl")[:1], corpus)


def test_cli_explains_missing_evaluation_evidence(tmp_path, capsys):
    (tmp_path / "policy.md").write_text("# Policy\n## Costs {#costs}\nDifferent text.\n")
    settings = Settings(_env_file=None, data_dir=tmp_path, runtime_dir=tmp_path / "runtime")
    with pytest.raises(SystemExit) as failure:
        cli.main(["eval", "--live", "--dry-run"], settings=settings)
    assert failure.value.code == 1
    assert "Cannot map evaluation ID shipping.standard-prices.3" in capsys.readouterr().err
    assert not settings.runtime_dir.exists()


def test_fake_retrieval_evaluation_uses_imported_section_ids(imported, tmp_path):
    directory, corpus, _ = imported
    settings = Settings(
        _env_file=None, data_dir=directory, runtime_dir=tmp_path, embedding_dimensions=36
    )
    client = RankedClient(settings)
    cases = load_cases(ROOT / "evals/cases.jsonl")
    client.cases = {c.question: c for c in resolve_cases(cases, corpus)}
    ledger = BudgetLedger(
        settings.spend_db_path,
        settings.budget_cap_usd,
        settings.price_table_version,
        price_table=settings.price_table,
    )
    build_index(corpus, settings, ledger, client=client)
    report, _ = evaluate_retrieval(
        [c for c in cases if c.split == "dev"], settings, Decimal("0.01"), client=client
    )
    assert report.gate == "PASS" and report.metrics.recall_at_5 == 1
    assert report.cases[0].expected_section_keys == ("shipping.ceny-beznych-zasilek",)


def test_switching_corpora_requires_ingest_and_new_manifest(imported, tmp_path):
    directory, corpus, _ = imported
    canonical = load_corpus(ROOT / "data/trailsupply")
    settings = Settings(
        _env_file=None,
        data_dir=ROOT / "data/trailsupply",
        runtime_dir=tmp_path,
        embedding_dimensions=8,
    )
    ledger = BudgetLedger(
        settings.spend_db_path,
        settings.budget_cap_usd,
        settings.price_table_version,
        price_table=settings.price_table,
    )
    client = FakeClient()
    build_index(canonical, settings, ledger, client=client)
    settings.data_dir = directory
    embeddings = UsageTrackingEmbeddings(settings, ledger, group_id="load", client=client)
    with pytest.raises(IndexMismatchError):
        load_index(corpus, settings, embeddings)
    manifest = build_index(corpus, settings, ledger, client=client)
    assert manifest.corpus_hash == corpus.corpus_hash
    with load_index(corpus, settings, embeddings):
        pass


def test_fake_answer_evaluation_uses_imported_citations(imported, rig, section_index_factory):
    directory, corpus, _ = imported
    settings = rig.context.settings
    settings.data_dir = directory
    store = section_index_factory(
        [
            d
            for d in section_documents(corpus)
            if d.metadata["section_key"] == "shipping.ceny-beznych-zasilek"
        ],
        rig.context.embeddings,
    )
    context = replace(rig.context, corpus=corpus, store=store)
    sid = "shipping.ceny-beznych-zasilek.3"
    rig.results["generate"] = structured(
        {
            "status": "answered",
            "answer_language": "en",
            "claims": [
                {
                    "text": "Standard shipping is free for goods worth at least "
                    "CZK 1,299 after all discounts.",
                    "citations": [sid],
                }
            ],
        }
    )
    report, _ = evaluate_answers(
        load_cases(ROOT / "evals/cases.jsonl")[:1], settings, Decimal("0.05"), context=context
    )
    assert report.complete and report.cases[0].status == "answered"
    assert report.cases[0].claims[0].citations == (sid,)
    assert report.cases[0].supporting_sentence_ids[0] == sid
    assert report.cases[0].sources[0].text == corpus.sentence(sid).text


def test_pdf_margin_removal_never_removes_body_numbers_or_matching_sentences(tmp_path):
    source = tmp_path / "margins.pdf"
    canvas = Canvas(str(source), pagesize=(600, 800))
    for page in (1, 2):
        canvas.setFont("Helvetica", 11)
        canvas.drawString(60, 770, "Repeated header")
        canvas.drawString(60, 30, "Same sentence.")
        canvas.drawString(500, 30, str(page))
        canvas.drawString(60, 745, "Edge body.")
        canvas.drawString(60, 700, "500")
        canvas.drawString(60, 680, "Same sentence.")
        canvas.drawString(60, 660, "Repeated header")
        canvas.showPage()
    canvas.save()
    assert (
        importer._extract_pdf(source.read_bytes()).splitlines()
        == [
            "Edge body.",
            "500",
            "Same sentence.",
            "Repeated header",
        ]
        * 2
    )


def test_pdf_unique_margin_text_is_retained(tmp_path):
    source = tmp_path / "margin.pdf"
    canvas = Canvas(str(source), pagesize=(600, 800))
    canvas.drawString(60, 770, "Unique condition.")
    canvas.drawString(60, 700, "Body.")
    canvas.drawString(60, 30, "500")
    canvas.save()
    assert importer._extract_pdf(source.read_bytes()).splitlines() == [
        "Unique condition.",
        "Body.",
        "500",
    ]


def test_pdf_inline_font_changes_preserve_baseline_and_word_order(tmp_path):
    source = tmp_path / "fonts.pdf"
    canvas = Canvas(str(source), pagesize=(600, 800))
    text = canvas.beginText(60, 700)
    for font, size, words in (
        ("Helvetica", 11, "First "),
        ("Helvetica-Bold", 24, "bold"),
        ("Helvetica", 11, " word in"),
        ("Helvetica-Oblique", 14, "line"),
        ("Helvetica", 11, " ends here."),
    ):
        text.setFont(font, size)
        text.textOut(words)
    canvas.drawText(text)
    canvas.drawString(60, 670, "Next line.")
    canvas.save()
    assert importer._extract_pdf(source.read_bytes()).splitlines() == [
        "First bold word inline ends here.",
        "Next line.",
    ]


@pytest.mark.parametrize("staggered", [False, True])
def test_pdf_columns_rejected_instead_of_interleaving(tmp_path, staggered):
    source = tmp_path / "columns.pdf"
    canvas = Canvas(str(source), pagesize=(600, 800))
    canvas.setFont("Helvetica", 11)
    for y in (700, 670):
        canvas.drawString(60, y, "Left column.")
        canvas.drawString(340, y - (15 if staggered else 0), "Right column.")
    canvas.save()
    with pytest.raises(
        DocumentImportError, match="page 1: multi-column.*convert to text or Markdown"
    ):
        import_documents([source], tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "cells",
    [
        # Narrow 20 pt gutter between two text columns.
        ((60, "Delivery requires"), (230, "Returns require")),
        # Borderless table: label and value separated by whitespace only.
        ((60, "Tent deposit"), (260, "5 000 Kč")),
    ],
)
def test_pdf_narrow_columns_and_borderless_tables_rejected(tmp_path, cells):
    source = tmp_path / "layout.pdf"
    canvas = Canvas(str(source), pagesize=(600, 800))
    canvas.setFont("Helvetica", 11)
    for y in (700, 685):
        for x, text in cells:
            canvas.drawString(x, y, text)
    canvas.save()
    with pytest.raises(DocumentImportError, match="page 1: multi-column"):
        import_documents([source], tmp_path / "out")


def test_pdf_bordered_table_rejected_instead_of_flattened(tmp_path):
    source = tmp_path / "table.pdf"
    canvas = Canvas(str(source), pagesize=(600, 800))
    for x in (60, 250, 440):
        canvas.line(x, 600, x, 700)
    for y in (600, 650, 700):
        canvas.line(60, y, 440, y)
    for x, y, text in (
        (70, 675, "Product"),
        (260, 675, "Price"),
        (70, 625, "Tent"),
        (260, 625, "500"),
    ):
        canvas.drawString(x, y, text)
    canvas.save()
    with pytest.raises(DocumentImportError, match="page 1: tables.*convert to text or Markdown"):
        import_documents([source], tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("content", ["image", "drawing"])
@pytest.mark.parametrize("margin_text", [False, True])
def test_pdf_nontext_page_in_text_document_is_rejected_with_page_number(
    tmp_path, content, margin_text
):
    from PIL import Image

    source = tmp_path / "mixed.pdf"
    canvas = Canvas(str(source), pagesize=(600, 800))
    canvas.drawString(60, 700, "Extractable policy text.")
    canvas.showPage()
    if margin_text:
        canvas.drawString(60, 770, "Company header")
        canvas.drawString(60, 30, "Page 2")
    if content == "image":
        image = BytesIO()
        Image.new("RGB", (10, 10), color="black").save(image, format="PNG")
        image.seek(0)
        canvas.drawImage(ImageReader(image), 60, 500, width=100, height=100)
    else:
        canvas.rect(60, 500, 100, 100, fill=True)
    canvas.showPage()
    canvas.save()
    with pytest.raises(DocumentImportError, match="page 2: no extractable text.*OCR"):
        import_documents([source], tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "tag,message",
    [
        ("sdt", "content controls.*convert"),
        ("ins", "tracked changes.*accept or reject"),
        ("del", "tracked changes.*accept or reject"),
        ("txbxContent", "text boxes.*convert"),
        ("tbl", "tables.*convert tables"),
        ("AlternateContent", "AlternateContent.*convert"),
    ],
)
def test_docx_unsupported_nested_body_structures_are_rejected(tmp_path, tag, message):
    source = tmp_path / "nested.docx"
    document = Document()
    document.add_heading("Policy", level=1)
    document.add_heading("Condition", level=2)
    document.add_paragraph("Visible condition.")
    wrapper = OxmlElement("w:customXml")
    unsupported = (
        Element("{http://schemas.openxmlformats.org/markup-compatibility/2006}AlternateContent")
        if tag == "AlternateContent"
        else OxmlElement(f"w:{tag}")
    )
    nested = OxmlElement("w:p")
    run = OxmlElement("w:r")
    text = OxmlElement("w:t")
    text.text = "Essential hidden condition."
    run.append(text)
    nested.append(run)
    unsupported.append(nested)
    wrapper.append(unsupported)
    document.element.body.append(wrapper)
    document.save(source)
    with pytest.raises(DocumentImportError, match=message):
        import_documents([source], tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("wrapper_tag", ["w:customXml", "w:fldSimple"])
def test_docx_text_outside_plain_paragraphs_is_rejected_not_dropped(tmp_path, wrapper_tag):
    source = tmp_path / "wrapped.docx"
    document = Document()
    document.add_heading("Policy", level=1)
    paragraph = document.add_paragraph("Returns are accepted.")
    wrapper = OxmlElement(wrapper_tag)
    run = OxmlElement("w:r")
    text = OxmlElement("w:t")
    text.text = "Only within 14 days."
    run.append(text)
    if wrapper_tag == "w:customXml":
        nested = OxmlElement("w:p")
        nested.append(run)
        wrapper.append(nested)
        document.element.body.append(wrapper)
    else:
        wrapper.append(run)
        paragraph._p.append(wrapper)
    document.save(source)
    with pytest.raises(DocumentImportError, match="text outside plain paragraphs"):
        import_documents([source], tmp_path / "out")


@pytest.mark.parametrize("title_style", ["Title", "Heading 1"])
def test_docx_only_first_title_becomes_document_title(tmp_path, title_style):
    source = tmp_path / "titles.docx"
    document = Document()
    document.add_paragraph("Policy", style=title_style)
    document.add_heading("First section", level=1)
    document.add_paragraph("First condition.")
    document.add_heading("Second section", level=1)
    document.add_paragraph("Second condition.")
    if title_style == "Title":
        document.add_paragraph("Later title", style="Title")
        document.add_paragraph("Third condition.")
    document.save(source)
    import_documents([source], tmp_path / "out")
    actual = load_corpus(tmp_path / "out").documents[0]
    assert actual.title == "Policy"
    assert [s.heading for s in actual.sections] == ["First section", "Second section"] + (
        ["Later title"] if title_style == "Title" else []
    )
    assert [s.text for section in actual.sections for s in section.sentences] == [
        "First condition.",
        "Second condition.",
    ] + (["Third condition."] if title_style == "Title" else [])


def test_docx_explicit_title_takes_precedence_over_earlier_heading_one(tmp_path):
    source = tmp_path / "title.docx"
    document = Document()
    document.add_heading("First section", level=1)
    document.add_paragraph("Condition.")
    document.add_paragraph("Policy", style="Title")
    document.save(source)
    import_documents([source], tmp_path / "out")
    actual = load_corpus(tmp_path / "out").documents[0]
    assert actual.title == "Policy"
    assert actual.sections[0].heading == "First section"
    assert actual.sections[0].sentences[0].text == "Condition."


@pytest.mark.parametrize("month", ["ledna", "Ledna", "Března"])
def test_leading_named_month_date_is_preserved(month):
    text = f"1. {month} začíná platnost."
    output, warnings = normalize(f"# Policy\n## Dates\n{text}", fallback_title="Policy")
    assert output.endswith(text + "\n")
    assert warnings == ()


def test_numbered_markers_require_a_consecutive_uppercase_list_block():
    output, _ = normalize(
        "# Policy\n## Conditions\n1. ledna začíná platnost.\n2. února končí platnost.\n\n"
        "1. First item.\n2) Second item.\n\nA paragraph.\n\n4. lowercase ordinal.",
        fallback_title="Policy",
    )
    assert "\n1. ledna začíná platnost.\n2. února končí platnost.\n" in output
    assert "\nFirst item.\nSecond item.\n" in output
    assert "\n4. lowercase ordinal.\n" in output


def test_docx_numbered_lists_require_numbering_metadata(tmp_path):
    source = tmp_path / "lists.docx"
    document = Document()
    document.add_heading("Policy", level=1)
    document.add_heading("Conditions", level=2)
    document.styles.add_style("List Pretend", 1)
    document.add_paragraph("1. First literal ordinal.", style="List Pretend")
    document.add_paragraph("2. Second literal ordinal.", style="List Pretend")
    document.add_paragraph("1. ledna začíná platnost.", style="List Number")
    document.add_paragraph("Final list item", style="List Number")
    document.add_paragraph("- Literal prefix", style="List Pretend")
    document.save(source)
    import_documents([source], tmp_path / "out")
    output = (tmp_path / "out/lists.md").read_text()
    assert "\n1.\nFirst literal ordinal.\n2.\nSecond literal ordinal.\n" in output
    assert "\n1. ledna začíná platnost.\nFinal list item\n" in output
    assert "\n- - Literal prefix\n" in output
    actual = load_corpus(tmp_path / "out").documents[0]
    assert actual.sections[0].sentences[-1].text == "- Literal prefix"


def test_sentence_final_count_is_not_an_ordinal():
    assert split_sentences("Počet kusů je 5. Doprava je zdarma.") == [
        "Počet kusů je 5.",
        "Doprava je zdarma.",
    ]


@pytest.mark.parametrize("opening,closing", [('"', '"'), ("„", "“"), ("«", "»"), ("'", "'")])
@pytest.mark.parametrize("punctuation", [".", "!", "?"])
def test_quoted_url_does_not_protect_following_sentence_punctuation(opening, closing, punctuation):
    sentence = f"Viz {opening}https://firma.example/a.b{closing}{punctuation}"
    assert split_sentences(sentence + " Další věta.") == [sentence, "Další věta."]


def test_email_matching_skips_long_tokens_and_tokens_without_at(monkeypatch):
    original = importer._EMAIL
    seen = []

    class BoundedEmail:
        def finditer(self, token):
            assert "@" in token and len(token) <= 320
            seen.append(token)
            return original.finditer(token)

    monkeypatch.setattr(importer, "_EMAIL", BoundedEmail())
    text = "x" * 32_000 + " " + "y" * 32_000 + "@invalid " + "a.b@firma.example. Děkujeme."
    assert split_sentences(text) == [text.removesuffix(" Děkujeme."), "Děkujeme."]
    assert seen == ["a.b@firma.example."]


@pytest.mark.parametrize("suffix", [".md", ".docx"])
def test_only_soft_hyphens_are_removed_and_hard_joins_are_reported(tmp_path, suffix):
    source = tmp_path / f"hyphens{suffix}"
    body = "Nakupujte e-\nshopem. Zboží je nepo\u00ad\nužité."
    if suffix == ".docx":
        document = Document()
        document.add_heading("Policy", level=1)
        document.add_heading("Conditions", level=2)
        document.add_paragraph(body)
        document.save(source)
    else:
        source.write_text("# Policy\n## Conditions\n" + body)
    report = import_documents([source], tmp_path / "out")
    actual = load_corpus(tmp_path / "out").documents[0]
    assert [s.text for s in actual.sections[0].sentences] == [
        "Nakupujte e-shopem.",
        "Zboží je nepoužité.",
    ]
    assert report.warnings == (
        f"hyphens{suffix}: hard-hyphen line join retained for review: e-shopem",
    )


def _corpus_with_replaced_text(corpus, replacements, *, keep_hash=False):
    documents = tuple(
        replace(
            doc,
            sections=tuple(
                replace(
                    section,
                    sentences=tuple(
                        replace(sentence, text=replacements.get(sentence.id, sentence.text))
                        for sentence in section.sentences
                    ),
                )
                for section in doc.sections
            ),
        )
        for doc in corpus.documents
    )
    return replace(
        corpus, documents=documents, corpus_hash=corpus.corpus_hash if keep_hash else "changed"
    )


@pytest.mark.parametrize("kind", ["case", "probe"])
def test_evaluation_existing_ids_are_resolved_by_canonical_text(kind):
    canonical = load_corpus(ROOT / "data/trailsupply")
    first, second = "shipping.standard-prices.1", "shipping.standard-prices.3"
    active = _corpus_with_replaced_text(
        canonical,
        {first: canonical.sentence(second).text, second: canonical.sentence(first).text},
    )
    if kind == "case":
        original = load_cases(ROOT / "evals/cases.jsonl")[:1]
        mapped = resolve_cases(original, active)
        assert mapped[0].supporting_sentence_ids == (first, "shipping.standard-prices.4")
    else:
        original = load_probes(ROOT / "evals/verifier_cases.jsonl")[:1]
        mapped = resolve_probes(original, active)
        assert mapped[0].cited_sentence_ids == (second, "shipping.standard-prices.2")


@pytest.mark.parametrize("kind", ["case", "probe"])
@pytest.mark.parametrize("duplicate", [False, True])
def test_evaluation_existing_id_with_missing_or_duplicate_text_fails(kind, duplicate):
    canonical = load_corpus(ROOT / "data/trailsupply")
    sid = "shipping.standard-prices.3" if kind == "case" else "shipping.standard-prices.1"
    replacements = (
        {"shipping.delivery-area.1": canonical.sentence(sid).text}
        if duplicate
        else {sid: "Different evidence."}
    )
    active = _corpus_with_replaced_text(canonical, replacements)
    resolver, records = (
        (resolve_cases, load_cases(ROOT / "evals/cases.jsonl")[:1])
        if kind == "case"
        else (resolve_probes, load_probes(ROOT / "evals/verifier_cases.jsonl")[:1])
    )
    with pytest.raises(
        EvidenceMappingError, match=f"exact sentence text match, found {2 if duplicate else 0}"
    ):
        resolver(records, active)


def test_evaluation_canonical_hash_still_requires_identical_id_text():
    canonical = load_corpus(ROOT / "data/trailsupply")
    active = _corpus_with_replaced_text(
        canonical, {"shipping.standard-prices.3": "Wrong evidence."}, keep_hash=True
    )
    with pytest.raises(EvidenceMappingError, match="standard-prices.3: text mismatch"):
        resolve_cases(load_cases(ROOT / "evals/cases.jsonl")[:1], active)


def test_evaluation_existing_section_key_is_verified_through_all_sentence_texts():
    canonical = load_corpus(ROOT / "data/trailsupply")
    a, b = (
        canonical.section("shipping.standard-prices"),
        canonical.section("shipping.delivery-area"),
    )
    replacements = {
        target.id: source.text
        for targets, sources in ((a.sentences, b.sentences), (b.sentences, a.sentences))
        for target, source in zip(targets, sources, strict=True)
    }
    active = _corpus_with_replaced_text(canonical, replacements)
    mapped = resolve_cases(load_cases(ROOT / "evals/cases.jsonl")[:1], active)
    assert mapped[0].supporting_sentence_ids == (
        "shipping.delivery-area.3",
        "shipping.delivery-area.4",
    )
    assert mapped[0].expected_section_keys == ("shipping.delivery-area",)


def test_chained_hard_hyphen_joins_are_all_retained_and_reported():
    output, warnings = normalize(
        "# Policy\n## Conditions\nNakupujte e-\nshop-\nem.", fallback_title="Policy"
    )
    assert output.endswith("Nakupujte e-shop-em.\n")
    assert warnings == (
        "hard-hyphen line join retained for review: e-shop",
        "hard-hyphen line join retained for review: shop-em",
    )
