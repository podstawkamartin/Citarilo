"""Offline regression tests for the pinned e-Sbírka converter."""

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "fetch_regulations", Path(__file__).resolve().parents[1] / "scripts/fetch_regulations.py"
)
assert spec and spec.loader
regulations = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = regulations
spec.loader.exec_module(regulations)

FIXTURES = Path(__file__).parent / "fixtures" / "regulations"


def fixture() -> dict:
    return json.loads((FIXTURES / "fragments.json").read_text(encoding="utf-8"))


def test_recorded_fragments_match_golden_and_load_as_corpus() -> None:
    result = regulations.convert(regulations.REGULATIONS[0], fixture()["fragments"])
    assert result == (FIXTURES / "expected.md").read_text(encoding="utf-8")
    assert "odst. 2 písm. a)" in result
    assert "síti. Doplňující pokračování." in result
    assert "<" not in result and "PPC" not in result
    assert "Na všech zařízeních LPS" in result
    assert "písm. a:2)" not in result
    assert "rozlišovací znak: E2." in result
    assert "Revizní lhůty: 3 roky." in result
    regulations.validate(regulations.REGULATIONS[0], result)
    assert regulations.convert(regulations.REGULATIONS[0], fixture()["fragments"]) == result


def test_recorded_multirow_header_and_values() -> None:
    title, table = fixture()["multi_header"]
    lines = regulations.render_table(table["xhtml"], "", regulations.text_of(title["xhtml"]))
    assert lines[1] == (
        "Tabulka č. 1 – lhůty revizí a zkoušek jeřábů a zdvihadel: "
        "Skupina jeřábu: 2; Součinitel spektra zatížení a třídy Q: 0,25 (Q3); "
        "Lhůta revize: 3 roky; Lhůta zkoušky: 6 roků."
    )


def test_pinned_metadata_and_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    regulation = replace(regulations.REGULATIONS[0], fragment_count=2)
    metadata = {
        "zkracenaCitace": "Nařízení vlády č. 190/2022 Sb.",
        "uplnaCitace": regulations.metadata_title(regulation),
        "datumUcinnostiZneniOd": regulation.version,
        "typZneni": "AKTUALNI",
    }
    calls = []

    def get(url: str) -> dict:
        calls.append(url)
        if "fragmenty?" not in url:
            return metadata
        page = int(url.rsplit("=", 1)[1])
        return {"pocetStranek": 2, "seznam": [{"eli": str(page), "kodTypuFragmentu": "Virtual_X"}]}

    monkeypatch.setattr(regulations, "get_json", get)
    assert [x["eli"] for x in regulations.fetch(regulation)] == ["0", "1"]
    assert "%2Fsb%2F2022%2F190%2F2024-07-01" in calls[0]
    metadata["uplnaCitace"] = "Changed"
    with pytest.raises(ValueError, match="uplnaCitace"):
        regulations.fetch(regulation)
    metadata["uplnaCitace"] = regulations.metadata_title(regulation)
    metadata["datumUcinnostiZneniOd"] = "2022-07-01"
    with pytest.raises(ValueError, match="datumUcinnostiZneniOd"):
        regulations.fetch(regulation)


def test_section_splitting_preserves_fragments() -> None:
    fragments = [
        {"kodTypuFragmentu": "Paragraf", "eli": "/par_3", "xhtml": "<var>§ 3</var>"},
        *(
            {
                "kodTypuFragmentu": "Odstavec_Dc",
                "eli": f"/par_3/odst_{i}",
                "xhtml": f"<var>({i})</var> " + "ž" * 900,
            }
            for i in range(1, 5)
        ),
    ]
    result = regulations.convert(regulations.REGULATIONS[0], fragments)
    assert "{#p-3-2}" in result
    assert result.count("ž" * 900) == 4
    regulations.validate(regulations.REGULATIONS[0], result)


def test_annex_packing_keeps_groups_and_introduction() -> None:
    lines = ["Úvod", "a" * 2400, "b" * 2400, "c" * 2400]
    chunks = regulations.pack_annex("a2", lines, [(1, "a2-b1"), (2, "a2-b2"), (3, "a2-b3")])
    assert chunks == [("a2", lines[:3]), ("a2-b3", lines[3:])]
    assert regulations.pack_annex("a2", lines[:3], [(1, "a2-b1"), (2, "a2-b2")]) == [
        ("a2", lines[:3])
    ]


def test_annex_oversized_group_keeps_intro_and_stable_ids() -> None:
    lines = ["Úvod", "a" * 3000, "b" * 3000, "c" * 3000, "Další bod"]
    chunks = regulations.pack_annex("a2", lines, [(1, "a2-b1"), (4, "a2-b2")])
    assert chunks == [
        ("a2", lines[:2]),
        ("a2-b1", lines[2:3]),
        ("a2-b1-part-2", lines[3:]),
    ]
    assert all(sum(regulations.line_size(line) for line in chunk) <= 6000 for _, chunk in chunks)


def test_annex_sections_keep_full_heading_and_point_ids() -> None:
    base = "/eli/cz/sb/2022/190/2024-07-01/dokument/prilohy/priloha_2"
    fragments = [
        {"kodTypuFragmentu": "Hlavicka_priloha", "eli": base, "xhtml": "Příloha č. 2"},
        {"kodTypuFragmentu": "Nadpis", "eli": base + "/title", "xhtml": "Nadpis přílohy"},
        {"kodTypuFragmentu": "Pred_Odrazka", "eli": base + "/intro", "xhtml": "Úvod"},
        *(
            {
                "kodTypuFragmentu": "Bod_Dd",
                "eli": base + f"/bod_{i}",
                "xhtml": f"<var>{i}.</var> " + chr(96 + i) * 2400,
            }
            for i in range(1, 4)
        ),
    ]
    result = regulations.convert(regulations.REGULATIONS[0], fragments)
    assert "## Příloha č. 2 Nadpis přílohy {#a2}\nÚvod\n1." in result
    assert "## Příloha č. 2 Nadpis přílohy (pokračování) {#a2-b3}\n3." in result
    assert "{#a2-b1}" not in result and "{#a2-b2}" not in result
    regulations.validate(regulations.REGULATIONS[0], result)


def test_unknown_nonempty_fragment_and_invalid_table_spans_fail() -> None:
    with pytest.raises(ValueError, match="Unhandled fragment type"):
        regulations.convert(
            regulations.REGULATIONS[0],
            [{"kodTypuFragmentu": "New_Type", "eli": "/new", "xhtml": "New text"}],
        )
    for value in ("0", "101", "-1", "many"):
        with pytest.raises(ValueError, match="colspan"):
            regulations.render_table(
                f'<table><tr><td colspan="{value}">x</td></tr></table>', "", ""
            )


def test_no_files_replaced_if_later_regulation_fails(tmp_path, monkeypatch) -> None:
    root = tmp_path
    destination = root / "data" / "regulations"
    destination.mkdir(parents=True)
    paths = [destination / f"nv-{r.number}-{r.year}.md" for r in regulations.REGULATIONS]
    for path in paths:
        path.write_text("original")
    monkeypatch.setattr(regulations, "ROOT", root)
    monkeypatch.setattr(
        regulations,
        "fetch",
        lambda r: [] if r.number == 190 else (_ for _ in ()).throw(ValueError("failed")),
    )
    monkeypatch.setattr(regulations, "convert", lambda r, f: "new")
    monkeypatch.setattr(regulations, "validate", lambda r, m: None)
    with pytest.raises(ValueError, match="failed"):
        regulations.main()
    assert [path.read_text() for path in paths] == ["original"] * len(paths)


def test_unmarked_paragraph_after_list_is_not_joined_to_last_point() -> None:
    base = "/eli/cz/sb/2022/190/2024-07-01/dokument/norma/par_4"
    fragments = [
        {"kodTypuFragmentu": "Paragraf", "eli": base, "xhtml": "<var>§ 4</var>"},
        {
            "kodTypuFragmentu": "Odstavec_Dc",
            "eli": base + "/odst_1",
            "xhtml": "<var>(1)</var> Text",
        },
        {
            "kodTypuFragmentu": "Pismeno_Lb",
            "eli": base + "/odst_1/pism_a",
            "xhtml": "<var>a)</var> Zařízení",
        },
        {
            "kodTypuFragmentu": "Bod_Dd",
            "eli": base + "/odst_1/pism_a/bod_1",
            "xhtml": "<var>1.</var> První a",
        },
        {
            "kodTypuFragmentu": "Bod_Dd",
            "eli": base + "/odst_1/pism_a/bod_2",
            "xhtml": "<var>2.</var> Druhý;",
        },
        {
            "kodTypuFragmentu": "Odstavec_Dc",
            "eli": base + "/odst_1/pism_a/frag_1",
            "xhtml": "Požadavek na dokumentaci,",
        },
    ]
    output = regulations.convert(regulations.REGULATIONS[0], fragments)
    assert "odst. 1 písm. a) bod 1. – Text: Zařízení: První a\n" in output
    assert (
        "odst. 1 písm. a) bod 2. – Text: Zařízení: Druhý.\n"
        "odst. 1 písm. a) Požadavek na dokumentaci.\n"
    ) in output


def test_recorded_rowspan_grid_matches_golden() -> None:
    html = fixture()["rowspan"]["xhtml"]
    parser = regulations.TableParser()
    parser.feed(html)
    rows = regulations.table_grid(parser.rows)
    assert rows[5][1][0] is True
    assert rows[5][0] == [
        "kotle",
        "III.",
        "jedenkrát za 6 měsíců",
        "jedenkrát za 1 rok, u kotlů I. a II. třídy může být prodloužen termín na 2 roky",
        "jedenkrát za 10 let",
    ]
    result = "\n".join(regulations.render_table(html, "", "")) + "\n"
    assert result == (FIXTURES / "rowspan-expected.txt").read_text(encoding="utf-8")
    assert len(result.splitlines()) == 6


def test_rowspan_with_blank_placeholders_keeps_header_caption() -> None:
    rows = regulations.render_table(fixture()["rowspan193"]["xhtml"], "", "")
    assert rows
    assert "§ 3 odst. 1 písm. c): Činnost na vyhrazeném zdvihacím zařízení:" in rows[0]
    assert "písm. c) – Činnost" not in rows[0]
    assert all("písm. c) – Nejnižší" not in row for row in rows)


def test_list_items_include_parent_and_nested_leadins() -> None:
    base = "/eli/cz/sb/2001/378/2003-01-01/dokument/norma/par_3"
    fragments = [
        {"kodTypuFragmentu": "Paragraf", "eli": base, "xhtml": "<var>§ 3</var>"},
        {
            "kodTypuFragmentu": "Odstavec_Dc",
            "eli": base + "/odst_1",
            "xhtml": "<var>(1)</var> Požadavky jsou:",
        },
        {
            "kodTypuFragmentu": "Pismeno_Lb",
            "eli": base + "/odst_1/pism_a",
            "xhtml": "<var>a)</var> ochrana zejména,",
        },
        {
            "kodTypuFragmentu": "Bod_Dd",
            "eli": base + "/odst_1/pism_a/bod_1",
            "xhtml": "<var>1.</var> před teplem;",
        },
        {
            "kodTypuFragmentu": "Odstavec_Dc",
            "eli": base + "/odst_2",
            "xhtml": "<var>(2)</var> Výjimky jsou",
        },
        {
            "kodTypuFragmentu": "Pismeno_Lb",
            "eli": base + "/odst_2/pism_a",
            "xhtml": "<var>a)</var> zařízení A.",
        },
    ]
    output = regulations.convert(regulations.REGULATIONS[3], fragments)
    assert "(1) Požadavky jsou:\n" in output
    assert "odst. 1 písm. a) – Požadavky jsou: ochrana zejména.\n" in output
    assert "odst. 1 písm. a) bod 1. – Požadavky jsou: ochrana zejména: před teplem.\n" in output
    assert "odst. 2 písm. a) – Výjimky jsou: zařízení A.\n" in output


def test_long_context_uses_nearest_leadin() -> None:
    base = "/eli/cz/sb/2001/378/2003-01-01/dokument/norma/par_3"
    fragments = [
        {"kodTypuFragmentu": "Paragraf", "eli": base, "xhtml": "<var>§ 3</var>"},
        {
            "kodTypuFragmentu": "Odstavec_Dc",
            "eli": base + "/odst_1",
            "xhtml": "<var>(1)</var> " + "A" * 590,
        },
        {
            "kodTypuFragmentu": "Pismeno_Lb",
            "eli": base + "/odst_1/pism_a",
            "xhtml": "<var>a)</var> Nejbližší úvod text",
        },
        {
            "kodTypuFragmentu": "Bod_Dd",
            "eli": base + "/odst_1/pism_a/bod_1",
            "xhtml": "<var>1.</var> Hodnota 23 V.",
        },
    ]
    output = regulations.convert(regulations.REGULATIONS[3], fragments)
    assert "odst. 1 písm. a) bod 1. – Nejbližší úvod text: Hodnota 23 V." in output
    regulations.validate(regulations.REGULATIONS[3], output)
