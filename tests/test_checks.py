"""Literal normalization against real corpus facts, without model judgments."""

from pathlib import Path

import pytest

from citarilo.checks import check_claim, check_claims, normalized_facts
from citarilo.corpus import load_corpus
from citarilo.evals import load_cases
from citarilo.records import CandidateClaim


@pytest.mark.parametrize(
    ("claim", "evidence"),
    [
        ("CZK 1,299", "1 299 Kč"),
        ("1,299 CZK", "1\u00a0299 Kč"),
        ("0.5%", "0,5 %"),
        ("10 percent", "10 %"),
        ("1 March 2026", "1. 3. 2026"),
        ("March 1, 2026", "1. 3. 2026"),
        ("March 1st, 2026", "1. 3. 2026"),
        ("March 2nd, 2026", "2. 3. 2026"),
        ("3rd March 2026", "3. 3. 2026"),
        ("March 14th, 2026", "14. 3. 2026"),
        ("2026-03-01", "1. 3. 2026"),
        ("8 a.m.", "8:00"),
        ("4 p.m.", "16:00"),
        ("12 a.m.", "0:00"),
        ("12 p.m.", "12:00"),
        ("two days", "do dvou dnů"),
        ("three days", "do tří dnů"),
        ("five days", "do pěti dnů"),
        ("seven days", "do sedmi dnů"),
        ("fourteen days", "do čtrnácti dnů"),
        ("thirty days", "do třiceti dnů"),
        ("forty-eight hours", "48 hodin"),
        ("one hundred cm", "100 cm"),
        ("twenty-one days", "21 dnů"),
        ("five thousand CZK", "5 000 Kč"),
        ("five-thousand CZK", "pět tisíc Kč"),
        ("one hundred and twenty-one days", "sto dvacet jedna dnů"),
        ("two hundred thirty-five CZK", "dvě stě třicet pět Kč"),
        ("five thousand two hundred and forty-eight CZK", "5 248 Kč"),
        ("one hundred twenty-three thousand four hundred fifty-six CZK", "123 456 Kč"),
        ("The service is set up online.", "Služba se nastavuje online."),
        ("https://trailsupply.example/vraceni.", "https://trailsupply.example/vraceni"),
        ("vraceni@trailsupply.example", "vraceni@trailsupply.example."),
    ],
)
def test_equivalent_literals(claim: str, evidence: str) -> None:
    assert normalized_facts(claim, language="en") == normalized_facts(evidence, language="cs")
    assert check_claim(CandidateClaim(text=claim, citations=("x",)), {"x": evidence}).ok


@pytest.mark.parametrize(
    ("claim", "evidence"),
    [
        ("3 January 2026", "1. 3. 2026"),
        ("0.5%", "5 %"),
        ("10%", "10 dnů"),
        ("8 a.m.", "18:00"),
        ("48 hours", "24 hodin"),
        ("CZK 2,000", "5 000 Kč"),
        ("one hundred", "1 a 1000"),
        ("five thousand CZK", "5 Kč"),
        ("twenty-one days", "20 nebo 1 den"),
        ("https://trailsupply.example/evil", "https://trailsupply.example/vraceni"),
        ("https://trailsupply.example/Path", "https://trailsupply.example/path"),
        ("evil@trailsupply.example", "vraceni@trailsupply.example"),
        ("1 March 2026", "1 den, 3 měsíce, 2026 kusů"),
    ],
)
def test_changed_literals_fail(claim: str, evidence: str) -> None:
    result = check_claim(CandidateClaim(text=claim, citations=("x",)), {"x": evidence})
    assert not result.ok
    assert result.reasons == ("literal_not_in_citations",)


def test_only_cited_retrieved_sentences_can_supply_literals() -> None:
    result = check_claim(
        CandidateClaim(text="The fee is CZK 999.", citations=("x",)),
        {"x": "Cena je 79 Kč.", "y": "Jiná cena je 999 Kč."},
    )
    assert not result.ok
    missing = check_claim(CandidateClaim(text="A policy.", citations=("unknown",)), {"x": "Text"})
    assert missing.reasons == ("unknown_citation",)


@pytest.mark.parametrize("citations", [(), ("x", "x"), ("x", "y", "z", "w")])
def test_invalid_citation_counts(citations: tuple[str, ...]) -> None:
    assert (
        "invalid_citation_count"
        in check_claim(
            CandidateClaim(text="A policy.", citations=citations),
            {sid: "Text" for sid in citations},
        ).reasons
    )


def test_all_answered_eval_facts_pass_with_their_supporting_evidence() -> None:
    # Offline contract check over BOTH splits; no retrieval/model tuning or live eval.
    corpus = load_corpus(Path("data/trailsupply"))
    for case in load_cases(Path("evals/cases.jsonl")):
        if case.expected_status != "answered":
            continue
        sentences = {sid: corpus.sentence(sid).text for sid in case.supporting_sentence_ids}
        for fact in case.required_facts:
            # Find a <=3 citation subset satisfying the literal filter. This does
            # not assert semantic support, which is separately hand-labelled.
            from itertools import combinations

            matches = [
                check_claim(CandidateClaim(text=fact, citations=ids), sentences)
                for count in range(1, min(3, len(sentences)) + 1)
                for ids in combinations(sentences, count)
            ]
            assert any(result.ok for result in matches), (case.id, fact)
        evidence_facts = frozenset().union(
            *(normalized_facts(text, language="cs") for text in sentences.values())
        )
        for forbidden in case.forbidden_claims:
            if normalized_facts(forbidden) - evidence_facts:
                for sid in sentences:
                    assert not check_claim(
                        CandidateClaim(text=forbidden, citations=(sid,)), sentences
                    ).ok, (case.id, forbidden)


def test_filter_leaves_relationships_and_exceptions_to_the_verifier() -> None:
    sentences = {"x": "Na adresu 79 Kč, výdejní místo 59 Kč."}
    assert check_claim(
        CandidateClaim(
            text="Address delivery costs CZK 59 and collection costs CZK 79.", citations=("x",)
        ),
        sentences,
    ).ok


def test_numeric_normalization_is_exact_under_low_decimal_precision() -> None:
    from decimal import localcontext

    with localcontext() as context:
        context.prec = 2
        assert normalized_facts("123456789") != normalized_facts("123456790")


def test_malformed_url_is_treated_as_data_without_crashing() -> None:
    assert normalized_facts("https://[malformed/path") == {"url:https://[malformed/path"}


def test_number_words_do_not_leak_between_languages() -> None:
    assert normalized_facts("set up a service", language="en") == frozenset()
    assert normalized_facts("pět tisíc", language="en") == frozenset()
    assert normalized_facts("five thousand", language="cs") == frozenset()
    assert normalized_facts("pět set", language="cs") == {"number:500"}


def test_english_claims_reject_czech_prose_but_allow_quoted_labels() -> None:
    sentences = {"returns.return-window.2": "U zboží označeného štítkem Výprodej je lhůta 14 dnů."}
    czech = CandidateClaim(
        text="Lhůta pro výprodej je 14 dnů.", citations=("returns.return-window.2",)
    )
    quoted = CandidateClaim(
        text='Goods labelled "Výprodej" have a 14-day window.',
        citations=("returns.return-window.2",),
    )
    assert check_claim(czech, sentences).reasons == ("language_mismatch",)
    assert check_claim(czech, sentences, answer_language="cs").ok
    assert check_claim(quoted, sentences).ok
    currency = CandidateClaim(text="The fee is 14 Kč.", citations=("returns.return-window.2",))
    assert check_claim(currency, sentences).ok


@pytest.mark.parametrize(
    ("claim", "evidence"),
    [
        ("Kauce za stan je 5 000 Kč.", "Kauce za stan činí pět tisíc Kč."),
        ("Kauce za stan je pět tisíc Kč.", "Kauce za stan činí 5 000 Kč."),
        ("Platba přijde do pěti pracovních dnů.", "Lhůta činí 5 pracovních dnů."),
        ("Platba přijde do pěti pracovních dnů.", "Refunds take 5 working days."),
        ("Platba přijde do 5 pracovních dnů.", "Lhůta činí pět pracovních dnů."),
        ("Zboží vraťte do třiceti dnů.", "Vrácení je možné do 30 dnů."),
        ("Sleva činí deset procent.", "Sleva činí 10 %."),
        ("Poplatek je 0,5 %.", "Poplatek činí 0,5 procenta."),
        ("Rozměr je 1,299 metru.", "Rozměr činí 1,299 m."),
        ("Cena je 1\u202f299,50 Kč.", "Cena je 1 299,5 Kč."),
        ("Platí od 1. března 2026.", "Platí od 1. 3. 2026."),
        ("Platí od 1. 3. 2026.", "Platí od 1. března 2026."),
        ("Platí od 2026-03-01.", "Platí od 1. března 2026."),
    ],
)
def test_czech_claim_literals(claim: str, evidence: str) -> None:
    assert normalized_facts(claim, language="cs") == normalized_facts(evidence, language="cs")
    (result,) = check_claims(
        [CandidateClaim(text=claim, citations=("x",))], {"x": evidence}, answer_language="cs"
    )
    assert result.ok


@pytest.mark.parametrize(
    ("claim", "evidence"),
    [
        ("Kauce za stan je 2 000 Kč.", "Kauce za stan činí 5 000 Kč."),
        ("Platba přijde do sedmi pracovních dnů.", "Lhůta činí 5 pracovních dnů."),
        ("Platba přijde do pěti pracovních dnů.", "Lhůta činí sedm pracovních dnů."),
        ("Rozměr je 1,299 m.", "Rozměr činí 1 299 m."),
        ("Platí od 1. března 2026.", "Platí od 1. 4. 2026."),
        ("Platí od 1. 3. 2026.", "Platí od 1. dubna 2026."),
        ("Sleva činí deset procent.", "Lhůta činí deset dnů."),
    ],
)
def test_changed_czech_literals_fail(claim: str, evidence: str) -> None:
    result = check_claim(
        CandidateClaim(text=claim, citations=("x",)), {"x": evidence}, answer_language="cs"
    )
    assert result.reasons == ("literal_not_in_citations",)


def test_czech_claims_have_no_heuristic_language_check() -> None:
    # The generator judges language; the literal filter deliberately cannot do so.
    claim = CandidateClaim(text="Returns are accepted.", citations=("x",))
    assert check_claim(claim, {"x": "Zboží lze vrátit."}, answer_language="cs").ok


def test_literals_echoed_from_the_question_are_allowed() -> None:
    sentences = {"returns.return-window.2": "Lhůta pro vrácení je 14 kalendářních dnů."}
    claim = CandidateClaim(
        text="Ne, lhůta je 14 dnů, takže po 20 dnech už uplynula.",
        citations=("returns.return-window.2",),
    )
    assert not check_claim(claim, sentences, answer_language="cs").ok
    assert check_claim(
        claim, sentences, answer_language="cs", question="Muzu vratit zbozi po 20 dnech?"
    ).ok
