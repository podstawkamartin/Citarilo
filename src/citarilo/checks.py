"""Conservative, table-driven citation filters, never a proof of entailment."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal

from citarilo.records import AnswerLanguage, CandidateClaim, CheckResult

# Include the inflections and compound adjective used by the Czech corpus.
_WORD_GROUPS = {
    0: ("zero", "nula"),
    1: ("one first", "jeden jedna jedno jedné jednu jedním prvním prvních"),
    2: ("two", "dva dvě dvou dvěma dvoumístného"),
    3: ("three", "tři tří třem třemi"),
    4: ("four", "čtyři čtyř čtyřmi"),
    5: ("five", "pět pěti"),
    6: ("six", "šest šesti"),
    7: ("seven", "sedm sedmi"),
    8: ("eight", "osm osmi"),
    9: ("nine", "devět devíti"),
    10: ("ten", "deset deseti"),
    11: ("eleven", "jedenáct jedenácti"),
    12: ("twelve", "dvanáct dvanácti"),
    13: ("thirteen", "třináct třinácti"),
    14: ("fourteen", "čtrnáct čtrnácti"),
    15: ("fifteen", "patnáct patnácti"),
    16: ("sixteen", "šestnáct šestnácti"),
    17: ("seventeen", "sedmnáct sedmnácti"),
    18: ("eighteen", "osmnáct osmnácti"),
    19: ("nineteen", "devatenáct devatenácti"),
    20: ("twenty", "dvacet dvaceti"),
    30: ("thirty", "třicet třiceti"),
    40: ("forty", "čtyřicet čtyřiceti"),
    50: ("fifty", "padesát padesáti"),
    60: ("sixty", "šedesát šedesáti"),
    70: ("seventy", "sedmdesát sedmdesáti"),
    80: ("eighty", "osmdesát osmdesáti"),
    90: ("ninety", "devadesát devadesáti"),
    100: ("hundred", "sto sta stě set"),
    1000: ("thousand", "tisíc tisíce tisíců tisíci"),
}
_WORDS = {
    language: {
        word: number for number, groups in _WORD_GROUPS.items() for word in groups[index].split()
    }
    for index, language in enumerate(("en", "cs"))
}
_WORD_PATTERNS = {
    language: re.compile(r"\b(?:" + "|".join(words) + r")\b", re.I)
    for language, words in _WORDS.items()
}
_WORD_GAPS = {"en": re.compile(r"[\s-]+(?:and[\s-]+)?"), "cs": re.compile(r"[\s-]+(?:a[\s-]+)?")}
_MONTH_GROUPS = {
    "en": [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ],
    "cs": [
        "leden ledna",
        "únor února",
        "březen března",
        "duben dubna",
        "květen května",
        "červen června",
        "červenec července",
        "srpen srpna",
        "září",
        "říjen října",
        "listopad listopadu",
        "prosinec prosince",
    ],
}
_MONTHS = {
    language: {name: index for index, group in enumerate(groups, 1) for name in group.split()}
    for language, groups in _MONTH_GROUPS.items()
}
_NUMERIC_DATES = (
    re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\b"),
    re.compile(r"\b(?P<day>\d{1,2})\.\s*(?P<month>\d{1,2})\.\s*(?P<year>\d{4})\b"),
)
_DATES = {
    "en": (
        *_NUMERIC_DATES,
        re.compile(
            rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month>{'|'.join(_MONTHS['en'])})"
            rf"\s+(?P<year>\d{{4}})\b"
        ),
        re.compile(
            rf"\b(?P<month>{'|'.join(_MONTHS['en'])})\s+(?P<day>\d{{1,2}})"
            r"(?:st|nd|rd|th)?,?\s+(?P<year>\d{4})\b"
        ),
    ),
    "cs": (
        *_NUMERIC_DATES,
        re.compile(
            rf"\b(?P<day>\d{{1,2}})\.?\s+(?P<month>{'|'.join(_MONTHS['cs'])})"
            r"\s+(?P<year>\d{4})\b"
        ),
    ),
}
_TIME = re.compile(
    r"\b(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<period>[ap]\.?m\.?)(?!\w)"
    r"|\b(?P<h24>\d{1,2}):(?P<m24>\d{2})\b"
)
_NUMBER = re.compile(r"(?<!\w)[+-]?\d+(?:[ \u00a0\u202f]\d{3})*(?:[.,]\d+)*")
_URL = re.compile(r"https?://[^\s<>\"']+", re.I)
# English claims allow Czech-only letters solely inside quoted labels or currency.
_CZECH_LETTER = re.compile(r"[ěščřžýáíéůúťďňĚŠČŘŽÝÁÍÉŮÚŤĎŇ]")
_ALLOWED_CZECH = re.compile(r"\bKč\b")  # Currency symbol, not Czech prose.
_QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”|„[^“”]*[“”]")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.I)


def _number(value: str, language: AnswerLanguage) -> str:
    value = re.sub(r"\s", "", value)
    # In English, a comma before three digits denotes thousands, except 0,xxx.
    if language == "en" and re.fullmatch(r"[+-]?[1-9]\d{0,2}(?:,\d{3})+(?:\.\d+)?", value):
        value = value.replace(",", "")
    else:
        value = value.replace(",", ".")
    try:
        number = Decimal(value)
        canonical = format(number, "f")
        if "." in canonical:
            canonical = canonical.rstrip("0").rstrip(".")
        return "0" if number == 0 else canonical
    except ArithmeticError:
        return value  # Malformed numbers cannot match a valid canonical number.


def _under_thousand(values: list[int]) -> int | None:
    if 100 in values:
        index = values.index(100)
        if index > 1 or (index == 1 and not 1 <= values[0] <= 9):
            return None
        rest = values[index + 1 :]
        remainder = _under_thousand(rest) if rest else 0
        if remainder is not None and remainder < 100:
            return (values[0] if index else 1) * 100 + remainder
    elif len(values) == 1 and values[0] < 100:
        return values[0]
    elif len(values) == 2 and values[0] in range(20, 100, 10) and 1 <= values[1] <= 9:
        return sum(values)
    return None


def _word_number(values: list[int]) -> int | None:
    if 1000 not in values:
        return _under_thousand(values)
    index = values.index(1000)
    thousands = _under_thousand(values[:index]) if index else 1
    rest = values[index + 1 :]
    remainder = _under_thousand(rest) if rest else 0
    if thousands and remainder is not None:
        return thousands * 1000 + remainder
    return None


def normalized_facts(text: str, *, language: AnswerLanguage = "en") -> frozenset[str]:
    """Extract comparable literals; mask dates/links before extracting numbers.

    Types stay separate: a matching day/month/year component or plain number
    cannot satisfy a date, time, URL, email or percentage. Associations, units,
    conditions and exceptions are left to the independent verifier.
    """
    text = unicodedata.normalize("NFC", text)
    facts: set[str] = set()

    def mask(pattern: re.Pattern[str], kind: str) -> None:
        nonlocal text

        def replace(match: re.Match[str]) -> str:
            value = match.group().rstrip(".,;:!?)]}")
            if kind == "url":
                # Paths and query values are case-sensitive; only host/scheme fold.
                value = re.sub(
                    r"^https?://[^/?#]+", lambda prefix: prefix.group().lower(), value, flags=re.I
                )
            else:
                value = value.lower()
            facts.add(f"{kind}:{value}")
            return " " * len(match.group())

        text = pattern.sub(replace, text)

    mask(_URL, "url")
    mask(_EMAIL, "email")
    text = text.lower()
    for pattern in _DATES[language]:

        def replace_date(match: re.Match[str]) -> str:
            month = match["month"]
            months = _MONTHS[language]
            month_number = months[month] if month in months else int(month)
            try:
                canonical = date(int(match["year"]), month_number, int(match["day"])).isoformat()
            except ValueError:
                canonical = "invalid:" + match.group()
            facts.add("date:" + canonical)
            return " " * len(match.group())

        text = pattern.sub(replace_date, text)

    def replace_time(match: re.Match[str]) -> str:
        hour = int(match["hour"] or match["h24"])
        minute = int(match["minute"] or match["m24"] or "0")
        period = match["period"]
        if period:
            hour = hour % 12 + (12 if period.startswith("p") else 0)
        facts.add(f"time:{hour:02}:{minute:02}")
        return " " * len(match.group())

    text = _TIME.sub(replace_time, text)
    # 'one-off' describes a fee, not a quantity; an explicit 'one' remains checked.
    text = text.replace("one-off", "single")
    replacements: list[tuple[int, int, list[int]]] = []
    for match in _WORD_PATTERNS[language].finditer(text):
        value = _WORDS[language][match.group()]
        if replacements and _WORD_GAPS[language].fullmatch(
            text[replacements[-1][1] : match.start()]
        ):
            start, _, previous = replacements[-1]
            if _word_number([*previous, value]) is not None:
                replacements[-1] = (start, match.end(), [*previous, value])
                continue
        replacements.append((match.start(), match.end(), [value]))
    for start, end, values in reversed(replacements):
        text = text[:start] + str(_word_number(values)) + text[end:]
    for match in _NUMBER.finditer(text):
        value = _number(match.group(), language)
        suffix = text[match.end() :]
        percent = re.match(r"\s*(?:%|percent\b|per cent\b|procent\w*)", suffix)
        facts.add(("percent:" if percent else "number:") + value)
    return frozenset(facts)


def check_claim(
    claim: CandidateClaim,
    sentences: Mapping[str, str],
    *,
    answer_language: AnswerLanguage = "en",
    question: str = "",
    index: int = 0,
) -> CheckResult:
    """Check citation membership; literals must come from cited sentences or the question.

    Echoing a value from the question (e.g. "after 20 days") is not a sourced fact;
    whether the claim is supported is left to the verifier.
    """
    reasons: list[str] = []
    if not 1 <= len(claim.citations) <= 3 or len(set(claim.citations)) != len(claim.citations):
        reasons.append("invalid_citation_count")
    if any(citation not in sentences for citation in claim.citations):
        reasons.append("unknown_citation")
    evidence = frozenset().union(
        *(
            normalized_facts(sentences[citation], language="cs")
            for citation in claim.citations
            if citation in sentences
        ),
        normalized_facts(question, language="en"),
        normalized_facts(question, language="cs"),
    )
    if normalized_facts(claim.text, language=answer_language) - evidence:
        reasons.append("literal_not_in_citations")
    if answer_language == "en" and _CZECH_LETTER.search(
        _ALLOWED_CZECH.sub("", _QUOTED.sub("", claim.text))
    ):
        reasons.append("language_mismatch")
    return CheckResult(claim_index=index, ok=not reasons, reasons=tuple(reasons))


def check_claims(
    claims: Sequence[CandidateClaim],
    sentences: Mapping[str, str],
    *,
    answer_language: AnswerLanguage = "en",
    question: str = "",
) -> tuple[CheckResult, ...]:
    """Return one deterministic result per claim, preserving candidate order."""
    return tuple(
        check_claim(claim, sentences, answer_language=answer_language, question=question, index=i)
        for i, claim in enumerate(claims)
    )
