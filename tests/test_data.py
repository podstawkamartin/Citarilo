"""The shipped TrailSupply corpus must always satisfy the corpus format."""

from pathlib import Path

from citarilo.config import Settings
from citarilo.corpus import load_corpus

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_corpus_loads() -> None:
    corpus = load_corpus(ROOT / Settings().data_dir)

    assert len(corpus.documents) == 6
    for document in corpus.documents:
        assert document.preamble.startswith("> Fiktivní ukázková data")
