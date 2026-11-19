"""The bundled corpora satisfy the corpus format."""

from pathlib import Path

import pytest

from citarilo.corpus import load_corpus

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["trailsupply", "regulations"])
def test_shipped_corpus_loads(name: str) -> None:
    corpus = load_corpus(ROOT / "data" / name)
    assert corpus.documents
    if name == "trailsupply":
        assert len(corpus.documents) == 6
        for document in corpus.documents:
            assert document.preamble.strip().startswith("> Fiktivní testovací dokument")


def test_set_names_and_storage_are_isolated(tmp_path: Path) -> None:
    from citarilo.config import Settings

    first = Settings(_env_file=None, data_dir=Path("data/trailsupply"), runtime_dir=tmp_path)
    second = Settings(_env_file=None, data_dir=Path("data/regulations"), runtime_dir=tmp_path)
    assert first.qdrant_path != second.qdrant_path
    assert first.index_manifest_path != second.index_manifest_path
    assert first.index_lock_path == second.index_lock_path == tmp_path / "index.lock"
    for name in (".", "Bad_Name", "two--parts"):
        settings = Settings(_env_file=None, data_dir=Path(name))
        assert settings.index_dir.name
    left = Settings(_env_file=None, data_dir=tmp_path / "one" / "Customer_2026")
    right = Settings(_env_file=None, data_dir=tmp_path / "two" / "Customer_2026")
    assert left.index_dir.name.startswith("customer-2026-")
    assert left.index_dir != right.index_dir
