"""Offline production-path benchmark for the embedded Qdrant section index."""

from __future__ import annotations

import argparse
import json
import os
import platform
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace
from typing import cast

# Fix BLAS parallelism before importing numpy/Qdrant for comparable local runs.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np  # noqa: E402

from citarilo.budget import BudgetLedger  # noqa: E402
from citarilo.config import Settings  # noqa: E402
from citarilo.corpus import load_corpus, section_documents  # noqa: E402
from citarilo.index import UsageTrackingEmbeddings, build_index, load_index  # noqa: E402
from citarilo.records import get_git_info  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = 3072


class _OfflineReservations:
    """Inert reservation callbacks: the benchmark never opens a spend ledger."""

    @contextmanager
    def reserve(self, *args, **kwargs):
        yield SimpleNamespace(
            id="offline-benchmark",
            mark_sent=lambda: None,
            settle=lambda *args: None,
            cancel=lambda: None,
        )


class _VectorClient:
    """SDK-shaped embeddings endpoint returning precomputed synthetic vectors."""

    def __init__(self, texts, vectors, query_vectors):
        self.embeddings = self
        self.positions = {text: i for i, text in enumerate(texts)}
        self.vectors = vectors
        self.query_vectors = query_vectors

    def create(self, *, input, **kwargs):
        def vector(text):
            if text in self.positions:
                return self.vectors[self.positions[text]].tolist()
            return self.query_vectors[int(text.removeprefix("query:"))].tolist()

        return SimpleNamespace(
            data=[SimpleNamespace(index=i, embedding=vector(text)) for i, text in enumerate(input)],
            usage=SimpleNamespace(prompt_tokens=0, total_tokens=0),
        )


def _unit_vectors(rng, count):
    vectors = rng.standard_normal((count, DIMENSIONS), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def _corpus(path, size):
    path.mkdir()
    for start in range(0, size, 100):
        lines = [f"# Synthetic policy {start // 100}", ""]
        for i in range(start, min(start + 100, size)):
            lines += [
                f"## Return policy {i} {{#rule-{i}}}",
                f"Goods in category {i} may be returned within 30 days.",
                "Keep the original packaging and include the order number.",
                "Contact customer support before sending a parcel.",
                "",
            ]
        (path / f"policy-{start // 100}.md").write_text("\n".join(lines), encoding="utf-8")
    return load_corpus(path)


def _measure_size(size, repeats, queries, seed):
    rng = np.random.default_rng(seed + size)
    vectors = _unit_vectors(rng, size)
    query_vectors = _unit_vectors(rng, queries)
    query_vectors[0] = vectors[0]  # Include an exact-match query.
    reference = query_vectors.astype(np.float64) @ vectors.astype(np.float64).T
    reference /= np.linalg.norm(query_vectors.astype(np.float64), axis=1, keepdims=True)
    reference /= np.linalg.norm(vectors.astype(np.float64), axis=1)[None, :]
    top_k = min(5, size)
    expected = np.argsort(-reference, axis=1)[:, :top_k]
    timings = {"build_publish_ms": [], "load_validation_ms": [], "disk_bytes": []}
    latencies = []
    overlaps = []
    differences = []
    with TemporaryDirectory(prefix="citarilo-vector-bench-") as directory:
        root = Path(directory)
        corpus = _corpus(root / "corpus", size)
        documents = section_documents(corpus)
        keys = [doc.metadata["section_key"] for doc in documents]
        positions = {key: i for i, key in enumerate(keys)}
        client = _VectorClient([doc.page_content for doc in documents], vectors, query_vectors)
        ledger = cast(BudgetLedger, _OfflineReservations())
        for repeat in range(repeats + 1):
            settings = Settings(
                _env_file=None,
                openai_api_key=None,
                data_dir=root / "corpus",
                runtime_dir=root / f"run-{repeat}",
                embedding_dimensions=DIMENSIONS,
            )
            started = perf_counter()
            manifest = build_index(corpus, settings, ledger, client=client)
            build_ms = (perf_counter() - started) * 1000
            embeddings = UsageTrackingEmbeddings(settings, ledger, client=client)
            started = perf_counter()
            with load_index(corpus, settings, embeddings) as store:
                load_ms = (perf_counter() - started) * 1000
                # Each repetition warms the search path before recording latency.
                store.search("query:0", top_k)
                for i in range(queries if repeat else 1):
                    started = perf_counter()
                    hits = store.search(f"query:{i}", top_k)
                    latency = (perf_counter() - started) * 1000
                    if repeat:
                        latencies.append(latency)
                        actual = [positions[key] for key, _, _ in hits]
                        overlaps.append(len(set(actual) & set(expected[i])) / top_k)
                        differences.extend(
                            abs(score - reference[i, positions[key]]) for key, score, _ in hits
                        )
            if repeat:
                timings["build_publish_ms"].append(build_ms)
                timings["load_validation_ms"].append(load_ms)
                timings["disk_bytes"].append(
                    sum(p.stat().st_size for p in settings.qdrant_path.rglob("*") if p.is_file())
                )
    result = {
        "points": manifest.section_count,
        "dimensions": DIMENSIONS,
        "repeats": repeats,
        "queries_per_repeat": queries,
        **timings,
        "build_publish_p50_ms": float(np.median(timings["build_publish_ms"])),
        "load_validation_p50_ms": float(np.median(timings["load_validation_ms"])),
        "query_p50_ms": float(np.percentile(latencies, 50)),
        "query_p95_ms": float(np.percentile(latencies, 95)),
        "top_5_overlap": float(np.mean(overlaps)),
        "max_abs_score_difference": float(max(differences)),
    }
    if min(overlaps) != 1 or result["max_abs_score_difference"] > 1e-5:
        raise RuntimeError(f"Exact cosine parity failed: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    count = sum(1 for _ in load_corpus(ROOT / "data/trailsupply").sections())
    parser.add_argument("--sizes", type=int, nargs="+", default=[count, 1000, 5000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.sizes) < 1 or args.repeats < 1 or args.queries < 200:
        parser.error("Sizes and repeats must be positive; queries must be at least 200")
    started_at = datetime.now(UTC).isoformat()
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    results = []
    for size in args.sizes:
        result = _measure_size(size, args.repeats, args.queries, args.seed)
        results.append(result)
        print(
            f"{size} points: build {result['build_publish_p50_ms']:.1f} ms, "
            f"load {result['load_validation_p50_ms']:.1f} ms, "
            f"query p50/p95 {result['query_p50_ms']:.3f}/{result['query_p95_ms']:.3f} ms, "
            f"disk {int(np.median(result['disk_bytes']))} bytes, "
            f"overlap {result['top_5_overlap']:.3f}, "
            f"max score difference {result['max_abs_score_difference']:.3g}",
            flush=True,
        )
    # Git is inspected from the repository even when invoked from another directory.
    git = get_git_info(ROOT)
    report = {
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "qdrant_client": version("qdrant-client"),
        "numpy": np.__version__,
        "cpu_model": cpu,
        "git": git.model_dump(mode="json"),
        "seed": args.seed,
        "blas_threads": 1,
        "warmup_repeats": 1,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
