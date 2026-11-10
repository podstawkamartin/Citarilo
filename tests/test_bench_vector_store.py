"""Keep the offline vector benchmark executable and its report verifiable."""

import json
import subprocess
import sys
from pathlib import Path


def test_tiny_vector_store_benchmark(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "bench.json"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/bench_vector_store.py"),
            "--sizes",
            "8",
            "--repeats",
            "1",
            "--out",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    report = json.loads(output.read_text())
    assert "8 points" in result.stdout
    assert report["qdrant_client"] and report["python"] and report["cpu_model"]
    assert report["git"]["commit"]
    assert report["started_at"] <= report["finished_at"]
    row = report["results"][0]
    assert row["points"] == 8 and row["dimensions"] == 3072
    assert row["queries_per_repeat"] >= 200
    assert row["top_5_overlap"] == 1
    assert row["max_abs_score_difference"] < 1e-5
    assert row["disk_bytes"][0] > 0
    assert row["build_publish_p50_ms"] > 0 and row["load_validation_p50_ms"] > 0
    assert row["query_p95_ms"] >= row["query_p50_ms"] > 0
    assert not (tmp_path / ".runtime").exists()
