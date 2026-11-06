"""Command-line entry point."""

import argparse
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from citarilo import __version__
from citarilo.budget import BudgetExceeded, BudgetLedger, actual_cost
from citarilo.config import Settings
from citarilo.corpus import CorpusError, load_corpus
from citarilo.evals import estimate_retrieval, evaluate_retrieval, load_cases
from citarilo.index import (
    EmbeddingsClient,
    IndexMismatchError,
    IndexMissingError,
    MissingAPIKeyError,
    build_index,
    estimate_ingest,
)


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    client: EmbeddingsClient | None = None,
) -> None:
    """Run a subcommand; settings and the embedding client are injectable for tests."""
    parser = argparse.ArgumentParser(
        prog="citarilo",
        description="Evidence-backed customer support for TrailSupply.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command")
    ingest = commands.add_parser("ingest", help="Build and persist the section vector index.")
    ingest.add_argument(
        "--dry-run", action="store_true", help="Estimate tokens and cost without network access."
    )
    evaluate = commands.add_parser("eval", help="Evaluate section retrieval with a run budget.")
    evaluate.add_argument("--retrieval", action="store_true", required=True)
    evaluate.add_argument("--max-usd", help="Maximum USD spend for this evaluation run.")
    evaluate.add_argument("--split", choices=("dev", "heldout"), default="dev")
    evaluate.add_argument("--dry-run", action="store_true", help="Estimate without network access.")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return
    max_usd = None
    if args.command == "eval":
        if args.max_usd is None and not args.dry_run:
            parser.exit(1, "citarilo: error: --max-usd is required for a retrieval evaluation.\n")
        if args.max_usd is not None:
            try:
                max_usd = Decimal(args.max_usd)
                if not max_usd.is_finite() or max_usd <= 0:
                    raise ValueError
            except (InvalidOperation, ValueError):
                parser.exit(1, "citarilo: error: --max-usd must be a finite positive amount.\n")
    try:
        # A dry run never needs credentials, including the contents of a local .env.
        settings = (
            settings
            if settings is not None
            else (Settings(_env_file=None, openai_api_key=None) if args.dry_run else Settings())
        )
    except ValidationError:
        parser.exit(1, "citarilo: error: Invalid settings; check the environment and .env file.\n")
    try:
        if args.command == "ingest":
            _ingest(settings, dry_run=args.dry_run, client=client)
        else:
            passed = _eval(
                settings, dry_run=args.dry_run, max_usd=max_usd, split=args.split, client=client
            )
            if not passed:
                parser.exit(2)
    except MissingAPIKeyError:
        parser.exit(1, "citarilo: error: Set OPENAI_API_KEY in the environment or .env file.\n")
    except BudgetExceeded:
        parser.exit(1, "citarilo: error: Budget exceeded; no further requests were sent.\n")
    except CorpusError:
        parser.exit(1, "citarilo: error: Invalid corpus; check the configured data directory.\n")
    except IndexMissingError:
        parser.exit(1, "citarilo: error: Index missing; run citarilo ingest explicitly.\n")
    except IndexMismatchError:
        parser.exit(
            1, "citarilo: error: Index mismatch or corruption; run citarilo ingest explicitly.\n"
        )
    except Exception:
        # Provider errors can contain request details or credentials. Do not echo them.
        operation = "Ingest" if args.command == "ingest" else "Retrieval evaluation"
        parser.exit(
            1, f"citarilo: error: {operation} failed; check configuration and the spend ledger.\n"
        )


def _eval(
    settings: Settings,
    *,
    dry_run: bool,
    max_usd: Decimal | None,
    split: str,
    client: EmbeddingsClient | None,
) -> bool:
    if split == "heldout":
        print("Warning: heldout is reserved for the final one-off run; do not use it for tuning.")
    cases = [case for case in load_cases(Path("evals/cases.jsonl")) if case.split == split]
    if dry_run:
        print(f"Dry run: {len(cases)} queries; split {split}; top_k {settings.top_k}.")
        print(f"Worst-case cost: ${estimate_retrieval(cases, settings):.6f}.")
        return True
    if max_usd is None:
        raise ValueError("A run budget is required")
    report, path = evaluate_retrieval(cases, settings, max_usd, split=split, client=client)
    print(f"{'Case':<36} {'Hit':<5} Expected section ranks (1-based; - = missing)")
    for result in report.cases:
        hit = "n/a" if result.hit is None else "yes" if result.hit else "no"
        ranks = ", ".join(
            f"{key}:{rank if rank is not None else '-'}" for key, rank in result.ranks.items()
        )
        if not ranks:
            ranks = "top hits: " + ", ".join(hit.key for hit in result.retrieved)
        print(f"{result.id:<36} {hit:<5} {ranks}")
    metrics = report.metrics
    recall = "n/a" if metrics.recall_at_k is None else f"{metrics.recall_at_k:.3f}"
    section_recall = (
        "n/a" if metrics.section_recall_at_k is None else f"{metrics.section_recall_at_k:.3f}"
    )
    gate_recall = "n/a" if metrics.recall_at_5 is None else f"{metrics.recall_at_5:.3f}"
    print(
        f"Recall@{settings.top_k}: {recall} ({metrics.hit_cases}/{metrics.answerable_cases}); "
        f"section recall: {section_recall}."
    )
    print(f"Gate recall@5 >= 0.900: {report.gate} ({gate_recall}).")
    print(
        f"Cost ${report.usage['cost_usd']:.6f}; "
        f"remaining run budget ${report.usage['remaining_usd']:.6f}; "
        f"remaining global budget ${report.usage['global_remaining_usd']:.6f}."
    )
    print(f"Report: {path}")
    return report.gate == "PASS"


def _ingest(settings: Settings, *, dry_run: bool, client: EmbeddingsClient | None) -> None:
    if not dry_run and not settings.openai_api_key:
        raise MissingAPIKeyError
    corpus = load_corpus(settings.data_dir)
    if dry_run:
        estimate = estimate_ingest(corpus, settings)
        expected_tokens = (estimate.upper_bound_tokens + 2) // 3
        expected_cost = actual_cost(
            settings.embedding_model, expected_tokens, 0, price_table=settings.price_table
        )
        print(
            f"Dry run: {estimate.sections} sections; "
            f"{estimate.upper_bound_tokens} upper-bound tokens; {estimate.requests} request(s)."
        )
        print(f"Worst-case cost: ${estimate.worst_case_usd:.6f}.")
        print(
            f"Estimated cost (~3 UTF-8 bytes/token, not measured): ${expected_cost:.8f} "
            f"(~{expected_tokens} tokens)."
        )
        return
    ledger = BudgetLedger(
        settings.spend_db_path,
        settings.budget_cap_usd,
        settings.price_table_version,
        price_table=settings.price_table,
    )
    group_id = f"ingest-{uuid4().hex}"
    manifest = build_index(corpus, settings, ledger, group_id=group_id, client=client)
    prompt_tokens = sum(usage.prompt_tokens for usage in manifest.usage)
    total_tokens = sum(usage.total_tokens for usage in manifest.usage)
    print(
        f"Indexed {manifest.section_count} sections; "
        f"{prompt_tokens} prompt tokens, {total_tokens} total tokens; "
        f"cost ${ledger.group_spent_usd(group_id):.6f}; "
        f"remaining budget ${ledger.remaining_usd():.6f}."
    )
