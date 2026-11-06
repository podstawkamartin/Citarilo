"""Command-line entry point."""

import argparse
from collections.abc import Sequence
from uuid import uuid4

from pydantic import ValidationError

from citarilo import __version__
from citarilo.budget import BudgetExceeded, BudgetLedger, actual_cost
from citarilo.config import Settings
from citarilo.corpus import CorpusError, load_corpus
from citarilo.index import (
    EmbeddingsClient,
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
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return
    try:
        settings = settings if settings is not None else Settings()
    except ValidationError:
        parser.exit(1, "citarilo: error: Invalid settings; check the environment and .env file.\n")
    try:
        _ingest(settings, dry_run=args.dry_run, client=client)
    except MissingAPIKeyError:
        parser.exit(1, "citarilo: error: Set OPENAI_API_KEY in the environment or .env file.\n")
    except BudgetExceeded:
        parser.exit(1, "citarilo: error: Budget exceeded; no further requests were sent.\n")
    except CorpusError:
        parser.exit(1, "citarilo: error: Invalid corpus; check the configured data directory.\n")
    except Exception:
        # Provider errors can contain request details or credentials. Do not echo them.
        parser.exit(
            1, "citarilo: error: Ingest failed; check configuration and the spend ledger.\n"
        )


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
