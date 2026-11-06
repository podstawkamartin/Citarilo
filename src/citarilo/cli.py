"""Command-line entry point."""

import argparse
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import uuid4

import uvicorn
from pydantic import ValidationError

from citarilo import __version__
from citarilo.api import create_app
from citarilo.budget import BudgetExceeded, BudgetLedger, actual_cost
from citarilo.config import Settings
from citarilo.corpus import CorpusError, load_corpus
from citarilo.evals import (
    estimate_answers,
    estimate_probes,
    estimate_retrieval,
    evaluate_answers,
    evaluate_probes,
    evaluate_retrieval,
    load_cases,
    load_probes,
)
from citarilo.grading import summarize, write_grading_template
from citarilo.index import (
    EmbeddingsClient,
    IndexMismatchError,
    IndexMissingError,
    MissingAPIKeyError,
    build_index,
    estimate_ingest,
)
from citarilo.pipeline import (
    AnswerContext,
    InvalidQuestionError,
    PipelineError,
    answer,
    create_live_context,
    validate_question,
)


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    client: EmbeddingsClient | None = None,
    context: AnswerContext | None = None,
) -> None:
    """Run a subcommand; settings, embedding client and answer context are injectable."""
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
    ask = commands.add_parser("ask", help="Answer a question with verified sentence citations.")
    ask.add_argument("question", help="A question in English or Czech (3–500 characters).")
    commands.add_parser("serve", help="Serve the local HTTP API.")
    evaluate = commands.add_parser("eval", help="Evaluate retrieval, answers or verifier probes.")
    modes = evaluate.add_mutually_exclusive_group(required=True)
    modes.add_argument("--retrieval", action="store_true")
    modes.add_argument("--live", action="store_true")
    modes.add_argument("--probes", action="store_true")
    modes.add_argument("--grading-template", type=Path, metavar="REPORT")
    modes.add_argument("--summarize", nargs=2, type=Path, metavar=("REPORT", "GRADES"))
    evaluate.add_argument("--max-usd", help="Maximum USD spend for this evaluation run.")
    evaluate.add_argument("--split", choices=("dev", "heldout"), default="dev")
    evaluate.add_argument("--dry-run", action="store_true", help="Estimate without network access.")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return
    max_usd = None
    if args.command == "eval" and (args.grading_template or args.summarize):
        if args.dry_run or args.max_usd is not None or args.split != "dev":
            parser.exit(1, "citarilo: error: Grading commands do not accept run options.\n")
        try:
            if args.grading_template:
                print(f"Grading template: {write_grading_template(args.grading_template)}")
            else:
                summary, path = summarize(*args.summarize)
                print(f"{'Metric / gate':<38} Result")
                for name in (
                    "answered_correct_rate",
                    "answered_unanswerables",
                    "successful_injections",
                ):
                    print(f"{name:<38} {summary[name]}")
                for name, value in summary["verifier_benefit"].items():
                    print(f"{'verifier_' + name:<38} {value}")
                for name, value in summary["gates"].items():
                    print(f"{name:<38} {value}")
                print(f"Overall: {summary['gate']}\nSummary: {path}")
                if summary["gate"] != "PASS":
                    parser.exit(2)
        except FileExistsError:
            parser.exit(1, "citarilo: error: Output already exists; it was not overwritten.\n")
        except (ValueError, OSError):
            parser.exit(
                1,
                "citarilo: error: Invalid/incomplete grades or report; "
                "check IDs, evidence, judgments and metadata.\n",
            )
        return
    if args.command == "eval":
        if args.max_usd is None and not args.dry_run:
            parser.exit(1, "citarilo: error: --max-usd is required for an evaluation.\n")
        if args.max_usd is not None:
            try:
                max_usd = Decimal(args.max_usd)
                if not max_usd.is_finite() or max_usd <= 0:
                    raise ValueError
            except (InvalidOperation, ValueError):
                parser.exit(1, "citarilo: error: --max-usd must be a finite positive amount.\n")
    try:
        # A dry run never needs credentials, including the contents of a local .env.
        if settings is None and context is not None and args.command in ("ask", "eval"):
            settings = context.settings
        if settings is None:
            settings = (
                Settings(_env_file=None, openai_api_key=None)
                if getattr(args, "dry_run", False)
                else Settings()
            )
    except ValidationError:
        parser.exit(1, "citarilo: error: Invalid settings; check the environment and .env file.\n")
    try:
        if args.command == "ingest":
            _ingest(settings, dry_run=args.dry_run, client=client)
        elif args.command == "ask":
            validate_question(args.question)
            _ask(args.question, context=context or create_live_context(settings))
        elif args.command == "serve":
            uvicorn.run(
                create_app(settings=settings),
                host=settings.api_host,
                port=settings.api_port,
                reload=False,
                workers=1,
                proxy_headers=False,
                access_log=False,
            )
        elif args.retrieval:
            passed = _eval(
                settings, dry_run=args.dry_run, max_usd=max_usd, split=args.split, client=client
            )
            if not passed:
                parser.exit(2)
        else:
            exit_code = _eval_answers_or_probes(settings, args, max_usd, context=context)
            if exit_code:
                parser.exit(exit_code)
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
    except InvalidQuestionError:
        parser.exit(1, "citarilo: error: Question must contain 3–500 characters.\n")
    except PipelineError as exc:
        parser.exit(1, f"citarilo: error: {exc.code}: {exc.message}\n")
    except Exception:
        # Provider errors can contain request details or credentials. Do not echo them.
        operation = {
            "ingest": "Ingest",
            "ask": "Answer",
            "serve": "Server",
            "eval": "Retrieval evaluation"
            if args.command == "eval" and args.retrieval
            else "Evaluation",
        }[args.command]
        parser.exit(
            1, f"citarilo: error: {operation} failed; check configuration and the spend ledger.\n"
        )


def _ask(question: str, *, context: AnswerContext) -> None:
    record = answer(question, context=context, origin="api")
    if record.status == "refused":
        print(f"Refused: {record.refusal_code}.")
    else:
        for number, claim in enumerate(record.claims, start=1):
            citations = ", ".join(claim.citations)
            print(f"{number}. {claim.text} [{citations}]")
        print("\nCited sentences (Czech):")
        for source in record.sources:
            print(f"[{source.sentence_id}] {source.text}")
    print(
        f"Cost ${record.metrics.cost_usd:.6f}; "
        f"remaining budget ${context.ledger.remaining_usd():.6f}."
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


def _eval_answers_or_probes(
    settings: Settings,
    args: argparse.Namespace,
    max_usd: Decimal | None,
    *,
    context: AnswerContext | None,
) -> int:
    if args.live:
        if args.split == "heldout":
            print(
                "Warning: heldout is reserved for the final one-off run; do not use it for tuning."
            )
        cases = [c for c in load_cases(Path("evals/cases.jsonl")) if c.split == args.split]
        if args.dry_run:
            print(
                f"Dry run: {len(cases)} answer cases; split {args.split}; top_k {settings.top_k}."
            )
            print(f"Worst-case cost: ${estimate_answers(cases, settings):.6f}.")
            return 0
        report, path = evaluate_answers(cases, settings, max_usd, split=args.split, context=context)
        print(f"{'Case':<36} {'Status':<10} Error")
        for case in report.cases:
            print(
                f"{case.id:<36} {case.status or 'not_run':<10} "
                f"{case.error.code if case.error else '-'}"
            )
        print("Semantic grading required: create a template with --grading-template REPORT.")
        failed = not report.complete or any(c.error for c in report.cases)
    else:
        probes = load_probes(Path("evals/verifier_cases.jsonl"))
        if args.dry_run:
            print(f"Dry run: {len(probes)} verifier probes; verification only.")
            print(f"Worst-case cost: ${estimate_probes(probes, settings):.6f}.")
            return 0
        report, path = evaluate_probes(probes, settings, max_usd)
        print(f"{'Probe':<48} {'Expected':<14} Actual")
        for case in report.cases:
            actual = (
                case.verdict.verdict
                if case.verdict
                else case.error.code
                if case.error
                else "not_run"
            )
            print(f"{case.probe.id:<48} {case.probe.expected_verdict:<14} {actual}")
        print(
            f"False accepts: {report.metrics['false_accepts']}; "
            f"false rejects: {report.metrics['false_rejects']}."
        )
        print(f"Gate zero false accepts: {report.gate}.")
        failed = report.gate != "PASS"
    print(f"Complete: {report.complete}; cost ${report.cost_usd:.6f}.")
    print(f"Report: {path}")
    if not report.complete or any(c.error for c in report.cases):
        return 1
    return 2 if failed else 0
