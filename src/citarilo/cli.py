"""Command-line entry point."""

import argparse
import json
import re
import sys
from collections.abc import Sequence
from contextlib import closing
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import uuid4

import uvicorn
from pydantic import ValidationError

from citarilo import __version__
from citarilo.api import create_app
from citarilo.budget import BudgetExceeded, BudgetLedger, actual_cost
from citarilo.config import Settings, validate_set_name
from citarilo.corpus import CorpusError, load_corpus
from citarilo.evals import (
    EvalCase,
    EvidenceMappingError,
    VerifierProbe,
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
    IndexLockedError,
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
from citarilo.records import RunRecord, iter_records, read_feedback, read_record

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUN_ID = r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}"


def _utc_day(value: str) -> datetime:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError("must be YYYY-MM-DD") from None
    return parsed.replace(tzinfo=UTC)


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
        description="Evidence-backed answers from company documents with verified citations.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command")
    importing = commands.add_parser("import", help="Convert local PDF, DOCX and MD into a corpus.")
    importing.add_argument("sources", nargs="+", type=Path, metavar="SRC")
    importing.add_argument("--out", required=True, type=Path, metavar="DIR")
    importing.add_argument("--force", action="store_true", help="Replace existing output files.")
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
    evaluate.add_argument("--eval-set")
    evaluate.add_argument("--max-usd", help="Maximum USD spend for this evaluation run.")
    evaluate.add_argument("--split", choices=("dev", "heldout"), default="dev")
    evaluate.add_argument("--dry-run", action="store_true", help="Estimate without network access.")
    evaluate.add_argument(
        "--case",
        action="append",
        metavar="ID",
        help="Evaluate only this case (repeatable). A probe pair id selects both probes.",
    )
    runs = commands.add_parser("runs", help="List local answer audits. Offline; no network.")
    runs.add_argument("--failed", action="store_true", help="Show only refused runs and why.")
    runs.add_argument(
        "--feedback", action="store_true", help="Show only runs that have an operator note."
    )
    runs.add_argument("--since", metavar="YYYY-MM-DD", type=_utc_day)
    runs.add_argument("--limit", type=int, metavar="N", help="Show at most N runs.")
    runs.add_argument(
        "--include-evals", action="store_true", help="Include runs that belong to an eval case."
    )
    runs.add_argument("--to-case", metavar="RUN_ID", help="Write a stub eval case from this run.")
    runs.add_argument("--eval-set", help="Evaluation set for --to-case.")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return
    if args.command == "import":
        from citarilo.importer import DocumentImportError, import_documents

        # Import is entirely offline and never loads settings or a local .env.
        try:
            report = import_documents(args.sources, args.out, force=args.force)
        except (DocumentImportError, OSError) as exc:
            parser.exit(1, f"citarilo: error: {exc}\n")
        print(
            f"Imported {report.documents} documents; {report.sections} sections; "
            f"{report.sentences} sentences; {len(report.warnings)} warnings."
        )
        for warning in report.warnings:
            print(f"Warning: {warning}")
        return
    max_usd = None
    if args.command == "eval" and (args.grading_template or args.summarize):
        if args.dry_run or args.max_usd is not None or args.split != "dev" or args.case:
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
    if args.command == "eval":
        root = _REPO_ROOT
        available = sorted(
            p.name
            for p in (root / "evals").iterdir()
            if p.is_dir() and (p / "cases.jsonl").is_file()
        )
        chosen = args.eval_set or settings.data_dir.name
        try:
            validate_set_name(chosen)
        except ValueError:
            chosen = ""
        required = [root / "evals" / chosen / "cases.jsonl", root / "data" / chosen]
        if args.probes:
            required.append(root / "evals" / chosen / "verifier_cases.jsonl")
        if (
            chosen not in available
            or not required[1].is_dir()
            or any(not p.is_file() for p in required[::2] if p != required[1])
        ):
            names = ", ".join(available) or "none"
            parser.exit(
                1, f"citarilo: error: Select a valid --eval-set; available sets: {names}.\n"
            )
        args.eval_set = chosen
    try:
        if args.command == "runs":
            _runs(parser, settings, args)
        elif args.command == "ingest":
            _ingest(settings, dry_run=args.dry_run, client=client)
        elif args.command == "ask":
            validate_question(args.question)
            if context is not None:
                _ask(args.question, context=context)
            else:
                with closing(create_live_context(settings)) as live_context:
                    _ask(args.question, context=live_context)
        elif args.command == "serve":
            uvicorn.run(
                create_app(settings=settings, web_dir=settings.web_dir),
                host=settings.api_host,
                port=settings.api_port,
                reload=False,
                workers=1,
                proxy_headers=False,
                access_log=False,
            )
        elif args.retrieval:
            passed = _eval(
                settings,
                dry_run=args.dry_run,
                max_usd=max_usd,
                split=args.split,
                client=client,
                eval_set=args.eval_set,
                case_ids=args.case,
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
    except IndexLockedError as exc:
        parser.exit(1, f"citarilo: error: {exc}\n")
    except IndexMismatchError:
        parser.exit(
            1, "citarilo: error: Index mismatch or corruption; run citarilo ingest explicitly.\n"
        )
    except InvalidQuestionError:
        parser.exit(1, "citarilo: error: Question must contain 3–500 characters.\n")
    except EvidenceMappingError as exc:
        parser.exit(1, f"citarilo: error: {exc}\n")
    except PipelineError as exc:
        parser.exit(1, f"citarilo: error: {exc.code}: {exc.message}\n")
    except Exception:
        # Provider errors can contain request details or credentials. Do not echo them.
        operation = {
            "ingest": "Ingest",
            "ask": "Answer",
            "serve": "Server",
            "runs": "Run listing",
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


def _print_run(record: RunRecord, *, failed: bool) -> None:
    label = f"{record.status}/{record.refusal_code}" if record.refusal_code else record.status
    question = " ".join(record.question.split())
    print(
        f"{record.created_at.isoformat()}  {record.run_id}  {label}  "
        f"{record.prompt_version}  {question}"
    )
    if not failed:
        return
    claims = record.candidate_claims
    for check in record.check_results:
        if check.ok:
            continue
        reasons = ", ".join(check.reasons) if check.reasons else "failed"
        print(f"  check {check.claim_index}: {reasons}")
    for verdict in record.verdicts:
        if verdict.verdict == "supported":
            continue
        if 0 <= verdict.claim_index < len(claims):
            claim = claims[verdict.claim_index]
            cited = ", ".join(claim.citations)
            print(f"  verdict {verdict.claim_index} {verdict.verdict}: {claim.text} [{cited}]")
        else:
            print(f"  verdict {verdict.claim_index} {verdict.verdict}:")
        if verdict.reason:
            print(f"    reason: {verdict.reason}")


def _case_stub(record: RunRecord, *, note: str | None = None) -> dict[str, object]:
    """Empty expectations plus cited IDs. EvalCase rejects this shape on purpose."""
    cited: list[str] = []
    claims = record.claims or record.candidate_claims
    for claim in claims:
        for citation in claim.citations:
            if citation not in cited:
                cited.append(citation)
    stub: dict[str, object] = {
        "id": f"manual-{record.run_id[:8].lower()}",
        "split": "dev",
        "question": record.question,
        "language": record.answer_language or "cs",
        "expected_status": "answered",
        "expected_refusal": None,
        "required_facts": [],
        "forbidden_claims": [],
        "supporting_sentence_ids": [],
        "expected_section_keys": [],
        "tags": ["manual"],
        "hint_cited_ids": cited,
    }
    if note:
        stub["note"] = note
    return stub


def _jsonl_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    found: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            found.add(item["id"])
    return found


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    if path.is_file():
        existing = path.read_bytes()
        if existing and not existing.endswith(b"\n"):
            line = "\n" + line
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line)


def _export_case(
    parser: argparse.ArgumentParser, settings: Settings, run_id: str, eval_set: str
) -> None:
    if re.fullmatch(_RUN_ID, run_id) is None:
        parser.exit(1, "citarilo: error: Unknown run.\n")
    path = settings.runs_dir / f"{run_id}.json"
    if not path.is_file():
        parser.exit(1, "citarilo: error: Unknown run.\n")
    try:
        record = read_record(path)
    except (OSError, ValidationError, ValueError):
        parser.exit(1, "citarilo: error: Run record is unreadable or invalid.\n")
    try:
        validate_set_name(eval_set)
    except ValueError:
        eval_set = ""
    root = _REPO_ROOT
    evals_dir = root / "evals"
    available = (
        sorted(
            entry.name
            for entry in evals_dir.iterdir()
            if entry.is_dir() and (entry / "cases.jsonl").is_file()
        )
        if evals_dir.is_dir()
        else []
    )
    if eval_set not in available:
        names = ", ".join(available) or "none"
        parser.exit(1, f"citarilo: error: Select a valid --eval-set; available sets: {names}.\n")
    notes = read_feedback(settings.feedback_dir, run_id)
    parts: list[str] = []
    for entry in notes:
        text = entry.note.strip()
        if entry.label and text:
            parts.append(f"[{entry.label}] {text}")
        elif entry.label:
            parts.append(f"[{entry.label}]")
        elif text:
            parts.append(text)
    stub = _case_stub(record, note="\n".join(parts) or None)
    case_id = str(stub["id"])
    set_dir = evals_dir / eval_set
    cases_path = set_dir / "cases.jsonl"
    # The loader accepts JSONL only. Empty facts, the manual tag and hint_cited_ids
    # are not a valid answered EvalCase, so the stub must not join cases.jsonl.
    try:
        EvalCase.model_validate_json(json.dumps(stub))
        destination = cases_path
    except ValidationError:
        destination = set_dir / "manual_cases.jsonl"
    if case_id in _jsonl_ids(cases_path) or case_id in _jsonl_ids(destination):
        parser.exit(1, f"citarilo: error: Case id {case_id} already exists.\n")
    _append_jsonl(destination, stub)
    print(f"Wrote {destination}")
    if destination.name == "manual_cases.jsonl":
        print("Eval does not load this file.")
    print(
        "A human must fill required_facts, supporting_sentence_ids and "
        "expected_section_keys before this stub can be graded."
    )


def _feedback_lines(settings: Settings, record: RunRecord) -> list[str]:
    lines: list[str] = []
    for entry in read_feedback(settings.feedback_dir, record.run_id):
        text = entry.note.strip()
        label = entry.label or "-"
        lines.append(f"  {label}  {text}".rstrip())
    return lines


def _runs(parser: argparse.ArgumentParser, settings: Settings, args: argparse.Namespace) -> None:
    listing = (
        args.failed
        or args.feedback
        or args.since is not None
        or args.limit is not None
        or args.include_evals
    )
    if args.to_case or args.eval_set:
        if not args.to_case or not args.eval_set:
            parser.exit(1, "citarilo: error: --to-case and --eval-set are required together.\n")
        if listing:
            parser.exit(1, "citarilo: error: --to-case does not accept listing options.\n")
        _export_case(parser, settings, args.to_case, args.eval_set)
        return
    if args.limit is not None and args.limit < 1:
        parser.exit(1, "citarilo: error: --limit must be a positive integer.\n")

    def warn(path: Path) -> None:
        sys.stderr.write(
            f"citarilo: warning: Skipping unreadable or invalid run file {path.name}.\n"
        )

    selected = []
    for record in iter_records(settings.runs_dir, on_invalid=warn):
        if record.eval_case_id is not None and not args.include_evals:
            continue
        if args.since is not None and record.created_at < args.since:
            continue
        if args.failed and record.status != "refused":
            continue
        if args.feedback and not read_feedback(settings.feedback_dir, record.run_id):
            continue
        selected.append(record)
    selected.sort(key=lambda record: (record.created_at, record.run_id), reverse=True)
    if args.limit is not None:
        selected = selected[: args.limit]
    for record in selected:
        _print_run(record, failed=args.failed)
        if args.feedback:
            for line in _feedback_lines(settings, record):
                print(line)


def _unknown_ids(ids: list[str]) -> None:
    label = "id" if len(ids) == 1 else "ids"
    sys.stderr.write(f"citarilo: error: Unknown evaluation {label}: {', '.join(ids)}.\n")
    raise SystemExit(2)


def _filter_cases(
    cases: list[EvalCase], requested: list[str] | None
) -> tuple[list[EvalCase], tuple[str, ...] | None]:
    """Apply --case after the split filter. None means the whole split."""
    if not requested:
        return cases, None
    known = {case.id for case in cases}
    missing = [item for item in dict.fromkeys(requested) if item not in known]
    if missing:
        _unknown_ids(missing)
    chosen = set(requested)
    selected = [case for case in cases if case.id in chosen]
    return selected, tuple(case.id for case in selected)


def _filter_probes(
    probes: list[VerifierProbe], requested: list[str] | None
) -> tuple[list[VerifierProbe], tuple[str, ...] | None]:
    """Select by probe id or pair id. A pair id selects both probes, in file order."""
    if not requested:
        return probes, None
    ids = {probe.id for probe in probes}
    pairs = {probe.pair_id for probe in probes}
    missing = [item for item in dict.fromkeys(requested) if item not in ids and item not in pairs]
    if missing:
        _unknown_ids(missing)
    chosen = set(requested)
    selected = [probe for probe in probes if probe.id in chosen or probe.pair_id in chosen]
    return selected, tuple(probe.id for probe in selected)


def _eval(
    settings: Settings,
    *,
    dry_run: bool,
    max_usd: Decimal | None,
    split: str,
    client: EmbeddingsClient | None,
    eval_set: str,
    case_ids: list[str] | None = None,
) -> bool:
    root = _REPO_ROOT
    reference_dir = root / "data" / eval_set
    if split == "heldout":
        print("Warning: heldout is reserved for the final one-off run; do not use it for tuning.")
    cases = [
        case
        for case in load_cases(root / "evals" / eval_set / "cases.jsonl")
        if case.split == split
    ]
    cases, selected = _filter_cases(cases, case_ids)
    if dry_run:
        print(f"Dry run: {len(cases)} queries; split {split}; top_k {settings.top_k}.")
        cost = estimate_retrieval(cases, settings, reference_dir=reference_dir)
        print(f"Worst-case cost: ${cost:.6f}.")
        return True
    if max_usd is None:
        raise ValueError("A run budget is required")
    report, path = evaluate_retrieval(
        cases,
        settings,
        max_usd,
        split=split,
        client=client,
        reference_dir=reference_dir,
        eval_set=eval_set,
        case_ids=selected,
    )
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
    root = _REPO_ROOT
    reference_dir = root / "data" / args.eval_set
    if args.live:
        if args.split == "heldout":
            print(
                "Warning: heldout is reserved for the final one-off run; do not use it for tuning."
            )
        cases = [
            c
            for c in load_cases(root / "evals" / args.eval_set / "cases.jsonl")
            if c.split == args.split
        ]
        cases, selected = _filter_cases(cases, args.case)
        if args.dry_run:
            print(
                f"Dry run: {len(cases)} answer cases; split {args.split}; top_k {settings.top_k}."
            )
            cost = estimate_answers(cases, settings, reference_dir=reference_dir)
            print(f"Worst-case cost: ${cost:.6f}.")
            return 0
        report, path = evaluate_answers(
            cases,
            settings,
            max_usd,
            split=args.split,
            context=context,
            reference_dir=reference_dir,
            eval_set=args.eval_set,
            case_ids=selected,
        )
        print(f"{'Case':<36} {'Status':<10} Error")
        for case in report.cases:
            print(
                f"{case.id:<36} {case.status or 'not_run':<10} "
                f"{case.error.code if case.error else '-'}"
            )
        print("Semantic grading required: create a template with --grading-template REPORT.")
        failed = not report.complete or any(c.error for c in report.cases)
    else:
        probes = load_probes(root / "evals" / args.eval_set / "verifier_cases.jsonl")
        probes, selected = _filter_probes(probes, args.case)
        if args.dry_run:
            print(f"Dry run: {len(probes)} verifier probes; verification only.")
            cost = estimate_probes(probes, settings, reference_dir=reference_dir)
            print(f"Worst-case cost: ${cost:.6f}.")
            return 0
        report, path = evaluate_probes(
            probes,
            settings,
            max_usd,
            reference_dir=reference_dir,
            eval_set=args.eval_set,
            case_ids=selected,
        )
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
