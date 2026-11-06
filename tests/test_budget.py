"""Exact accounting and durable, atomic spending limits without paid calls."""

import multiprocessing
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from citarilo.budget import (
    SCHEMA_VERSION,
    BudgetExceeded,
    BudgetLedger,
    ReservationStateError,
    actual_cost,
    max_cost,
)
from citarilo.config import TokenPrice

MODEL = "test-model"
VERSION = "test-prices-v1"
PRICES = {
    MODEL: TokenPrice(
        input_usd_per_million=Decimal("0.10"),
        output_usd_per_million=Decimal("0.50"),
    ),
    "embedding": TokenPrice(input_usd_per_million=Decimal("0.13")),
}


def _ledger(path: Path, cap: str = "3", version: str = VERSION) -> BudgetLedger:
    return BudgetLedger(path, Decimal(cap), version, price_table=PRICES)


@pytest.fixture
def ledger(tmp_path: Path) -> BudgetLedger:
    return _ledger(tmp_path / "spend.sqlite")


def _row(ledger: BudgetLedger, reservation_id: str) -> sqlite3.Row:
    with sqlite3.connect(ledger.db_path) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(
            "SELECT * FROM reservations WHERE id = ?", (reservation_id,)
        ).fetchone()


@pytest.mark.parametrize(
    ("model", "inputs", "outputs", "actual", "maximum"),
    [
        (MODEL, 0, 0, "0", "0"),
        (MODEL, 1_000_000, 1_000_000, "0.60", "0.60"),
        (MODEL, 1, 0, "0.0000001", "0.000001"),
        (MODEL, 1, 1, "0.0000006", "0.000001"),
        (MODEL, 10, 0, "0.000001", "0.000001"),
        (MODEL, 11, 0, "0.0000011", "0.000002"),
        ("embedding", 1_000_000, 0, "0.13", "0.13"),
    ],
)
def test_cost_math_and_rounding(
    model: str, inputs: int, outputs: int, actual: str, maximum: str
) -> None:
    assert actual_cost(model, inputs, outputs, price_table=PRICES) == Decimal(actual)
    assert max_cost(model, inputs, outputs, price_table=PRICES) == Decimal(maximum)


def test_helpers_use_settings_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "PRICE_TABLE",
        '{"configured":{"input_usd_per_million":"2","output_usd_per_million":"7"}}',
    )
    assert actual_cost("configured", 1_000_000, 1_000_000) == Decimal("9")
    assert max_cost("configured", 1, 0) == Decimal("0.000002")


def test_decimal_context_cannot_round_cost_or_spend_down(tmp_path: Path) -> None:
    prices = {MODEL: TokenPrice(input_usd_per_million=Decimal("0.1234567890123456789"))}
    with localcontext() as context:
        context.prec = 2
        assert actual_cost(MODEL, 123_456_789, 0, price_table=prices) == Decimal(
            "15.2415787517146788750190521"
        )
        assert max_cost(MODEL, 123_456_789, 0, price_table=prices) == Decimal("15.241579")
        ledger = _ledger(tmp_path / "precise.sqlite", "2.0000009")
        ledger.reserve("ask", MODEL, Decimal("1.2345671"))
        assert ledger.spent_usd() == Decimal("1.234568")
        assert ledger.remaining_usd() == Decimal("0.765432")


@pytest.mark.parametrize("helper", [max_cost, actual_cost])
def test_unknown_model_is_never_free(helper) -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        helper("missing", 0, 0, price_table=PRICES)


def test_unknown_model_cannot_reserve(ledger: BudgetLedger) -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        ledger.reserve("ask", "missing", Decimal("0"))
    assert ledger.spent_usd() == 0


@pytest.mark.parametrize("tokens", [-1, 1.5, True, 2**63])
@pytest.mark.parametrize("helper", [max_cost, actual_cost])
def test_invalid_tokens_are_rejected(helper, tokens) -> None:
    for inputs, outputs in ((tokens, 0), (0, tokens)):
        with pytest.raises((TypeError, ValueError)):
            helper(MODEL, inputs, outputs, price_table=PRICES)


@pytest.mark.parametrize("amount", ["-1", "NaN", "Infinity", "-Infinity", "1e20"])
def test_invalid_money_is_rejected(ledger: BudgetLedger, tmp_path: Path, amount: str) -> None:
    value = Decimal(amount)
    with pytest.raises(ValueError):
        ledger.reserve("ask", MODEL, value)
    with pytest.raises(ValueError):
        _ledger(tmp_path / "invalid.sqlite", amount)
    reservation = ledger.reserve("ask", MODEL, Decimal("1"))
    with pytest.raises(ValueError):
        reservation.settle_cost(value)
    assert reservation.status == "reserved"


def test_money_api_rejects_floats(ledger: BudgetLedger) -> None:
    with pytest.raises(TypeError, match="Decimal"):
        ledger.reserve("ask", MODEL, 0.1)


def test_reserve_under_at_and_over_cap(ledger: BudgetLedger) -> None:
    first = ledger.reserve("ingest", MODEL, Decimal("1"))
    assert first.status == "reserved"
    assert ledger.spent_usd() == Decimal("1")
    assert ledger.remaining_usd() == Decimal("2")
    ledger.reserve("ask", MODEL, Decimal("2"))
    assert ledger.remaining_usd() == 0
    ledger.reserve("free", MODEL, Decimal("0"))
    with pytest.raises(BudgetExceeded) as error:
        ledger.reserve("eval", MODEL, Decimal("0.0000001"))
    assert error.value.remaining_usd == 0
    assert error.value.group_id is None
    assert ledger.spent_usd() == Decimal("3")
    assert ledger.summary()["reserved"] == 3


def test_budget_exception_reports_available_amount(ledger: BudgetLedger) -> None:
    ledger.reserve("ask", MODEL, Decimal("0.25"))
    with pytest.raises(BudgetExceeded) as error:
        ledger.reserve("ask", MODEL, Decimal("3"))
    assert error.value.remaining_usd == Decimal("2.75")
    assert ledger.summary()["reserved"] == 1


def test_rounding_cannot_exceed_fractional_cap(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "spend.sqlite", "0.0000019")
    ledger.reserve("ask", MODEL, Decimal("0.0000001"))
    assert ledger.spent_usd() == Decimal("0.000001")
    assert ledger.remaining_usd() == 0
    with pytest.raises(BudgetExceeded):
        ledger.reserve("ask", MODEL, Decimal("0.0000001"))


def test_group_cap_counts_only_its_group_and_does_not_replace_global(ledger: BudgetLedger) -> None:
    ledger.reserve("eval", MODEL, Decimal("0.4"), group_id="a", group_cap_usd=Decimal("0.5"))
    ledger.reserve("eval", MODEL, Decimal("0.1"), group_id="a", group_cap_usd=Decimal("0.5"))
    with pytest.raises(BudgetExceeded) as error:
        ledger.reserve("eval", MODEL, Decimal("0.01"), group_id="a", group_cap_usd=Decimal("0.5"))
    assert error.value.remaining_usd == 0
    assert error.value.group_id == "a"
    ledger.reserve("eval", MODEL, Decimal("2.5"), group_id="b", group_cap_usd=Decimal("3"))
    assert ledger.group_spent_usd("a") == Decimal("0.5")
    assert ledger.group_spent_usd("b") == Decimal("2.5")
    assert ledger.group_spent_usd("unknown") == 0
    with pytest.raises(BudgetExceeded) as error:
        ledger.reserve("eval", MODEL, Decimal("0.1"), group_id="b", group_cap_usd=Decimal("3"))
    assert error.value.group_id is None


def test_group_cap_rounds_down_and_reports_remaining(ledger: BudgetLedger) -> None:
    ledger.reserve("eval", MODEL, Decimal("0.1"), group_id="a", group_cap_usd=Decimal("0.2000009"))
    with pytest.raises(BudgetExceeded) as error:
        ledger.reserve(
            "eval", MODEL, Decimal("0.100001"), group_id="a", group_cap_usd=Decimal("0.2000009")
        )
    assert error.value.remaining_usd == Decimal("0.1")


def test_group_cap_requires_group_id(ledger: BudgetLedger) -> None:
    with pytest.raises(ValueError, match="group ID"):
        ledger.reserve("eval", MODEL, Decimal("0.1"), group_cap_usd=Decimal("0.2"))
    with pytest.raises(ValueError, match="group ID"):
        ledger.group_spent_usd(None)


def test_settle_usage_releases_only_unused_reservation(ledger: BudgetLedger) -> None:
    with ledger.reserve("ask", MODEL, Decimal("1"), group_id="a") as reservation:
        reservation.settle(1_000_000, 200_000)
    assert reservation.status == "settled"
    assert not reservation.over_reserved
    assert ledger.spent_usd() == Decimal("0.2")
    assert ledger.group_spent_usd("a") == Decimal("0.2")
    assert ledger.remaining_usd() == Decimal("2.8")
    row = _row(ledger, reservation.id)
    assert row["reserved_micro_usd"] == 1_000_000
    assert row["actual_micro_usd"] == 200_000
    assert row["input_tokens"] == 1_000_000
    assert row["output_tokens"] == 200_000
    assert row["operation"] == "ask"
    assert row["model"] == MODEL
    assert row["group_id"] == "a"
    assert row["price_table_version"] == VERSION
    assert datetime.fromisoformat(row["created_at"]).utcoffset().total_seconds() == 0
    assert datetime.fromisoformat(row["settled_at"]) >= datetime.fromisoformat(row["created_at"])


@pytest.mark.parametrize("use_tokens", [False, True])
def test_settlement_rounds_fractional_micro_up(ledger: BudgetLedger, use_tokens: bool) -> None:
    reservation = ledger.reserve("ask", MODEL, Decimal("1"))
    if use_tokens:
        reservation.settle(1, 0)
    else:
        reservation.settle_cost(Decimal("0.0000001"))
    assert ledger.spent_usd() == Decimal("0.000001")


def test_zero_actual_releases_reservation(ledger: BudgetLedger) -> None:
    with ledger.reserve("ask", MODEL, Decimal("3")) as reservation:
        reservation.settle_cost(Decimal("0"))
    assert ledger.spent_usd() == 0
    assert ledger.remaining_usd() == Decimal("3")
    assert _row(ledger, reservation.id)["input_tokens"] is None


@pytest.mark.parametrize("fail", [False, True])
def test_unsettled_context_is_abandoned_and_counted(ledger: BudgetLedger, fail: bool) -> None:
    try:
        with ledger.reserve("ask", MODEL, Decimal("2"), group_id="eval") as reservation:
            if fail:
                raise TimeoutError("uncertain provider outcome")
    except TimeoutError:
        assert fail
    assert reservation.status == "abandoned"
    assert ledger.spent_usd() == Decimal("2")
    assert ledger.group_spent_usd("eval") == Decimal("2")
    assert _row(ledger, reservation.id)["actual_micro_usd"] is None
    with pytest.raises(ReservationStateError):
        reservation.cancel()
    with pytest.raises(ReservationStateError):
        reservation.settle_cost(Decimal("0"))


def test_cancel_only_unsent_reserved_call(ledger: BudgetLedger) -> None:
    with ledger.reserve("ask", MODEL, Decimal("3"), group_id="eval") as reservation:
        reservation.cancel()
    assert reservation.status == "cancelled"
    assert ledger.spent_usd() == 0
    assert ledger.group_spent_usd("eval") == 0
    assert ledger.remaining_usd() == Decimal("3")
    with pytest.raises(ReservationStateError):
        reservation.cancel()
    with pytest.raises(ReservationStateError):
        reservation.settle(0, 0)


def test_actual_above_reserved_is_recorded_and_flagged(ledger: BudgetLedger) -> None:
    with ledger.reserve("ask", MODEL, Decimal("0.1"), group_id="eval") as reservation:
        reservation.settle_cost(Decimal("3.5"))
    assert reservation.over_reserved
    assert reservation.status == "settled"
    assert ledger.spent_usd() == Decimal("3.5")
    assert ledger.group_spent_usd("eval") == Decimal("3.5")
    assert ledger.remaining_usd() == 0
    assert ledger.summary()["overruns"] == 1
    with pytest.raises(BudgetExceeded):
        ledger.reserve("ask", MODEL, Decimal("0"))
    with pytest.raises(ReservationStateError):
        reservation.cancel()


def test_double_settle_cannot_rewrite_spend(ledger: BudgetLedger) -> None:
    reservation = ledger.reserve("ask", MODEL, Decimal("1"))
    reservation.settle_cost(Decimal("0.4"))
    for settle in (lambda: reservation.settle(0, 0), lambda: reservation.settle_cost(Decimal("0"))):
        with pytest.raises(ReservationStateError):
            settle()
    assert ledger.spent_usd() == Decimal("0.4")


def test_concurrent_double_settle_has_one_winner(ledger: BudgetLedger) -> None:
    reservation = ledger.reserve("ask", MODEL, Decimal("1"))
    barrier = threading.Barrier(8)

    def settle_once(_: int) -> bool:
        barrier.wait(timeout=20)
        try:
            reservation.settle_cost(Decimal("0.25"))
        except ReservationStateError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert sum(executor.map(settle_once, range(8))) == 1
    assert ledger.spent_usd() == Decimal("0.25")


def test_settled_call_survives_exception_on_context_exit(ledger: BudgetLedger) -> None:
    with (
        pytest.raises(RuntimeError, match="after settlement"),
        ledger.reserve("ask", MODEL, Decimal("1")) as reservation,
    ):
        reservation.settle_cost(Decimal("0.25"))
        raise RuntimeError("after settlement")
    assert reservation.status == "settled"
    assert ledger.spent_usd() == Decimal("0.25")


def test_summary_has_consistent_counts_and_amounts(ledger: BudgetLedger) -> None:
    ledger.reserve("ask", MODEL, Decimal("1"))
    ledger.reserve("ask", MODEL, Decimal("1")).settle_cost(Decimal("0.25"))
    with ledger.reserve("ask", MODEL, Decimal("0.5")):
        pass
    ledger.reserve("ask", MODEL, Decimal("1")).cancel()
    assert ledger.summary() == {
        "cap_usd": Decimal("3"),
        "spent_usd": Decimal("1.75"),
        "remaining_usd": Decimal("1.25"),
        "reserved": 1,
        "in_flight": 0,
        "settled": 1,
        "abandoned": 1,
        "cancelled": 1,
        "overruns": 0,
    }


def test_persistence_and_price_versions(tmp_path: Path) -> None:
    path = tmp_path / "spend.sqlite"
    first = _ledger(path)
    first.reserve("ask", MODEL, Decimal("0.5")).settle_cost(Decimal("0.2"))
    unresolved = first.reserve("ask", MODEL, Decimal("2"))
    second = _ledger(path, version="new-prices")
    assert second.spent_usd() == Decimal("2.2")
    assert second.summary()["reserved"] == 1
    assert _row(second, unresolved.id)["price_table_version"] == VERSION
    newer = second.reserve("ask", MODEL, Decimal("0.8"))
    assert _row(first, newer.id)["price_table_version"] == "new-prices"
    with pytest.raises(BudgetExceeded):
        second.reserve("ask", MODEL, Decimal("0.000001"))


def test_cap_cannot_be_increased_by_reopening(ledger: BudgetLedger) -> None:
    with pytest.raises(ValueError, match="fixed budget cap"):
        _ledger(ledger.db_path, "2")
    assert _ledger(ledger.db_path).remaining_usd() == Decimal("3")


def test_relative_path_stays_bound_to_original_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    ledger = _ledger(Path("runtime/spend.sqlite"))
    ledger.reserve("ask", MODEL, Decimal("3"))
    monkeypatch.chdir(tmp_path / "runtime")
    assert ledger.spent_usd() == Decimal("3")
    with pytest.raises(BudgetExceeded):
        ledger.reserve("ask", MODEL, Decimal("0.01"))


def test_settlement_uses_snapshot_of_supplied_prices(tmp_path: Path) -> None:
    prices = {MODEL: TokenPrice(input_usd_per_million=Decimal("1"))}
    ledger = BudgetLedger(tmp_path / "spend.sqlite", Decimal("3"), VERSION, price_table=prices)
    prices[MODEL].input_usd_per_million = Decimal("0")
    reservation = ledger.reserve("ask", MODEL, Decimal("1"))
    reservation.settle(1_000_000, 0)
    assert ledger.spent_usd() == Decimal("1")


def test_creates_directories_schema_and_wal(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "nested" / "runtime" / "spend.sqlite")
    with sqlite3.connect(ledger.db_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    with ledger._connect() as connection:
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 10_000
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2


def test_unknown_schema_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "future.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 99")
    with pytest.raises(ValueError, match="Unsupported budget schema version: 99"):
        _ledger(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 99


def test_in_memory_database_is_rejected() -> None:
    with pytest.raises(ValueError, match="durable"):
        BudgetLedger(":memory:", Decimal("3"), VERSION)


def test_sql_values_are_parameterized(ledger: BudgetLedger) -> None:
    hostile = "eval'); DROP TABLE reservations; --"
    reservation = ledger.reserve(hostile, MODEL, Decimal("0.1"), group_id=hostile)
    assert _row(ledger, reservation.id)["operation"] == hostile
    assert ledger.group_spent_usd(hostile) == Decimal("0.1")
    assert ledger.summary()["reserved"] == 1


@pytest.mark.parametrize("group_cap", [False, True])
def test_many_threads_cannot_exceed_cap(tmp_path: Path, group_cap: bool) -> None:
    path = tmp_path / "threaded.sqlite"
    barrier = threading.Barrier(16)
    cap = "3" if group_cap else "0.37"

    def worker(index: int) -> int:
        # Race initialization too; some callers share an object, others a connection path.
        barrier.wait(timeout=20)
        current = shared if index % 2 else _ledger(path, cap)
        accepted = 0
        for _ in range(8):
            try:
                current.reserve(
                    "eval",
                    MODEL,
                    Decimal("0.01"),
                    group_id="eval",
                    group_cap_usd=Decimal("0.37") if group_cap else None,
                )
                accepted += 1
            except BudgetExceeded:
                pass
        return accepted

    shared = _ledger(path, cap)
    with ThreadPoolExecutor(max_workers=16) as executor:
        accepted = sum(executor.map(worker, range(16)))
    assert accepted == 37
    assert shared.spent_usd() == Decimal("0.37")
    assert shared.summary()["reserved"] == 37


def _process_reservations(path, start, sender, group_cap) -> None:
    try:
        if not start.wait(timeout=20):
            raise TimeoutError("Process race did not start")
        ledger = _ledger(path, "3" if group_cap else "0.23")
        accepted = 0
        for _ in range(16):
            try:
                ledger.reserve(
                    "eval",
                    MODEL,
                    Decimal("0.01"),
                    group_id="eval",
                    group_cap_usd=Decimal("0.23") if group_cap else None,
                )
                accepted += 1
            except BudgetExceeded:
                pass
        sender.send(("ok", accepted))
    except BaseException as error:
        sender.send(("error", repr(error)))
    finally:
        sender.close()


def _crash_after_reserving(path: Path) -> None:
    ledger = _ledger(path)
    with ledger.reserve("ask", MODEL, Decimal("3")):
        os._exit(17)


def test_abrupt_process_crash_never_refunds_reservation(tmp_path: Path) -> None:
    path = tmp_path / "crash.sqlite"
    process = multiprocessing.get_context("spawn").Process(
        target=_crash_after_reserving, args=(path,)
    )
    process.start()
    try:
        process.join(timeout=30)
        assert process.exitcode == 17
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        process.close()
    ledger = _ledger(path)
    assert ledger.summary()["reserved"] == 1
    assert ledger.spent_usd() == Decimal("3")
    with pytest.raises(BudgetExceeded):
        ledger.reserve("ask", MODEL, Decimal("0.000001"))


@pytest.mark.parametrize("group_cap", [False, True])
def test_process_race_and_crash_reservations_persist(tmp_path: Path, group_cap: bool) -> None:
    path = tmp_path / "processes.sqlite"
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    processes = []
    readers = []
    for _ in range(4):
        reader, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_process_reservations, args=(path, start, sender, group_cap)
        )
        process.start()
        sender.close()
        readers.append(reader)
        processes.append(process)
    start.set()
    try:
        results = []
        for reader in readers:
            assert reader.poll(30), "Reservation process timed out"
            results.append(reader.recv())
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        assert all(status == "ok" for status, _ in results), results
        assert sum(count for _, count in results) == 23
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            process.close()
        for reader in readers:
            reader.close()
    ledger = _ledger(path, "3" if group_cap else "0.23")
    assert ledger.spent_usd() == Decimal("0.23")
    assert ledger.group_spent_usd("eval") == Decimal("0.23")
    assert ledger.summary()["reserved"] == 23
    with pytest.raises(BudgetExceeded):
        ledger.reserve(
            "eval",
            MODEL,
            Decimal("0.01"),
            group_id="eval",
            group_cap_usd=Decimal("0.23") if group_cap else None,
        )


def test_cap_above_hard_limit_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        _ledger(tmp_path / "spend.sqlite", "3.000001")


def test_tampered_stored_cap_is_clamped_to_hard_limit(ledger: BudgetLedger) -> None:
    with sqlite3.connect(ledger.db_path) as connection:
        connection.execute("UPDATE budget_metadata SET cap_micro_usd = 4000000")
    ledger.reserve("ask", MODEL, Decimal("3"))
    with pytest.raises(BudgetExceeded):
        ledger.reserve("ask", MODEL, Decimal("0.000001"))


@pytest.mark.parametrize("later_cap", [None, "1.5"])
def test_group_cap_is_fixed_by_first_reservation(
    ledger: BudgetLedger, later_cap: str | None
) -> None:
    ledger.reserve("eval", MODEL, Decimal("0.5"), group_id="a", group_cap_usd=Decimal("0.5"))
    if later_cap is None:
        with pytest.raises(BudgetExceeded):
            ledger.reserve("eval", MODEL, Decimal("0.5"), group_id="a")
    else:
        with pytest.raises(ValueError, match="different fixed cap"):
            ledger.reserve(
                "eval", MODEL, Decimal("0.5"), group_id="a", group_cap_usd=Decimal(later_cap)
            )
    assert ledger.group_spent_usd("a") == Decimal("0.5")


def test_group_started_without_cap_cannot_gain_one_later(ledger: BudgetLedger) -> None:
    ledger.reserve("eval", MODEL, Decimal("0.5"), group_id="a")
    with pytest.raises(ValueError, match="without a cap"):
        ledger.reserve("eval", MODEL, Decimal("0.1"), group_id="a", group_cap_usd=Decimal("1"))


def test_open_group_fixes_persisted_cap_without_a_reservation(ledger: BudgetLedger) -> None:
    ledger.open_group("eval", Decimal("0.2"))
    assert ledger.spent_usd() == 0
    assert ledger.summary()["reserved"] == 0
    ledger.open_group("eval", Decimal("0.2"))
    reopened = _ledger(ledger.db_path)
    reopened.reserve("eval", MODEL, Decimal("0.2"), group_id="eval")
    with pytest.raises(BudgetExceeded) as error:
        ledger.reserve("eval", MODEL, Decimal("0.000001"), group_id="eval")
    assert error.value.group_id == "eval"
    assert error.value.remaining_usd == 0
    # Even an exhausted group can be reopened idempotently.
    ledger.open_group("eval", Decimal("0.2"))


def test_open_group_conflicts_with_existing_fixed_cap(ledger: BudgetLedger) -> None:
    ledger.open_group("a", Decimal("0.2"))
    with pytest.raises(ValueError, match="different fixed cap"):
        ledger.open_group("a", Decimal("0.3"))
    ledger.reserve("eval", MODEL, Decimal("0"), group_id="b", group_cap_usd=Decimal("0.1"))
    ledger.open_group("b", Decimal("0.1"))
    with pytest.raises(ValueError, match="different fixed cap"):
        ledger.open_group("b", Decimal("0.2"))


@pytest.mark.parametrize("cancel", [True, False])
def test_open_group_rejects_preexisting_uncapped_reservations(
    ledger: BudgetLedger, cancel: bool
) -> None:
    reservation = ledger.reserve("eval", MODEL, Decimal("0"), group_id="a")
    if cancel:
        reservation.cancel()
    with pytest.raises(ValueError, match="started without a cap"):
        ledger.open_group("a", Decimal("0.1"))


@pytest.mark.parametrize("group_id", [None, "", 123])
def test_open_group_rejects_invalid_id(ledger: BudgetLedger, group_id) -> None:
    with pytest.raises(ValueError, match="nonempty group ID"):
        ledger.open_group(group_id, Decimal("0.1"))


@pytest.mark.parametrize("cap", [Decimal("-1"), Decimal("NaN"), Decimal("Infinity"), 0.1])
def test_open_group_rejects_invalid_money(ledger: BudgetLedger, cap) -> None:
    with pytest.raises((TypeError, ValueError)):
        ledger.open_group("eval", cap)


def test_open_group_rounds_cap_down(ledger: BudgetLedger) -> None:
    ledger.open_group("eval", Decimal("0.0000019"))
    ledger.open_group("eval", Decimal("0.0000011"))
    with pytest.raises(BudgetExceeded):
        ledger.reserve("eval", MODEL, Decimal("0.000002"), group_id="eval")
    ledger.reserve("eval", MODEL, Decimal("0.000001"), group_id="eval")


def test_concurrent_open_group_never_changes_winning_cap(ledger: BudgetLedger) -> None:
    start = threading.Barrier(2)

    def open_cap(cap: str) -> Decimal | None:
        start.wait()
        try:
            ledger.open_group("eval", Decimal(cap))
            return Decimal(cap)
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(open_cap, ["0.1", "0.2"]))
    winners = [cap for cap in results if cap is not None]
    assert len(winners) == 1
    ledger.reserve("eval", MODEL, winners[0], group_id="eval")
    with pytest.raises(BudgetExceeded):
        ledger.reserve("eval", MODEL, Decimal("0.000001"), group_id="eval")


def test_sent_call_cannot_be_cancelled_and_stays_counted(ledger: BudgetLedger) -> None:
    reservation = ledger.reserve("ask", MODEL, Decimal("1"))
    reservation.mark_sent()
    assert reservation.status == "in_flight"
    with pytest.raises(ReservationStateError):
        reservation.cancel()
    with pytest.raises(ReservationStateError):
        reservation.mark_sent()
    assert ledger.spent_usd() == Decimal("1")
    reservation.settle_cost(Decimal("0.3"))
    assert ledger.spent_usd() == Decimal("0.3")


def test_cancelled_call_cannot_be_marked_sent(ledger: BudgetLedger) -> None:
    reservation = ledger.reserve("ask", MODEL, Decimal("1"))
    reservation.cancel()
    with pytest.raises(ReservationStateError):
        reservation.mark_sent()


def test_in_flight_call_left_unsettled_is_abandoned(ledger: BudgetLedger) -> None:
    with ledger.reserve("ask", MODEL, Decimal("1")) as reservation:
        reservation.mark_sent()
    assert reservation.status == "abandoned"
    assert ledger.spent_usd() == Decimal("1")
