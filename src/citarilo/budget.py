"""Durable, conservative accounting for paid calls; reserve before sending a request."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from uuid import uuid4

from citarilo.config import Settings, TokenPrice

SCHEMA_VERSION = 1
HARD_CAP_USD = Decimal("3")
_MICRO_USD = 1_000_000
_MAX_SQLITE_INTEGER = 2**63 - 1
_COMMITTED = """
    CASE WHEN status = 'settled' THEN actual_micro_usd
         WHEN status IN ('reserved', 'in_flight', 'abandoned') THEN reserved_micro_usd
         ELSE 0 END
"""


class BudgetExceeded(RuntimeError):
    """A global or group limit cannot accommodate a reservation."""

    def __init__(self, remaining_usd: Decimal, *, group_id: str | None = None) -> None:
        self.remaining_usd = remaining_usd
        self.group_id = group_id
        scope = "Global budget" if group_id is None else f"Group {group_id!r} budget"
        super().__init__(f"{scope} exceeded; ${remaining_usd} remaining")


class ReservationStateError(RuntimeError):
    """A reservation has already been settled, abandoned, or cancelled."""


def _money(value: Decimal) -> Fraction:
    if not isinstance(value, Decimal):
        raise TypeError("Money must be a Decimal")
    if not value.is_finite() or value < 0:
        raise ValueError("Money must be finite and nonnegative")
    return Fraction(value)


def _micros(value: Decimal, *, round_up: bool = True) -> int:
    amount = _money(value) * _MICRO_USD
    result = (
        -(-amount.numerator // amount.denominator)
        if round_up
        else amount.numerator // amount.denominator
    )
    if result > _MAX_SQLITE_INTEGER:
        raise ValueError("Amount exceeds SQLite's integer range")
    return result


def _usd(micros: int) -> Decimal:
    # String construction is exact even under a caller's low Decimal precision.
    return Decimal(f"{micros}e-6")


def _tokens(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("Token counts must be integers")
    if not 0 <= value <= _MAX_SQLITE_INTEGER:
        raise ValueError("Token counts must be nonnegative SQLite integers")
    return value


def _price(model: str, price_table: Mapping[str, TokenPrice]) -> TokenPrice:
    try:
        return price_table[model]
    except KeyError:
        raise ValueError(f"Unknown model: {model!r}") from None


def actual_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    price_table: Mapping[str, TokenPrice] | None = None,
) -> Decimal:
    """Return exact token cost in USD, before ledger micro-USD rounding."""
    prices = Settings().price_table if price_table is None else price_table
    price = _price(model, prices)
    # Add integer coefficients at a common decimal exponent, without context rounding.
    rates = (price.input_usd_per_million, price.output_usd_per_million)
    for rate in rates:
        _money(rate)
    exponent = min(rate.as_tuple().exponent for rate in rates)
    total = 0
    for rate, tokens in zip(rates, (input_tokens, output_tokens), strict=True):
        parts = rate.as_tuple()
        coefficient = int("".join(map(str, parts.digits)))
        total += coefficient * 10 ** (parts.exponent - exponent) * _tokens(tokens)
    return Decimal(f"{total}e{exponent - 6}")


def max_cost(
    model: str,
    max_input_tokens: int,
    max_output_tokens: int,
    *,
    price_table: Mapping[str, TokenPrice] | None = None,
) -> Decimal:
    """Return worst-case USD cost rounded up to a whole micro-USD."""
    return _usd(
        _micros(actual_cost(model, max_input_tokens, max_output_tokens, price_table=price_table))
    )


class BudgetLedger:
    """A file-backed ledger shared by threads/processes using the same database.

    The first opener fixes the global cap (at most HARD_CAP_USD); conflicting later
    caps are rejected. The first reservation of a group fixes that group's cap.
    Call Reservation.mark_sent() immediately before dispatching the request.
    Pass the same Settings.price_table used to estimate calls when overriding prices.
    Charges round up and caps round down to micro-USD. Actual usage above a
    reservation is recorded honestly and blocks later spending if it exceeds the cap;
    callers must reserve valid upper bounds to prevent provider-side overspend.
    """

    def __init__(
        self,
        db_path: str | Path,
        cap_usd: Decimal,
        price_table_version: str,
        *,
        price_table: Mapping[str, TokenPrice] | None = None,
    ) -> None:
        if str(db_path) == ":memory:":
            raise ValueError("A durable file-backed database is required")
        if not price_table_version:
            raise ValueError("price_table_version must not be empty")
        if _money(cap_usd) > HARD_CAP_USD:
            raise ValueError(f"The budget cap must not exceed ${HARD_CAP_USD}")
        self.db_path = Path(db_path).resolve()
        cap_micro_usd = _micros(cap_usd, round_up=False)
        self.price_table_version = price_table_version
        prices = Settings().price_table if price_table is None else price_table
        self._prices = {model: price.model_copy(deep=True) for model, price in prices.items()}
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if mode != "wal":
                raise RuntimeError("The budget database requires WAL mode")
        with self._transaction() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise ValueError(f"Unsupported budget schema version: {version}")
            if version == 0:
                connection.execute("""
                    CREATE TABLE budget_metadata (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        cap_micro_usd INTEGER NOT NULL CHECK (cap_micro_usd >= 0)
                    )
                """)
                connection.execute(
                    "INSERT INTO budget_metadata (id, cap_micro_usd) VALUES (1, ?)",
                    (cap_micro_usd,),
                )
                connection.execute("""
                    CREATE TABLE budget_groups (
                        group_id TEXT PRIMARY KEY,
                        cap_micro_usd INTEGER NOT NULL CHECK (cap_micro_usd >= 0)
                    )
                """)
                connection.execute("""
                    CREATE TABLE reservations (
                        id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        settled_at TEXT,
                        operation TEXT NOT NULL,
                        model TEXT NOT NULL,
                        group_id TEXT,
                        reserved_micro_usd INTEGER NOT NULL CHECK (reserved_micro_usd >= 0),
                        actual_micro_usd INTEGER CHECK (actual_micro_usd >= 0),
                        input_tokens INTEGER CHECK (input_tokens >= 0),
                        output_tokens INTEGER CHECK (output_tokens >= 0),
                        status TEXT NOT NULL CHECK (
                            status IN (
                                'reserved', 'in_flight', 'settled', 'abandoned', 'cancelled'
                            )
                        ),
                        price_table_version TEXT NOT NULL,
                        over_reserved INTEGER NOT NULL DEFAULT 0 CHECK (over_reserved IN (0, 1)),
                        CHECK ((status = 'settled') = (actual_micro_usd IS NOT NULL))
                    )
                """)
                connection.execute("CREATE INDEX reservations_group ON reservations (group_id)")
                connection.execute("PRAGMA user_version = 1")
            if self._cap(connection) != cap_micro_usd:
                raise ValueError("The requested cap differs from the database's fixed budget cap")

    @staticmethod
    def _cap(connection: sqlite3.Connection) -> int:
        # Always read the persisted cap; instance state is never trusted for enforcement.
        cap = connection.execute("SELECT cap_micro_usd FROM budget_metadata WHERE id = 1")
        return min(cap.fetchone()[0], _micros(HARD_CAP_USD, round_up=False))

    @staticmethod
    def _group_cap(
        connection: sqlite3.Connection, group_id: str, requested: int | None
    ) -> int | None:
        row = connection.execute(
            "SELECT cap_micro_usd FROM budget_groups WHERE group_id = ?", (group_id,)
        ).fetchone()
        if row is not None:
            if requested is not None and requested != row[0]:
                raise ValueError(f"Group {group_id!r} already has a different fixed cap")
            return row[0]
        if requested is not None:
            if connection.execute(
                "SELECT 1 FROM reservations WHERE group_id = ? LIMIT 1", (group_id,)
            ).fetchone():
                raise ValueError(f"Group {group_id!r} started without a cap")
            connection.execute(
                "INSERT INTO budget_groups (group_id, cap_micro_usd) VALUES (?, ?)",
                (group_id, requested),
            )
        return requested

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @staticmethod
    def _committed(connection: sqlite3.Connection, group_id: str | None = None) -> int:
        # Python sums cannot overflow after an unexpectedly expensive settlement.
        query = f"SELECT {_COMMITTED} FROM reservations"
        parameters = () if group_id is None else (group_id,)
        if group_id is not None:
            query += " WHERE group_id = ?"
        return sum(row[0] for row in connection.execute(query, parameters))

    def reserve(
        self,
        operation: str,
        model: str,
        max_cost_usd: Decimal,
        *,
        group_id: str | None = None,
        group_cap_usd: Decimal | None = None,
    ) -> Reservation:
        """Commit a reservation before sending; a group's first call fixes its cap."""
        _price(model, self._prices)
        amount = _micros(max_cost_usd)
        if not operation or not model or group_id == "":
            raise ValueError("Operation, model, and supplied group ID must not be empty")
        if group_cap_usd is not None and group_id is None:
            raise ValueError("A group cap requires a group ID")
        group_cap = None if group_cap_usd is None else _micros(group_cap_usd, round_up=False)
        reservation_id = uuid4().hex
        with self._transaction() as connection:
            remaining = self._cap(connection) - self._committed(connection)
            if amount > remaining:
                raise BudgetExceeded(_usd(max(0, remaining)))
            if group_id is not None:
                group_cap = self._group_cap(connection, group_id, group_cap)
            if group_cap is not None:
                remaining = group_cap - self._committed(connection, group_id)
                if amount > remaining:
                    raise BudgetExceeded(_usd(max(0, remaining)), group_id=group_id)
            connection.execute(
                """INSERT INTO reservations (
                    id, created_at, operation, model, group_id, reserved_micro_usd,
                    status, price_table_version
                ) VALUES (?, ?, ?, ?, ?, ?, 'reserved', ?)""",
                (
                    reservation_id,
                    datetime.now(UTC).isoformat(),
                    operation,
                    model,
                    group_id,
                    amount,
                    self.price_table_version,
                ),
            )
        return Reservation(self, reservation_id, model)

    def spent_usd(self) -> Decimal:
        """Return committed spend, including uncertain calls and live reservations."""
        with closing(self._connect()) as connection:
            return _usd(self._committed(connection))

    def remaining_usd(self) -> Decimal:
        """Return available global budget, clamped to zero after an overrun."""
        with closing(self._connect()) as connection:
            return _usd(max(0, self._cap(connection) - self._committed(connection)))

    def group_spent_usd(self, group_id: str) -> Decimal:
        """Return the group's settled and conservatively reserved spend."""
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("A nonempty group ID is required")
        with closing(self._connect()) as connection:
            return _usd(self._committed(connection, group_id))

    def summary(self) -> dict[str, Decimal | int]:
        """Return a consistent snapshot of budget amounts and per-status row counts."""
        with closing(self._connect()) as connection:
            cap = self._cap(connection)
            rows = connection.execute(
                f"SELECT status, over_reserved, {_COMMITTED} AS committed FROM reservations"
            ).fetchall()
        spent = sum(row["committed"] for row in rows)
        return {
            "cap_usd": _usd(cap),
            "spent_usd": _usd(spent),
            "remaining_usd": _usd(max(0, cap - spent)),
            **{
                status: sum(row["status"] == status for row in rows)
                for status in ("reserved", "in_flight", "settled", "abandoned", "cancelled")
            },
            "overruns": sum(row["over_reserved"] for row in rows),
        }


class Reservation:
    """A durable handle; leaving its context unsettled never refunds the charge."""

    def __init__(self, ledger: BudgetLedger, reservation_id: str, model: str) -> None:
        self._ledger = ledger
        self.id = reservation_id
        self.model = model

    def __enter__(self) -> Reservation:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        with self._ledger._transaction() as connection:
            connection.execute(
                """UPDATE reservations SET status = 'abandoned'
                    WHERE id = ? AND status IN ('reserved', 'in_flight')""",
                (self.id,),
            )

    @property
    def status(self) -> str:
        with closing(self._ledger._connect()) as connection:
            return connection.execute(
                "SELECT status FROM reservations WHERE id = ?", (self.id,)
            ).fetchone()[0]

    @property
    def over_reserved(self) -> bool:
        with closing(self._ledger._connect()) as connection:
            return bool(
                connection.execute(
                    "SELECT over_reserved FROM reservations WHERE id = ?", (self.id,)
                ).fetchone()[0]
            )

    def mark_sent(self) -> None:
        """Mark the call as dispatched; from now on it can never be cancelled."""
        with self._ledger._transaction() as connection:
            cursor = connection.execute(
                "UPDATE reservations SET status = 'in_flight' WHERE id = ? AND status = 'reserved'",
                (self.id,),
            )
            if cursor.rowcount != 1:
                raise ReservationStateError("Only a reserved, unsent call can be marked sent")

    def settle(self, input_tokens: int, output_tokens: int) -> None:
        """Record provider usage using the prices captured by the ledger."""
        cost = actual_cost(
            self.model, input_tokens, output_tokens, price_table=self._ledger._prices
        )
        self._settle(cost, input_tokens, output_tokens)

    def settle_cost(self, actual_usd: Decimal) -> None:
        """Record known actual cost, rounded up to micro-USD; token counts stay NULL."""
        self._settle(actual_usd, None, None)

    def _settle(
        self, actual_usd: Decimal, input_tokens: int | None, output_tokens: int | None
    ) -> None:
        amount = _micros(actual_usd)
        with self._ledger._transaction() as connection:
            cursor = connection.execute(
                """UPDATE reservations SET status = 'settled', settled_at = ?,
                    actual_micro_usd = ?, input_tokens = ?, output_tokens = ?,
                    over_reserved = (? > reserved_micro_usd)
                    WHERE id = ? AND status IN ('reserved', 'in_flight')""",
                (
                    datetime.now(UTC).isoformat(),
                    amount,
                    input_tokens,
                    output_tokens,
                    amount,
                    self.id,
                ),
            )
            if cursor.rowcount != 1:
                raise ReservationStateError("Only a reserved call can be settled")

    def cancel(self) -> None:
        """Refund ONLY a call that was never marked sent.

        Never use for timeouts, exceptions after sending, or missing usage: those
        calls may have been billed. In-flight, settled and abandoned calls cannot
        be cancelled.
        """
        with self._ledger._transaction() as connection:
            cursor = connection.execute(
                "UPDATE reservations SET status = 'cancelled' WHERE id = ? AND status = 'reserved'",
                (self.id,),
            )
            if cursor.rowcount != 1:
                raise ReservationStateError("Only an unsent reserved call can be cancelled")
