import sqlite3
from pathlib import Path

from retrievalbench.model import ExperimentRun, GoldenItem


class RunStore:
    """SQLite persistence for ExperimentRuns.

    Design (§8): no ORM. One run is stored as a single JSON blob in `data`
    (via model_dump_json), plus a few denormalized columns so listing/sorting
    runs doesn't require parsing every blob. Read back with model_validate_json.
    """

    def __init__(self, db_path: str | Path = "data/retrievalbench.db"):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        # IF NOT EXISTS -> idempotent: constructing the store twice is safe.
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id          TEXT PRIMARY KEY,
                corpus_id   TEXT NOT NULL,
                config_name TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                data        TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def save_run(self, run: ExperimentRun) -> None:
        # Parameterized (?) query: lets sqlite handle quoting/escaping and
        # closes the SQL-injection hole. INSERT OR REPLACE -> re-running a run
        # with the same id overwrites instead of erroring on the primary key.
        self.conn.execute(
            "INSERT OR REPLACE INTO runs "
            "(id, corpus_id, config_name, created_at, data) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                run.id,
                run.corpus_id,
                run.config.name,
                run.created_at.isoformat(),
                run.model_dump_json(),
            ),
        )
        self.conn.commit()

    def get_run(self, run_id: str) -> ExperimentRun | None:
        row = self.conn.execute(
            "SELECT data FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return ExperimentRun.model_validate_json(row[0])

    def list_runs(self) -> list[tuple[str, str, str]]:
        # Cheap listing: reads only denormalized columns, never parses JSON.
        return self.conn.execute(
            "SELECT id, config_name, created_at FROM runs ORDER BY created_at DESC"
        ).fetchall()

    def close(self) -> None:
        self.conn.close()


class GoldenStore:
    """SQLite persistence for generated+reviewed GoldenItems (Design §8).

    Separate from the hand-written GOLDEN_SET literal in golden.py: that
    literal stays the curated, git-versioned seed set; this store holds items
    `rbench gen-golden` produced and a human kept/edited, grown incrementally
    per corpus. Callers merge both sources at read time (see cli.py).
    """

    def __init__(self, db_path: str | Path = "data/retrievalbench.db"):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS golden_items (
                id        TEXT PRIMARY KEY,
                corpus_id TEXT NOT NULL,
                data      TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def load_golden_set(self, corpus_id: str) -> list[GoldenItem]:
        rows = self.conn.execute(
            "SELECT data FROM golden_items WHERE corpus_id = ?", (corpus_id,)
        ).fetchall()
        return [GoldenItem.model_validate_json(row[0]) for row in rows]

    def save_golden_set(self, corpus_id: str, items: list[GoldenItem]) -> None:
        """Replace this corpus's stored set with exactly `items`. Callers that
        want to grow the set incrementally (the gen-golden review flow) load
        the existing set first, merge in newly-kept items, and pass the union
        back in — this method itself does a clean replace, not a merge."""
        self.conn.execute("DELETE FROM golden_items WHERE corpus_id = ?", (corpus_id,))
        self.conn.executemany(
            "INSERT INTO golden_items (id, corpus_id, data) VALUES (?, ?, ?)",
            [(item.id, corpus_id, item.model_dump_json()) for item in items],
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
