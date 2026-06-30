import sqlite3
from pathlib import Path

from retrievalbench.model import ExperimentRun


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
