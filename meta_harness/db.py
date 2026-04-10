"""
Meta-Harness Database — SQLite-backed storage for the outer-loop search.

Tables
------
harness          One row per harness variant (source snapshot, parentage, metadata).
search_iteration Logical grouping: one proposer turn that may yield several harnesses.
eval_run         One full evaluation of a harness on a task set.
task_result      Per-task outcome inside an eval_run.
compile_trace    Per-pass compiler output (the raw execution trace kept for the proposer).
harness_file     Versioned snapshot of every file that belongs to a harness.
pareto_frontier  Cached Pareto-optimal harness set (recomputed on demand).
"""

from __future__ import annotations

import json
import os
import sqlite3
import textwrap
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCHEMA_VERSION = 1

_SCHEMA_SQL = textwrap.dedent("""\
    -- versioning ---------------------------------------------------------------
    CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    -- harness variants ---------------------------------------------------------
    CREATE TABLE IF NOT EXISTS harness (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        name            TEXT    NOT NULL,
        parent_id       INTEGER REFERENCES harness(id),
        iteration_id    INTEGER REFERENCES search_iteration(id),
        description     TEXT,
        created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
        extra           TEXT    -- free-form JSON for proposer notes
    );

    -- search iterations --------------------------------------------------------
    CREATE TABLE IF NOT EXISTS search_iteration (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        iteration_num   INTEGER NOT NULL,
        proposer_model  TEXT,
        proposer_prompt TEXT,
        reasoning       TEXT,     -- proposer's reasoning trace
        started_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
        finished_at     TEXT,
        extra           TEXT
    );

    -- harness file snapshots ---------------------------------------------------
    CREATE TABLE IF NOT EXISTS harness_file (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        harness_id      INTEGER NOT NULL REFERENCES harness(id),
        rel_path        TEXT    NOT NULL,   -- e.g. "skills/ascendc/dsl-lowering/SKILL.md"
        content         TEXT    NOT NULL,
        sha256          TEXT,
        UNIQUE(harness_id, rel_path)
    );

    -- evaluation runs ----------------------------------------------------------
    CREATE TABLE IF NOT EXISTS eval_run (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        harness_id      INTEGER NOT NULL REFERENCES harness(id),
        run_type        TEXT    NOT NULL DEFAULT 'search',   -- search | test | debug
        started_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
        finished_at     TEXT,
        total_tasks     INTEGER NOT NULL DEFAULT 0,
        compile_pass    INTEGER NOT NULL DEFAULT 0,
        compile_fail    INTEGER NOT NULL DEFAULT 0,
        accuracy_pass   INTEGER NOT NULL DEFAULT 0,
        accuracy_fail   INTEGER NOT NULL DEFAULT 0,
        compile_rate    REAL,
        accuracy_rate   REAL,
        avg_speedup     REAL,
        extra           TEXT
    );

    -- per-task results ---------------------------------------------------------
    CREATE TABLE IF NOT EXISTS task_result (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        eval_run_id     INTEGER NOT NULL REFERENCES eval_run(id),
        task_name       TEXT    NOT NULL,   -- e.g. "level2/81_Gemm_Swish_Divide_Clamp_Tanh_Clamp"
        level           INTEGER,
        problem_id      INTEGER,
        op_name         TEXT,
        status          TEXT    NOT NULL,   -- compile_pass | compile_fail | accuracy_pass | accuracy_fail | timeout | error
        iterations_used INTEGER NOT NULL DEFAULT 0,
        wall_time_s     REAL,
        speedup         REAL,
        failure_reason  TEXT,
        generated_code  TEXT,
        extra           TEXT,
        created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
    );

    -- compiler traces (per pass) -----------------------------------------------
    CREATE TABLE IF NOT EXISTS compile_trace (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        task_result_id  INTEGER NOT NULL REFERENCES task_result(id),
        pass_name       TEXT    NOT NULL,   -- tiling_pass | init_pass | process_pass | nonaligned_pass
        attempt         INTEGER NOT NULL DEFAULT 1,
        success         INTEGER NOT NULL DEFAULT 0,   -- 0=fail, 1=pass
        compiler_output TEXT,       -- full stderr/stdout from build.sh
        error_category  TEXT,       -- auto-classified error bucket
        generated_code  TEXT,       -- code that was compiled
        fix_applied     TEXT,       -- description of fix from error_correction
        wall_time_s     REAL,
        created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
    );

    -- pareto frontier cache ----------------------------------------------------
    CREATE TABLE IF NOT EXISTS pareto_frontier (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        harness_id      INTEGER NOT NULL REFERENCES harness(id),
        eval_run_id     INTEGER NOT NULL REFERENCES eval_run(id),
        compile_rate    REAL    NOT NULL,
        accuracy_rate   REAL,
        avg_speedup     REAL,
        computed_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
    );

    -- indices ------------------------------------------------------------------
    CREATE INDEX IF NOT EXISTS idx_harness_iteration   ON harness(iteration_id);
    CREATE INDEX IF NOT EXISTS idx_harness_file_hid    ON harness_file(harness_id);
    CREATE INDEX IF NOT EXISTS idx_eval_run_hid        ON eval_run(harness_id);
    CREATE INDEX IF NOT EXISTS idx_task_result_run     ON task_result(eval_run_id);
    CREATE INDEX IF NOT EXISTS idx_task_result_status  ON task_result(status);
    CREATE INDEX IF NOT EXISTS idx_compile_trace_task  ON compile_trace(task_result_id);
    CREATE INDEX IF NOT EXISTS idx_compile_trace_pass  ON compile_trace(pass_name);
    CREATE INDEX IF NOT EXISTS idx_compile_trace_err   ON compile_trace(error_category);
""")


class MetaHarnessDB:
    """Manages the SQLite database for Meta-Harness search history."""

    def __init__(self, db_path: str | Path = "meta_harness.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    # -- connection management --------------------------------------------------

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def _ensure_schema(self):
        with self._cursor() as cur:
            cur.executescript(_SCHEMA_SQL)
            cur.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
                ("schema_version", str(_SCHEMA_VERSION)),
            )

    # =========================================================================
    # Search Iterations
    # =========================================================================

    def create_iteration(
        self,
        iteration_num: int,
        proposer_model: str = "",
        proposer_prompt: str = "",
        reasoning: str = "",
        extra: Optional[dict] = None,
    ) -> int:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO search_iteration
                   (iteration_num, proposer_model, proposer_prompt, reasoning, extra)
                   VALUES (?, ?, ?, ?, ?)""",
                (iteration_num, proposer_model, proposer_prompt, reasoning,
                 json.dumps(extra) if extra else None),
            )
            return cur.lastrowid

    def finish_iteration(self, iteration_id: int):
        with self._cursor() as cur:
            cur.execute(
                "UPDATE search_iteration SET finished_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
                (iteration_id,),
            )

    def get_iteration(self, iteration_id: int) -> Optional[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM search_iteration WHERE id = ?", (iteration_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def list_iterations(self) -> List[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM search_iteration ORDER BY iteration_num")
            return [dict(r) for r in cur.fetchall()]

    # =========================================================================
    # Harness Variants
    # =========================================================================

    def create_harness(
        self,
        name: str,
        parent_id: Optional[int] = None,
        iteration_id: Optional[int] = None,
        description: str = "",
        extra: Optional[dict] = None,
    ) -> int:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO harness
                   (name, parent_id, iteration_id, description, extra)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, parent_id, iteration_id, description,
                 json.dumps(extra) if extra else None),
            )
            return cur.lastrowid

    def get_harness(self, harness_id: int) -> Optional[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM harness WHERE id = ?", (harness_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def list_harnesses(
        self,
        iteration_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[dict]:
        with self._cursor() as cur:
            if iteration_id is not None:
                cur.execute(
                    "SELECT * FROM harness WHERE iteration_id = ? ORDER BY id DESC LIMIT ?",
                    (iteration_id, limit),
                )
            else:
                cur.execute("SELECT * FROM harness ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    def get_harness_lineage(self, harness_id: int) -> List[dict]:
        """Walk parent_id chain to build full ancestry."""
        lineage = []
        current = harness_id
        seen = set()
        while current and current not in seen:
            seen.add(current)
            h = self.get_harness(current)
            if not h:
                break
            lineage.append(h)
            current = h.get("parent_id")
        lineage.reverse()
        return lineage

    # =========================================================================
    # Harness File Snapshots
    # =========================================================================

    def save_harness_file(
        self,
        harness_id: int,
        rel_path: str,
        content: str,
        sha256: str = "",
    ):
        with self._cursor() as cur:
            cur.execute(
                """INSERT OR REPLACE INTO harness_file
                   (harness_id, rel_path, content, sha256)
                   VALUES (?, ?, ?, ?)""",
                (harness_id, rel_path, content, sha256),
            )

    def save_harness_files_from_dir(
        self,
        harness_id: int,
        base_dir: str | Path,
        patterns: Optional[List[str]] = None,
    ):
        """Snapshot files matching patterns (default: *.md, *.py) under base_dir."""
        import hashlib

        base = Path(base_dir)
        if patterns is None:
            patterns = ["**/*.md", "**/*.py"]
        for pattern in patterns:
            for fpath in base.glob(pattern):
                if fpath.is_file():
                    content = fpath.read_text(encoding="utf-8", errors="replace")
                    digest = hashlib.sha256(content.encode()).hexdigest()
                    rel = str(fpath.relative_to(base))
                    self.save_harness_file(harness_id, rel, content, digest)

    def get_harness_files(self, harness_id: int) -> List[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM harness_file WHERE harness_id = ? ORDER BY rel_path",
                (harness_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_harness_file(self, harness_id: int, rel_path: str) -> Optional[str]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT content FROM harness_file WHERE harness_id = ? AND rel_path = ?",
                (harness_id, rel_path),
            )
            row = cur.fetchone()
            return row["content"] if row else None

    def restore_harness_files(
        self,
        harness_id: int,
        target_dir: str | Path,
    ):
        """Write all stored files for a harness back to disk."""
        target = Path(target_dir)
        for f in self.get_harness_files(harness_id):
            out = target / f["rel_path"]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(f["content"], encoding="utf-8")

    def diff_harnesses(self, harness_a: int, harness_b: int) -> List[dict]:
        """Return files that differ between two harness snapshots."""
        files_a = {f["rel_path"]: f["sha256"] for f in self.get_harness_files(harness_a)}
        files_b = {f["rel_path"]: f["sha256"] for f in self.get_harness_files(harness_b)}
        all_paths = sorted(set(files_a) | set(files_b))
        diffs = []
        for p in all_paths:
            ha = files_a.get(p)
            hb = files_b.get(p)
            if ha != hb:
                diffs.append({
                    "rel_path": p,
                    "in_a": ha is not None,
                    "in_b": hb is not None,
                    "changed": ha is not None and hb is not None and ha != hb,
                })
        return diffs

    # =========================================================================
    # Evaluation Runs
    # =========================================================================

    def create_eval_run(
        self,
        harness_id: int,
        run_type: str = "search",
        extra: Optional[dict] = None,
    ) -> int:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO eval_run (harness_id, run_type, extra)
                   VALUES (?, ?, ?)""",
                (harness_id, run_type,
                 json.dumps(extra) if extra else None),
            )
            return cur.lastrowid

    def finish_eval_run(self, run_id: int):
        """Recompute aggregates from task_results and mark finished."""
        with self._cursor() as cur:
            cur.execute(
                """SELECT
                     COUNT(*)                                          AS total,
                     SUM(CASE WHEN status IN ('compile_pass','accuracy_pass','accuracy_fail') THEN 1 ELSE 0 END) AS comp_pass,
                     SUM(CASE WHEN status = 'compile_fail' THEN 1 ELSE 0 END)   AS comp_fail,
                     SUM(CASE WHEN status = 'accuracy_pass' THEN 1 ELSE 0 END)  AS acc_pass,
                     SUM(CASE WHEN status = 'accuracy_fail' THEN 1 ELSE 0 END)  AS acc_fail,
                     AVG(CASE WHEN speedup IS NOT NULL THEN speedup END)         AS avg_spd
                   FROM task_result WHERE eval_run_id = ?""",
                (run_id,),
            )
            row = cur.fetchone()
            total = row["total"] or 0
            comp_pass = row["comp_pass"] or 0
            comp_fail = row["comp_fail"] or 0
            acc_pass = row["acc_pass"] or 0
            acc_fail = row["acc_fail"] or 0
            avg_spd = row["avg_spd"]
            comp_rate = comp_pass / total if total else None
            acc_rate = acc_pass / (acc_pass + acc_fail) if (acc_pass + acc_fail) else None
            cur.execute(
                """UPDATE eval_run SET
                     finished_at   = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
                     total_tasks   = ?,
                     compile_pass  = ?,
                     compile_fail  = ?,
                     accuracy_pass = ?,
                     accuracy_fail = ?,
                     compile_rate  = ?,
                     accuracy_rate = ?,
                     avg_speedup   = ?
                   WHERE id = ?""",
                (total, comp_pass, comp_fail, acc_pass, acc_fail,
                 comp_rate, acc_rate, avg_spd, run_id),
            )

    def get_eval_run(self, run_id: int) -> Optional[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM eval_run WHERE id = ?", (run_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def list_eval_runs(
        self,
        harness_id: Optional[int] = None,
        run_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        with self._cursor() as cur:
            conditions, params = [], []
            if harness_id is not None:
                conditions.append("harness_id = ?")
                params.append(harness_id)
            if run_type is not None:
                conditions.append("run_type = ?")
                params.append(run_type)
            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            params.append(limit)
            cur.execute(
                f"SELECT * FROM eval_run {where} ORDER BY id DESC LIMIT ?", params
            )
            return [dict(r) for r in cur.fetchall()]

    # =========================================================================
    # Task Results
    # =========================================================================

    def add_task_result(
        self,
        eval_run_id: int,
        task_name: str,
        status: str,
        level: Optional[int] = None,
        problem_id: Optional[int] = None,
        op_name: str = "",
        iterations_used: int = 0,
        wall_time_s: Optional[float] = None,
        speedup: Optional[float] = None,
        failure_reason: str = "",
        generated_code: str = "",
        extra: Optional[dict] = None,
    ) -> int:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO task_result
                   (eval_run_id, task_name, status, level, problem_id,
                    op_name, iterations_used, wall_time_s, speedup,
                    failure_reason, generated_code, extra)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (eval_run_id, task_name, status, level, problem_id,
                 op_name, iterations_used, wall_time_s, speedup,
                 failure_reason, generated_code,
                 json.dumps(extra) if extra else None),
            )
            return cur.lastrowid

    def get_task_results(
        self,
        eval_run_id: Optional[int] = None,
        status: Optional[str] = None,
        level: Optional[int] = None,
        limit: int = 500,
    ) -> List[dict]:
        with self._cursor() as cur:
            conditions, params = [], []
            if eval_run_id is not None:
                conditions.append("eval_run_id = ?")
                params.append(eval_run_id)
            if status is not None:
                conditions.append("status = ?")
                params.append(status)
            if level is not None:
                conditions.append("level = ?")
                params.append(level)
            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            params.append(limit)
            cur.execute(
                f"SELECT * FROM task_result {where} ORDER BY id LIMIT ?", params
            )
            return [dict(r) for r in cur.fetchall()]

    # =========================================================================
    # Compile Traces
    # =========================================================================

    def add_compile_trace(
        self,
        task_result_id: int,
        pass_name: str,
        attempt: int = 1,
        success: bool = False,
        compiler_output: str = "",
        error_category: str = "",
        generated_code: str = "",
        fix_applied: str = "",
        wall_time_s: Optional[float] = None,
    ) -> int:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO compile_trace
                   (task_result_id, pass_name, attempt, success,
                    compiler_output, error_category, generated_code,
                    fix_applied, wall_time_s)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (task_result_id, pass_name, attempt, int(success),
                 compiler_output, error_category, generated_code,
                 fix_applied, wall_time_s),
            )
            return cur.lastrowid

    def get_compile_traces(
        self,
        task_result_id: Optional[int] = None,
        pass_name: Optional[str] = None,
        success: Optional[bool] = None,
        error_category: Optional[str] = None,
        limit: int = 500,
    ) -> List[dict]:
        with self._cursor() as cur:
            conditions, params = [], []
            if task_result_id is not None:
                conditions.append("task_result_id = ?")
                params.append(task_result_id)
            if pass_name is not None:
                conditions.append("pass_name = ?")
                params.append(pass_name)
            if success is not None:
                conditions.append("success = ?")
                params.append(int(success))
            if error_category is not None:
                conditions.append("error_category = ?")
                params.append(error_category)
            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            params.append(limit)
            cur.execute(
                f"SELECT * FROM compile_trace {where} ORDER BY id LIMIT ?", params
            )
            return [dict(r) for r in cur.fetchall()]

    # =========================================================================
    # Analytics Queries (for the proposer)
    # =========================================================================

    def get_error_category_stats(
        self, harness_id: Optional[int] = None
    ) -> List[dict]:
        """Frequency of each error_category across all compile traces."""
        with self._cursor() as cur:
            if harness_id is not None:
                cur.execute(
                    """SELECT ct.error_category, COUNT(*) AS count,
                              SUM(ct.success) AS fixed
                       FROM compile_trace ct
                       JOIN task_result tr ON ct.task_result_id = tr.id
                       JOIN eval_run er ON tr.eval_run_id = er.id
                       WHERE er.harness_id = ? AND ct.error_category != ''
                       GROUP BY ct.error_category
                       ORDER BY count DESC""",
                    (harness_id,),
                )
            else:
                cur.execute(
                    """SELECT error_category, COUNT(*) AS count,
                              SUM(success) AS fixed
                       FROM compile_trace
                       WHERE error_category != ''
                       GROUP BY error_category
                       ORDER BY count DESC"""
                )
            return [dict(r) for r in cur.fetchall()]

    def get_pass_failure_rates(
        self, harness_id: Optional[int] = None
    ) -> List[dict]:
        """Compile failure rate per lowering pass."""
        with self._cursor() as cur:
            if harness_id is not None:
                cur.execute(
                    """SELECT ct.pass_name,
                              COUNT(*) AS total_attempts,
                              SUM(CASE WHEN ct.success = 0 THEN 1 ELSE 0 END) AS failures,
                              ROUND(1.0 * SUM(CASE WHEN ct.success = 0 THEN 1 ELSE 0 END) / COUNT(*), 4) AS fail_rate
                       FROM compile_trace ct
                       JOIN task_result tr ON ct.task_result_id = tr.id
                       JOIN eval_run er ON tr.eval_run_id = er.id
                       WHERE er.harness_id = ?
                       GROUP BY ct.pass_name
                       ORDER BY fail_rate DESC""",
                    (harness_id,),
                )
            else:
                cur.execute(
                    """SELECT pass_name,
                              COUNT(*) AS total_attempts,
                              SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failures,
                              ROUND(1.0 * SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) / COUNT(*), 4) AS fail_rate
                       FROM compile_trace
                       GROUP BY pass_name
                       ORDER BY fail_rate DESC"""
                )
            return [dict(r) for r in cur.fetchall()]

    def get_harness_rankings(self, run_type: str = "search") -> List[dict]:
        """Rank harnesses by compile_rate (primary) and accuracy_rate (secondary)."""
        with self._cursor() as cur:
            cur.execute(
                """SELECT h.id AS harness_id, h.name, h.description,
                          er.id AS eval_run_id,
                          er.compile_rate, er.accuracy_rate, er.avg_speedup,
                          er.total_tasks, er.compile_pass, er.compile_fail
                   FROM eval_run er
                   JOIN harness h ON er.harness_id = h.id
                   WHERE er.run_type = ? AND er.finished_at IS NOT NULL
                   ORDER BY er.compile_rate DESC, er.accuracy_rate DESC""",
                (run_type,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_task_failure_history(self, task_name: str) -> List[dict]:
        """All results for a single task across all harness versions."""
        with self._cursor() as cur:
            cur.execute(
                """SELECT tr.*, er.harness_id, h.name AS harness_name
                   FROM task_result tr
                   JOIN eval_run er ON tr.eval_run_id = er.id
                   JOIN harness h ON er.harness_id = h.id
                   WHERE tr.task_name = ?
                   ORDER BY tr.created_at""",
                (task_name,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_most_failed_tasks(self, limit: int = 20) -> List[dict]:
        """Tasks with the highest failure rate across all evaluations."""
        with self._cursor() as cur:
            cur.execute(
                """SELECT task_name, level, op_name,
                          COUNT(*) AS total_evals,
                          SUM(CASE WHEN status = 'compile_fail' THEN 1 ELSE 0 END) AS compile_fails,
                          ROUND(1.0 * SUM(CASE WHEN status = 'compile_fail' THEN 1 ELSE 0 END) / COUNT(*), 4) AS fail_rate
                   FROM task_result
                   GROUP BY task_name
                   HAVING total_evals > 0
                   ORDER BY fail_rate DESC, compile_fails DESC
                   LIMIT ?""",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_compile_traces_for_proposer(
        self,
        harness_id: Optional[int] = None,
        only_failures: bool = True,
        limit: int = 200,
    ) -> List[dict]:
        """Rich trace records formatted for the proposer agent to inspect."""
        with self._cursor() as cur:
            conditions = []
            params: list = []
            if harness_id is not None:
                conditions.append("er.harness_id = ?")
                params.append(harness_id)
            if only_failures:
                conditions.append("ct.success = 0")
            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            params.append(limit)
            cur.execute(
                f"""SELECT ct.pass_name, ct.attempt, ct.success,
                           ct.compiler_output, ct.error_category,
                           ct.generated_code, ct.fix_applied,
                           tr.task_name, tr.op_name, tr.level,
                           h.name AS harness_name, h.id AS harness_id
                    FROM compile_trace ct
                    JOIN task_result tr ON ct.task_result_id = tr.id
                    JOIN eval_run er ON tr.eval_run_id = er.id
                    JOIN harness h ON er.harness_id = h.id
                    {where}
                    ORDER BY ct.id DESC LIMIT ?""",
                params,
            )
            return [dict(r) for r in cur.fetchall()]

    # =========================================================================
    # Pareto Frontier
    # =========================================================================

    def compute_pareto_frontier(
        self,
        run_type: str = "search",
        objectives: Tuple[str, ...] = ("compile_rate", "accuracy_rate"),
    ) -> List[dict]:
        """Compute and cache the Pareto-optimal set of harnesses.

        A harness is Pareto-optimal if no other harness dominates it on all
        objectives simultaneously (higher is better for all objectives).
        """
        rankings = self.get_harness_rankings(run_type)
        if not rankings:
            return []

        def dominates(a: dict, b: dict) -> bool:
            dominated = False
            for obj in objectives:
                va = a.get(obj) if a.get(obj) is not None else -1
                vb = b.get(obj) if b.get(obj) is not None else -1
                if va < vb:
                    return False
                if va > vb:
                    dominated = True
            return dominated

        frontier = []
        for candidate in rankings:
            if not any(dominates(existing, candidate) for existing in frontier):
                frontier = [
                    f for f in frontier if not dominates(candidate, f)
                ]
                frontier.append(candidate)

        with self._cursor() as cur:
            cur.execute("DELETE FROM pareto_frontier")
            for f in frontier:
                cur.execute(
                    """INSERT INTO pareto_frontier
                       (harness_id, eval_run_id, compile_rate, accuracy_rate, avg_speedup)
                       VALUES (?, ?, ?, ?, ?)""",
                    (f["harness_id"], f["eval_run_id"],
                     f.get("compile_rate"), f.get("accuracy_rate"),
                     f.get("avg_speedup")),
                )
        return frontier

    def get_pareto_frontier(self) -> List[dict]:
        with self._cursor() as cur:
            cur.execute(
                """SELECT pf.*, h.name AS harness_name, h.description
                   FROM pareto_frontier pf
                   JOIN harness h ON pf.harness_id = h.id
                   ORDER BY pf.compile_rate DESC"""
            )
            return [dict(r) for r in cur.fetchall()]

    # =========================================================================
    # Export for Proposer
    # =========================================================================

    def export_search_context(self, last_n_iterations: int = 5) -> dict:
        """Build a structured summary the proposer agent can consume.

        Returns a dict with rankings, error patterns, pass failure rates,
        hardest tasks, and recent iteration history.
        """
        return {
            "harness_rankings": self.get_harness_rankings(),
            "pareto_frontier": self.get_pareto_frontier(),
            "error_category_stats": self.get_error_category_stats(),
            "pass_failure_rates": self.get_pass_failure_rates(),
            "most_failed_tasks": self.get_most_failed_tasks(),
            "recent_iterations": self.list_iterations()[-last_n_iterations:],
        }
