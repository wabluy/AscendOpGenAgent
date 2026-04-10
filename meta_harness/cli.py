#!/usr/bin/env python3
"""
Meta-Harness CLI — query and manage the search database.

Usage:
    python -m meta_harness.cli <command> [options]

Commands:
    init            Initialize a new database
    status          Show search progress overview
    iterations      List search iterations
    harnesses       List harness variants
    rankings        Show harness rankings by compile rate
    pareto          Compute and show Pareto frontier
    errors          Show error category statistics
    passes          Show per-pass failure rates
    hardest         Show most-failed tasks
    traces          Show compile traces (for proposer inspection)
    history         Show a task's history across harnesses
    diff            Diff two harness file snapshots
    restore         Restore harness files to disk
    export-context  Export proposer context as JSON
    ingest-result   Ingest a lingxi-code pipeline result into the DB
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from meta_harness.db import MetaHarnessDB


def _get_db(args) -> MetaHarnessDB:
    return MetaHarnessDB(args.db)


def _print_table(rows: list[dict], columns: Optional[list[str]] = None, max_col_width: int = 60):
    """Simple aligned table printer."""
    if not rows:
        print("  (no results)")
        return
    if columns is None:
        columns = list(rows[0].keys())
    columns = [c for c in columns if c in rows[0]]

    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.4f}"
        s = str(v)
        if len(s) > max_col_width:
            return s[:max_col_width - 3] + "..."
        return s

    headers = columns
    data = [[fmt(row.get(c)) for c in columns] for row in rows]
    widths = [
        max(len(h), *(len(d[i]) for d in data))
        for i, h in enumerate(headers)
    ]
    sep = "  "
    header_line = sep.join(h.ljust(w) for h, w in zip(headers, widths))
    print(header_line)
    print(sep.join("-" * w for w in widths))
    for d in data:
        print(sep.join(v.ljust(w) for v, w in zip(d, widths)))


# -- commands -----------------------------------------------------------------

def cmd_init(args):
    db = _get_db(args)
    print(f"Database initialized: {db.db_path}")
    db.close()


def cmd_status(args):
    db = _get_db(args)
    iterations = db.list_iterations()
    harnesses = db.list_harnesses(limit=9999)
    runs = db.list_eval_runs(limit=9999)
    frontier = db.get_pareto_frontier()

    best_run = None
    for r in runs:
        if r.get("compile_rate") is not None:
            if best_run is None or (r["compile_rate"] or 0) > (best_run["compile_rate"] or 0):
                best_run = r

    print("=" * 60)
    print("  Meta-Harness Search Status")
    print("=" * 60)
    print(f"  Database:          {db.db_path}")
    print(f"  Search iterations: {len(iterations)}")
    print(f"  Harness variants:  {len(harnesses)}")
    print(f"  Evaluation runs:   {len(runs)}")
    print(f"  Pareto frontier:   {len(frontier)} harnesses")
    if best_run:
        print(f"  Best compile rate: {best_run['compile_rate']:.2%} "
              f"({best_run['compile_pass']}/{best_run['total_tasks']} tasks, "
              f"harness #{best_run['harness_id']})")
    print("=" * 60)
    db.close()


def cmd_iterations(args):
    db = _get_db(args)
    rows = db.list_iterations()
    _print_table(rows, [
        "id", "iteration_num", "proposer_model",
        "started_at", "finished_at",
    ])
    db.close()


def cmd_harnesses(args):
    db = _get_db(args)
    rows = db.list_harnesses(
        iteration_id=args.iteration,
        limit=args.limit,
    )
    _print_table(rows, [
        "id", "name", "parent_id", "iteration_id",
        "description", "created_at",
    ])
    db.close()


def cmd_rankings(args):
    db = _get_db(args)
    rows = db.get_harness_rankings(args.run_type)
    _print_table(rows, [
        "harness_id", "name", "compile_rate", "accuracy_rate",
        "avg_speedup", "total_tasks", "compile_pass", "compile_fail",
    ])
    db.close()


def cmd_pareto(args):
    db = _get_db(args)
    frontier = db.compute_pareto_frontier()
    print(f"Pareto frontier: {len(frontier)} harnesses")
    _print_table(frontier, [
        "harness_id", "harness_name", "compile_rate",
        "accuracy_rate", "avg_speedup",
    ])
    db.close()


def cmd_errors(args):
    db = _get_db(args)
    rows = db.get_error_category_stats(harness_id=args.harness)
    _print_table(rows, ["error_category", "count", "fixed"])
    db.close()


def cmd_passes(args):
    db = _get_db(args)
    rows = db.get_pass_failure_rates(harness_id=args.harness)
    _print_table(rows, ["pass_name", "total_attempts", "failures", "fail_rate"])
    db.close()


def cmd_hardest(args):
    db = _get_db(args)
    rows = db.get_most_failed_tasks(limit=args.limit)
    _print_table(rows, [
        "task_name", "level", "op_name",
        "total_evals", "compile_fails", "fail_rate",
    ])
    db.close()


def cmd_traces(args):
    db = _get_db(args)
    rows = db.get_compile_traces_for_proposer(
        harness_id=args.harness,
        only_failures=not args.all,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        _print_table(rows, [
            "harness_name", "task_name", "pass_name", "attempt",
            "success", "error_category", "compiler_output",
        ])
    db.close()


def cmd_history(args):
    db = _get_db(args)
    rows = db.get_task_failure_history(args.task_name)
    _print_table(rows, [
        "harness_name", "status", "iterations_used",
        "failure_reason", "created_at",
    ])
    db.close()


def cmd_diff(args):
    db = _get_db(args)
    diffs = db.diff_harnesses(args.a, args.b)
    if not diffs:
        print("No differences found.")
    else:
        for d in diffs:
            indicator = "M" if d["changed"] else ("+" if d["in_b"] and not d["in_a"] else "-")
            print(f"  {indicator} {d['rel_path']}")
        if args.verbose:
            for d in diffs:
                if d["changed"] or (d["in_b"] and not d["in_a"]):
                    content_a = db.get_harness_file(args.a, d["rel_path"]) or ""
                    content_b = db.get_harness_file(args.b, d["rel_path"]) or ""
                    print(f"\n--- harness #{args.a}: {d['rel_path']}")
                    print(f"+++ harness #{args.b}: {d['rel_path']}")
                    import difflib
                    diff_lines = difflib.unified_diff(
                        content_a.splitlines(keepends=True),
                        content_b.splitlines(keepends=True),
                        lineterm="",
                    )
                    for line in diff_lines:
                        print(line)
    db.close()


def cmd_restore(args):
    db = _get_db(args)
    db.restore_harness_files(args.harness_id, args.target_dir)
    files = db.get_harness_files(args.harness_id)
    print(f"Restored {len(files)} files to {args.target_dir}")
    db.close()


def cmd_export_context(args):
    db = _get_db(args)
    ctx = db.export_search_context(last_n_iterations=args.last_n)
    print(json.dumps(ctx, indent=2, ensure_ascii=False))
    db.close()


def cmd_ingest_result(args):
    """Ingest a completed lingxi-code pipeline result directory into the DB."""
    db = _get_db(args)
    result_dir = Path(args.result_dir)

    eval_run_id = args.eval_run
    if eval_run_id is None:
        print("Error: --eval-run is required", file=sys.stderr)
        sys.exit(1)

    summary_path = result_dir / "summary.json"
    if not summary_path.exists():
        # Try to find summary.json in subdirectories
        candidates = list(result_dir.rglob("summary.json"))
        if candidates:
            summary_path = candidates[0]
        else:
            print(f"Error: no summary.json found under {result_dir}", file=sys.stderr)
            sys.exit(1)

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    task_name = args.task_name or result_dir.name
    level = args.level
    op_name = args.op_name or ""

    success = summary.get("success", False)
    iterations = summary.get("iterations", 0)

    if success:
        status = "compile_pass"
        perf = summary.get("perf_data") or {}
        speedup = perf.get("speedup_vs_torch")
    else:
        status = "compile_fail"
        speedup = None

    failure_reason = summary.get("failure_reason", summary.get("last_error", ""))

    generated_code = ""
    gen_code_path = result_dir / "generated_code.py"
    if gen_code_path.exists():
        generated_code = gen_code_path.read_text(encoding="utf-8", errors="replace")

    task_id = db.add_task_result(
        eval_run_id=eval_run_id,
        task_name=task_name,
        status=status,
        level=level,
        op_name=op_name,
        iterations_used=iterations,
        speedup=speedup,
        failure_reason=failure_reason,
        generated_code=generated_code,
    )

    error_history = summary.get("error_history", [])
    for entry in error_history:
        it = entry.get("iteration", 0)
        err_msg = entry.get("error_message", "")
        err_type = entry.get("error_type", "")

        db.add_compile_trace(
            task_result_id=task_id,
            pass_name=f"iteration_{it}",
            attempt=it + 1,
            success=False,
            compiler_output=err_msg,
            error_category=err_type,
        )

    print(f"Ingested: task_result #{task_id} for '{task_name}' ({status})")
    db.close()


# -- main ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meta-harness",
        description="Meta-Harness: outer-loop harness optimization database CLI",
    )
    parser.add_argument(
        "--db", default="meta_harness.db",
        help="Path to SQLite database (default: meta_harness.db)",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", help="Initialize database")
    sub.add_parser("status", help="Show search progress overview")
    sub.add_parser("iterations", help="List search iterations")

    p = sub.add_parser("harnesses", help="List harness variants")
    p.add_argument("--iteration", type=int, default=None)
    p.add_argument("--limit", type=int, default=50)

    p = sub.add_parser("rankings", help="Show harness rankings")
    p.add_argument("--run-type", default="search")

    sub.add_parser("pareto", help="Compute and show Pareto frontier")

    p = sub.add_parser("errors", help="Error category statistics")
    p.add_argument("--harness", type=int, default=None)

    p = sub.add_parser("passes", help="Per-pass failure rates")
    p.add_argument("--harness", type=int, default=None)

    p = sub.add_parser("hardest", help="Most-failed tasks")
    p.add_argument("--limit", type=int, default=20)

    p = sub.add_parser("traces", help="Show compile traces")
    p.add_argument("--harness", type=int, default=None)
    p.add_argument("--all", action="store_true", help="Include successes")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("history", help="Task history across harnesses")
    p.add_argument("task_name", help="Task name to look up")

    p = sub.add_parser("diff", help="Diff two harness file snapshots")
    p.add_argument("a", type=int, help="Harness ID A")
    p.add_argument("b", type=int, help="Harness ID B")
    p.add_argument("-v", "--verbose", action="store_true")

    p = sub.add_parser("restore", help="Restore harness files to disk")
    p.add_argument("harness_id", type=int)
    p.add_argument("target_dir")

    p = sub.add_parser("export-context", help="Export proposer context JSON")
    p.add_argument("--last-n", type=int, default=5)

    p = sub.add_parser("ingest-result", help="Ingest a pipeline result directory")
    p.add_argument("result_dir", help="Path to result directory")
    p.add_argument("--eval-run", type=int, required=True)
    p.add_argument("--task-name", default=None)
    p.add_argument("--level", type=int, default=None)
    p.add_argument("--op-name", default=None)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    cmd_map = {
        "init": cmd_init,
        "status": cmd_status,
        "iterations": cmd_iterations,
        "harnesses": cmd_harnesses,
        "rankings": cmd_rankings,
        "pareto": cmd_pareto,
        "errors": cmd_errors,
        "passes": cmd_passes,
        "hardest": cmd_hardest,
        "traces": cmd_traces,
        "history": cmd_history,
        "diff": cmd_diff,
        "restore": cmd_restore,
        "export-context": cmd_export_context,
        "ingest-result": cmd_ingest_result,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
