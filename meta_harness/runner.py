"""
Meta-Harness Runner — orchestrates the outer-loop search.

This module does NOT modify the existing pipeline.  It:
  1. Snapshots harness files (SKILL.md, references, etc.) into the DB.
  2. Applies a harness variant by writing files to the workspace.
  3. Invokes the existing lingxi-code pipeline (or a user-supplied evaluator).
  4. Collects results and compiler traces back into the DB.
  5. Restores the original harness files when done.

The *proposer* is external (a coding agent); the runner only manages
the evaluate → log → restore cycle.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from meta_harness.db import MetaHarnessDB


# Default paths relative to the workspace root that constitute a "harness"
DEFAULT_HARNESS_PATHS = [
    "skills/ascendc/dsl-lowering/SKILL.md",
    "skills/ascendc/dsl-lowering/references/error_correction/error_correction_examples.md",
    "skills/ascendc/dsl-baseline-generation/SKILL.md",
    "skills/ascendc/dsl-optimization/SKILL.md",
    "skills/ascendc/ascend-call-generation/SKILL.md",
    "skills/ascendc/functional-conversion/SKILL.md",
    "skills/ascendc/ascendc-evaluation/SKILL.md",
    "skills/ascendc/op-desc-generation/SKILL.md",
    "skills/ascendc/reference-generation/SKILL.md",
    "agents/lingxi-code.md",
]


class HarnessRunner:
    """Manages the evaluate → log cycle for one harness variant."""

    def __init__(
        self,
        db: MetaHarnessDB,
        workspace_root: str | Path,
        harness_paths: Optional[List[str]] = None,
    ):
        self.db = db
        self.workspace = Path(workspace_root)
        self.harness_paths = harness_paths or DEFAULT_HARNESS_PATHS
        self._backup_dir: Optional[Path] = None

    # -- harness snapshot / restore -------------------------------------------

    def snapshot_current_harness(self, harness_id: int):
        """Read current files from disk and store them under harness_id."""
        for rel in self.harness_paths:
            full = self.workspace / rel
            if full.exists():
                content = full.read_text(encoding="utf-8", errors="replace")
                import hashlib
                sha = hashlib.sha256(content.encode()).hexdigest()
                self.db.save_harness_file(harness_id, rel, content, sha)

        ref_dirs = [
            "skills/ascendc/dsl-lowering/references",
            "skills/ascendc/dsl-baseline-generation/references",
            "skills/ascendc/ascend-call-generation/references",
        ]
        for ref_dir in ref_dirs:
            full_dir = self.workspace / ref_dir
            if full_dir.exists():
                self.db.save_harness_files_from_dir(
                    harness_id, full_dir,
                    patterns=["**/*.md", "**/*.py", "**/*.cpp", "**/*.json"],
                )

    def backup_current_files(self):
        """Backup current harness files before applying a variant."""
        self._backup_dir = self.workspace / ".meta_harness_backup"
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        for rel in self.harness_paths:
            src = self.workspace / rel
            if src.exists():
                dst = self._backup_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    def restore_backup(self):
        """Restore harness files from backup."""
        if self._backup_dir and self._backup_dir.exists():
            for rel in self.harness_paths:
                backup = self._backup_dir / rel
                target = self.workspace / rel
                if backup.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup, target)
            shutil.rmtree(self._backup_dir, ignore_errors=True)
            self._backup_dir = None

    def apply_harness(self, harness_id: int):
        """Write harness files from DB to disk."""
        self.db.restore_harness_files(harness_id, self.workspace)

    # -- evaluation -----------------------------------------------------------

    def evaluate_harness(
        self,
        harness_id: int,
        task_list: List[Dict],
        run_type: str = "search",
        evaluator_fn: Optional[Callable] = None,
        extra: Optional[dict] = None,
    ) -> int:
        """Run the existing pipeline for each task, log results.

        Args:
            harness_id: which harness variant to evaluate
            task_list: list of dicts with keys: task_name, task_file, level, op_name
            run_type: "search" or "test"
            evaluator_fn: optional callable(task_dict) -> result_dict
                          if None, uses the default subprocess evaluator
            extra: additional metadata for the eval_run

        Returns:
            eval_run_id
        """
        eval_run_id = self.db.create_eval_run(
            harness_id=harness_id,
            run_type=run_type,
            extra=extra,
        )

        self.backup_current_files()
        try:
            self.apply_harness(harness_id)

            for task in task_list:
                if evaluator_fn:
                    result = evaluator_fn(task)
                else:
                    result = self._default_evaluate(task)

                task_result_id = self.db.add_task_result(
                    eval_run_id=eval_run_id,
                    task_name=result.get("task_name", task.get("task_name", "")),
                    status=result.get("status", "error"),
                    level=result.get("level", task.get("level")),
                    problem_id=result.get("problem_id", task.get("problem_id")),
                    op_name=result.get("op_name", task.get("op_name", "")),
                    iterations_used=result.get("iterations_used", 0),
                    wall_time_s=result.get("wall_time_s"),
                    speedup=result.get("speedup"),
                    failure_reason=result.get("failure_reason", ""),
                    generated_code=result.get("generated_code", ""),
                    extra=result.get("extra"),
                )

                for trace in result.get("compile_traces", []):
                    self.db.add_compile_trace(
                        task_result_id=task_result_id,
                        pass_name=trace.get("pass_name", "unknown"),
                        attempt=trace.get("attempt", 1),
                        success=trace.get("success", False),
                        compiler_output=trace.get("compiler_output", ""),
                        error_category=trace.get("error_category", ""),
                        generated_code=trace.get("generated_code", ""),
                        fix_applied=trace.get("fix_applied", ""),
                        wall_time_s=trace.get("wall_time_s"),
                    )

        finally:
            self.restore_backup()

        self.db.finish_eval_run(eval_run_id)
        return eval_run_id

    def _default_evaluate(self, task: dict) -> dict:
        """Placeholder evaluator — returns a minimal result.

        In production, this would invoke the lingxi-code pipeline.
        Override with evaluator_fn for real evaluations.
        """
        return {
            "task_name": task.get("task_name", ""),
            "status": "error",
            "failure_reason": "No evaluator configured. Pass evaluator_fn to evaluate_harness().",
            "iterations_used": 0,
            "compile_traces": [],
        }


class SearchLoop:
    """Orchestrates the full Meta-Harness search loop.

    The search loop itself is simple; all intelligence lives in the
    external proposer agent.  This class manages:
      - iteration tracking
      - baseline registration
      - evaluation dispatch
      - Pareto frontier updates
    """

    def __init__(
        self,
        db: MetaHarnessDB,
        runner: HarnessRunner,
        task_list: List[Dict],
    ):
        self.db = db
        self.runner = runner
        self.task_list = task_list

    def register_baseline(self, name: str = "baseline_v0", description: str = "") -> int:
        """Snapshot and evaluate the current (unmodified) harness as baseline."""
        harness_id = self.db.create_harness(
            name=name,
            description=description or "Initial baseline harness (current skills/agents as-is)",
        )
        self.runner.snapshot_current_harness(harness_id)
        return harness_id

    def evaluate_baseline(
        self,
        harness_id: int,
        evaluator_fn: Optional[Callable] = None,
    ) -> int:
        return self.runner.evaluate_harness(
            harness_id=harness_id,
            task_list=self.task_list,
            run_type="search",
            evaluator_fn=evaluator_fn,
        )

    def run_iteration(
        self,
        iteration_num: int,
        proposed_harnesses: List[Dict],
        evaluator_fn: Optional[Callable] = None,
        proposer_model: str = "",
        proposer_prompt: str = "",
        reasoning: str = "",
    ) -> List[int]:
        """Execute one search iteration.

        Args:
            iteration_num: sequential iteration number
            proposed_harnesses: list of dicts, each with:
                - name: str
                - description: str
                - parent_id: Optional[int]
                - files: dict mapping rel_path -> content
            evaluator_fn: callable(task) -> result
            proposer_model: model used for proposing
            proposer_prompt: prompt given to proposer
            reasoning: proposer's reasoning trace

        Returns:
            list of eval_run_ids
        """
        iter_id = self.db.create_iteration(
            iteration_num=iteration_num,
            proposer_model=proposer_model,
            proposer_prompt=proposer_prompt,
            reasoning=reasoning,
        )

        eval_run_ids = []
        for proposal in proposed_harnesses:
            harness_id = self.db.create_harness(
                name=proposal["name"],
                parent_id=proposal.get("parent_id"),
                iteration_id=iter_id,
                description=proposal.get("description", ""),
                extra=proposal.get("extra"),
            )

            for rel_path, content in proposal.get("files", {}).items():
                import hashlib
                sha = hashlib.sha256(content.encode()).hexdigest()
                self.db.save_harness_file(harness_id, rel_path, content, sha)

            run_id = self.runner.evaluate_harness(
                harness_id=harness_id,
                task_list=self.task_list,
                run_type="search",
                evaluator_fn=evaluator_fn,
            )
            eval_run_ids.append(run_id)

        self.db.finish_iteration(iter_id)
        self.db.compute_pareto_frontier()
        return eval_run_ids

    def get_proposer_context(self, last_n: int = 5) -> dict:
        """Build context for the proposer agent to inspect."""
        return self.db.export_search_context(last_n_iterations=last_n)
