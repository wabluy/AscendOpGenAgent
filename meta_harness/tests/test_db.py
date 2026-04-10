"""Tests for MetaHarnessDB core operations."""
import json
import tempfile
from pathlib import Path

import pytest

from meta_harness.db import MetaHarnessDB


@pytest.fixture
def db(tmp_path):
    d = MetaHarnessDB(tmp_path / "test.db")
    yield d
    d.close()


class TestSchemaInit:
    def test_creates_database_file(self, tmp_path):
        db_path = tmp_path / "new.db"
        db = MetaHarnessDB(db_path)
        assert db_path.exists()
        db.close()

    def test_schema_version_stored(self, db):
        with db._cursor() as cur:
            cur.execute("SELECT value FROM meta WHERE key = 'schema_version'")
            assert cur.fetchone()["value"] == "1"

    def test_idempotent_init(self, tmp_path):
        db_path = tmp_path / "idempotent.db"
        db1 = MetaHarnessDB(db_path)
        db1.close()
        db2 = MetaHarnessDB(db_path)
        db2.close()


class TestSearchIteration:
    def test_create_and_list(self, db):
        id1 = db.create_iteration(iteration_num=0, proposer_model="test-model")
        id2 = db.create_iteration(iteration_num=1, proposer_model="test-model")
        iters = db.list_iterations()
        assert len(iters) == 2
        assert iters[0]["iteration_num"] == 0
        assert iters[1]["iteration_num"] == 1

    def test_finish_iteration(self, db):
        it_id = db.create_iteration(iteration_num=0)
        assert db.get_iteration(it_id)["finished_at"] is None
        db.finish_iteration(it_id)
        assert db.get_iteration(it_id)["finished_at"] is not None


class TestHarness:
    def test_create_and_get(self, db):
        hid = db.create_harness(name="baseline", description="initial")
        h = db.get_harness(hid)
        assert h["name"] == "baseline"
        assert h["description"] == "initial"
        assert h["parent_id"] is None

    def test_parent_child(self, db):
        parent = db.create_harness(name="v0")
        child = db.create_harness(name="v1", parent_id=parent)
        h = db.get_harness(child)
        assert h["parent_id"] == parent

    def test_lineage(self, db):
        h0 = db.create_harness(name="v0")
        h1 = db.create_harness(name="v1", parent_id=h0)
        h2 = db.create_harness(name="v2", parent_id=h1)
        lineage = db.get_harness_lineage(h2)
        assert [h["name"] for h in lineage] == ["v0", "v1", "v2"]

    def test_list_with_iteration_filter(self, db):
        it_id = db.create_iteration(iteration_num=0)
        db.create_harness(name="h1", iteration_id=it_id)
        db.create_harness(name="h2", iteration_id=it_id)
        db.create_harness(name="h3")
        assert len(db.list_harnesses(iteration_id=it_id)) == 2
        assert len(db.list_harnesses()) == 3


class TestHarnessFiles:
    def test_save_and_get(self, db):
        hid = db.create_harness(name="test")
        db.save_harness_file(hid, "SKILL.md", "# test content", "abc123")
        content = db.get_harness_file(hid, "SKILL.md")
        assert content == "# test content"

    def test_list_files(self, db):
        hid = db.create_harness(name="test")
        db.save_harness_file(hid, "a.md", "aaa")
        db.save_harness_file(hid, "b.md", "bbb")
        files = db.get_harness_files(hid)
        assert len(files) == 2
        assert files[0]["rel_path"] == "a.md"

    def test_restore_to_disk(self, db, tmp_path):
        hid = db.create_harness(name="test")
        db.save_harness_file(hid, "sub/dir/test.md", "restored content")
        target = tmp_path / "restore_target"
        db.restore_harness_files(hid, target)
        assert (target / "sub" / "dir" / "test.md").read_text() == "restored content"

    def test_diff_harnesses(self, db):
        h1 = db.create_harness(name="v1")
        h2 = db.create_harness(name="v2")
        db.save_harness_file(h1, "same.md", "same content", "aaa")
        db.save_harness_file(h2, "same.md", "same content", "aaa")
        db.save_harness_file(h1, "changed.md", "old", "bbb")
        db.save_harness_file(h2, "changed.md", "new", "ccc")
        db.save_harness_file(h1, "removed.md", "gone", "ddd")
        db.save_harness_file(h2, "added.md", "fresh", "eee")

        diffs = db.diff_harnesses(h1, h2)
        paths = {d["rel_path"]: d for d in diffs}
        assert "same.md" not in paths
        assert paths["changed.md"]["changed"] is True
        assert paths["removed.md"]["in_a"] is True
        assert paths["removed.md"]["in_b"] is False
        assert paths["added.md"]["in_a"] is False
        assert paths["added.md"]["in_b"] is True

    def test_save_from_dir(self, db, tmp_path):
        hid = db.create_harness(name="test")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "test.md").write_text("hello")
        (tmp_path / "test.py").write_text("print('hi')")
        db.save_harness_files_from_dir(hid, tmp_path)
        files = db.get_harness_files(hid)
        assert len(files) == 2


class TestEvalRun:
    def test_create_and_finish(self, db):
        hid = db.create_harness(name="test")
        run_id = db.create_eval_run(hid)
        db.add_task_result(run_id, "task_a", "compile_pass")
        db.add_task_result(run_id, "task_b", "compile_fail", failure_reason="error X")
        db.add_task_result(run_id, "task_c", "accuracy_pass", speedup=1.5)
        db.finish_eval_run(run_id)
        run = db.get_eval_run(run_id)
        assert run["total_tasks"] == 3
        assert run["compile_pass"] == 2  # compile_pass + accuracy_pass
        assert run["compile_fail"] == 1
        assert run["accuracy_pass"] == 1
        assert run["finished_at"] is not None

    def test_list_runs(self, db):
        h1 = db.create_harness(name="h1")
        h2 = db.create_harness(name="h2")
        db.create_eval_run(h1, run_type="search")
        db.create_eval_run(h2, run_type="test")
        assert len(db.list_eval_runs(run_type="search")) == 1
        assert len(db.list_eval_runs(harness_id=h1)) == 1


class TestCompileTrace:
    def test_add_and_query(self, db):
        hid = db.create_harness(name="test")
        run_id = db.create_eval_run(hid)
        tid = db.add_task_result(run_id, "task_a", "compile_fail")
        db.add_compile_trace(tid, "tiling_pass", attempt=1, success=False,
                             compiler_output="error: undeclared 'log'",
                             error_category="undeclared_identifier")
        db.add_compile_trace(tid, "tiling_pass", attempt=2, success=True,
                             fix_applied="replaced logf with AscendC::Log")

        traces = db.get_compile_traces(task_result_id=tid)
        assert len(traces) == 2
        assert traces[0]["success"] == 0
        assert traces[1]["success"] == 1

    def test_filter_by_pass(self, db):
        hid = db.create_harness(name="test")
        run_id = db.create_eval_run(hid)
        tid = db.add_task_result(run_id, "task_a", "compile_fail")
        db.add_compile_trace(tid, "tiling_pass", attempt=1, success=False)
        db.add_compile_trace(tid, "init_pass", attempt=1, success=True)

        assert len(db.get_compile_traces(pass_name="tiling_pass")) == 1
        assert len(db.get_compile_traces(pass_name="init_pass")) == 1

    def test_filter_by_error_category(self, db):
        hid = db.create_harness(name="test")
        run_id = db.create_eval_run(hid)
        tid = db.add_task_result(run_id, "task_a", "compile_fail")
        db.add_compile_trace(tid, "tiling_pass", error_category="type_cast")
        db.add_compile_trace(tid, "init_pass", error_category="undeclared_id")

        result = db.get_compile_traces(error_category="type_cast")
        assert len(result) == 1
        assert result[0]["pass_name"] == "tiling_pass"


class TestAnalytics:
    @pytest.fixture
    def populated_db(self, db):
        """Create a DB with realistic data for analytics tests."""
        h1 = db.create_harness(name="baseline")
        h2 = db.create_harness(name="improved", parent_id=h1)

        run1 = db.create_eval_run(h1, run_type="search")
        t1 = db.add_task_result(run1, "level2/81_Gemm_Swish", "compile_fail",
                                level=2, op_name="Gemm_Swish")
        db.add_compile_trace(t1, "tiling_pass", attempt=1, success=False,
                             error_category="type_cast", compiler_output="float cast error")
        db.add_compile_trace(t1, "tiling_pass", attempt=2, success=True,
                             fix_applied="removed static_cast")
        db.add_compile_trace(t1, "process_pass", attempt=1, success=False,
                             error_category="undeclared_id", compiler_output="use of undeclared 'log'")

        t2 = db.add_task_result(run1, "level2/76_Gemm_Add_ReLU", "compile_pass",
                                level=2, op_name="Gemm_Add_ReLU")
        db.finish_eval_run(run1)

        run2 = db.create_eval_run(h2, run_type="search")
        t3 = db.add_task_result(run2, "level2/81_Gemm_Swish", "compile_pass",
                                level=2, op_name="Gemm_Swish")
        t4 = db.add_task_result(run2, "level2/76_Gemm_Add_ReLU", "compile_pass",
                                level=2, op_name="Gemm_Add_ReLU")
        db.finish_eval_run(run2)

        return db

    def test_error_category_stats(self, populated_db):
        stats = populated_db.get_error_category_stats()
        cats = {s["error_category"]: s for s in stats}
        assert "type_cast" in cats
        assert "undeclared_id" in cats

    def test_pass_failure_rates(self, populated_db):
        rates = populated_db.get_pass_failure_rates()
        names = {r["pass_name"]: r for r in rates}
        assert "tiling_pass" in names
        assert "process_pass" in names

    def test_harness_rankings(self, populated_db):
        rankings = populated_db.get_harness_rankings()
        assert len(rankings) == 2
        assert rankings[0]["compile_rate"] >= rankings[1]["compile_rate"]

    def test_task_failure_history(self, populated_db):
        history = populated_db.get_task_failure_history("level2/81_Gemm_Swish")
        assert len(history) == 2
        assert history[0]["status"] == "compile_fail"
        assert history[1]["status"] == "compile_pass"

    def test_most_failed_tasks(self, populated_db):
        failed = populated_db.get_most_failed_tasks()
        assert len(failed) > 0
        assert failed[0]["task_name"] == "level2/81_Gemm_Swish"

    def test_compile_traces_for_proposer(self, populated_db):
        traces = populated_db.get_compile_traces_for_proposer(only_failures=True)
        assert len(traces) > 0
        assert all(t["success"] == 0 for t in traces)

    def test_pareto_frontier(self, populated_db):
        frontier = populated_db.compute_pareto_frontier()
        assert len(frontier) >= 1
        # improved harness (100% compile rate) should be on frontier
        assert any(f["name"] == "improved" for f in frontier)

    def test_export_search_context(self, populated_db):
        ctx = populated_db.export_search_context()
        assert "harness_rankings" in ctx
        assert "error_category_stats" in ctx
        assert "pass_failure_rates" in ctx
        assert "most_failed_tasks" in ctx


class TestSearchLoopIntegration:
    def test_full_flow(self, db, tmp_path):
        from meta_harness.runner import HarnessRunner, SearchLoop

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        skills_dir = workspace / "skills" / "ascendc" / "dsl-lowering"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("# Original DSL Lowering Skill")

        runner = HarnessRunner(
            db=db,
            workspace_root=workspace,
            harness_paths=["skills/ascendc/dsl-lowering/SKILL.md"],
        )

        tasks = [
            {"task_name": "level2/test_task_1", "level": 2, "op_name": "TestOp1"},
            {"task_name": "level2/test_task_2", "level": 2, "op_name": "TestOp2"},
        ]

        loop = SearchLoop(db=db, runner=runner, task_list=tasks)

        baseline_id = loop.register_baseline("baseline")
        files = db.get_harness_files(baseline_id)
        assert len(files) == 1
        assert files[0]["content"] == "# Original DSL Lowering Skill"

        def mock_evaluator(task):
            if "task_1" in task["task_name"]:
                return {
                    "task_name": task["task_name"],
                    "status": "compile_pass",
                    "level": task.get("level"),
                    "op_name": task.get("op_name"),
                    "compile_traces": [
                        {"pass_name": "tiling_pass", "attempt": 1, "success": True},
                    ],
                }
            else:
                return {
                    "task_name": task["task_name"],
                    "status": "compile_fail",
                    "level": task.get("level"),
                    "op_name": task.get("op_name"),
                    "failure_reason": "tiling struct missing field",
                    "compile_traces": [
                        {
                            "pass_name": "tiling_pass", "attempt": 1, "success": False,
                            "compiler_output": "error: no member named 'attr0'",
                            "error_category": "missing_tiling_field",
                        },
                    ],
                }

        baseline_run = loop.evaluate_baseline(baseline_id, evaluator_fn=mock_evaluator)
        run = db.get_eval_run(baseline_run)
        assert run["total_tasks"] == 2
        assert run["compile_pass"] == 1
        assert run["compile_fail"] == 1

        # Simulate iteration 1: propose improved harness
        run_ids = loop.run_iteration(
            iteration_num=1,
            proposed_harnesses=[{
                "name": "v1_fix_tiling",
                "parent_id": baseline_id,
                "description": "Added tiling struct field guidance",
                "files": {
                    "skills/ascendc/dsl-lowering/SKILL.md": "# Improved DSL Lowering\n## Added tiling fix guidance",
                },
            }],
            evaluator_fn=lambda task: {
                "task_name": task["task_name"],
                "status": "compile_pass",
                "level": task.get("level"),
                "op_name": task.get("op_name"),
                "compile_traces": [
                    {"pass_name": "tiling_pass", "attempt": 1, "success": True},
                ],
            },
            proposer_model="test-model",
            reasoning="Fixed tiling struct by adding field guidance to prompt",
        )

        assert len(run_ids) == 1
        improved_run = db.get_eval_run(run_ids[0])
        assert improved_run["compile_pass"] == 2
        assert improved_run["compile_rate"] == 1.0

        # Verify original file was restored
        assert (skills_dir / "SKILL.md").read_text() == "# Original DSL Lowering Skill"

        # Check rankings
        rankings = db.get_harness_rankings()
        assert rankings[0]["compile_rate"] == 1.0

        # Check Pareto frontier
        frontier = db.get_pareto_frontier()
        assert len(frontier) >= 1

        # Proposer context
        ctx = loop.get_proposer_context()
        assert ctx["harness_rankings"][0]["compile_rate"] == 1.0
