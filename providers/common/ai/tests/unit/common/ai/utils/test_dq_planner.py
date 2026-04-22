# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from airflow.providers.common.ai.hooks.pydantic_ai import PydanticAIHook
from airflow.providers.common.ai.utils.dq_models import DQCheck, DQCheckGroup, DQCheckInput, DQPlan
from airflow.providers.common.ai.utils.dq_planner import SQLDQPlanner, _extract_table_names
from airflow.providers.common.ai.utils.sql_validation import SQLSafetyError
from airflow.providers.common.sql.hooks.sql import DbApiHook


def _make_checks(*check_names: str) -> list[DQCheckInput]:
    """Helper: build a minimal checks list from names."""
    return [DQCheckInput(name=n, description=f"check {n}") for n in check_names]


def _make_plan(*check_names: str) -> DQPlan:
    """Helper: build a minimal DQPlan with one group per check."""
    groups = [
        DQCheckGroup(
            group_id="numeric_aggregate",
            query=f"SELECT COUNT(*) AS {name}_count FROM t",
            checks=[
                DQCheck(
                    check_name=name,
                    metric_key=f"{name}_count",
                    group_id="numeric_aggregate",
                    validator_name="exact_check",
                    validator_args={"expected": 0},
                )
            ],
        )
        for name in check_names
    ]
    return DQPlan(groups=groups)


def _make_llm_hook(plan: DQPlan) -> MagicMock:
    """Helper: mock PydanticAIHook that returns *plan* from agent.run_sync."""
    mock_usage = MagicMock(requests=1, tool_calls=0, input_tokens=100, output_tokens=50, total_tokens=150)
    mock_result = MagicMock(spec=["output", "all_messages", "usage", "response"])
    mock_result.output = plan
    mock_result.all_messages.return_value = []
    mock_result.usage.return_value = mock_usage
    mock_result.response.model_name = "test-model"
    mock_agent = MagicMock(spec=["run_sync"])
    mock_agent.run_sync.return_value = mock_result
    mock_hook = MagicMock(spec=PydanticAIHook)
    mock_hook.create_agent.return_value = mock_agent
    return mock_hook


class TestSQLDQPlannerBuildSchema:
    def test_returns_manual_schema_context_verbatim(self):
        planner = SQLDQPlanner(llm_hook=MagicMock(spec=PydanticAIHook), db_hook=None)
        result = planner.build_schema_context(
            table_names=None,
            schema_context="Table: t\nColumns: id INT",
        )
        assert result == "Table: t\nColumns: id INT"

    def test_introspects_via_db_hook_when_no_manual_context(self):
        mock_db_hook = MagicMock(spec=DbApiHook)
        mock_db_hook.get_table_schema.return_value = [{"name": "id", "type": "INT"}]

        planner = SQLDQPlanner(llm_hook=MagicMock(spec=PydanticAIHook), db_hook=mock_db_hook)
        result = planner.build_schema_context(
            table_names=["customers"],
            schema_context=None,
        )

        mock_db_hook.get_table_schema.assert_called_once_with("customers")
        assert "customers" in result
        assert "id INT" in result

    def test_manual_context_takes_priority_over_db_hook(self):
        mock_db_hook = MagicMock(spec=DbApiHook)

        planner = SQLDQPlanner(llm_hook=MagicMock(spec=PydanticAIHook), db_hook=mock_db_hook)
        result = planner.build_schema_context(
            table_names=["t"],
            schema_context="manual override",
        )

        mock_db_hook.get_table_schema.assert_not_called()
        assert result == "manual override"

    def test_returns_empty_string_when_no_source(self):
        planner = SQLDQPlanner(llm_hook=MagicMock(spec=PydanticAIHook), db_hook=None)
        result = planner.build_schema_context(
            table_names=None,
            schema_context=None,
        )
        assert result == ""


class TestSQLDQPlannerGeneratePlan:
    def test_returns_plan_when_check_names_match(self):
        checks = _make_checks("null_emails", "dup_ids")
        plan = _make_plan("null_emails", "dup_ids")
        mock_hook = _make_llm_hook(plan)

        planner = SQLDQPlanner(llm_hook=mock_hook, db_hook=None)
        result = planner.generate_plan(checks, schema_context="")

        assert set(result.check_names) == {c.name for c in checks}

    def test_raises_when_llm_drops_a_check(self):
        checks = _make_checks("null_emails", "dup_ids")
        plan = _make_plan("null_emails")  # missing dup_ids
        mock_hook = _make_llm_hook(plan)

        planner = SQLDQPlanner(llm_hook=mock_hook, db_hook=None)
        with pytest.raises(ValueError, match="dup_ids"):
            planner.generate_plan(checks, schema_context="")

    def test_raises_when_llm_adds_extra_check(self):
        checks = _make_checks("null_emails")
        plan = _make_plan("null_emails", "hallucinated_check")  # unexpected extra
        mock_hook = _make_llm_hook(plan)

        planner = SQLDQPlanner(llm_hook=mock_hook, db_hook=None)
        with pytest.raises(ValueError, match="hallucinated_check"):
            planner.generate_plan(checks, schema_context="")

    def test_agent_receives_schema_context_in_prompt(self):
        checks = _make_checks("row_count")
        plan = _make_plan("row_count")
        mock_hook = _make_llm_hook(plan)

        planner = SQLDQPlanner(llm_hook=mock_hook, db_hook=None)
        planner.generate_plan(checks, schema_context="Table: orders\nColumns: id INT")

        call_kwargs = mock_hook.create_agent.call_args
        instructions = call_kwargs.kwargs["instructions"]
        assert "orders" in instructions

    def test_extra_system_prompt_appended_to_instructions(self):
        checks = _make_checks("row_count")
        plan = _make_plan("row_count")
        mock_hook = _make_llm_hook(plan)

        planner = SQLDQPlanner(
            llm_hook=mock_hook, db_hook=None, system_prompt="Always use lowercase aliases."
        )
        planner.generate_plan(checks, schema_context="")

        call_kwargs = mock_hook.create_agent.call_args
        instructions = call_kwargs.kwargs["instructions"]
        assert "Always use lowercase aliases." in instructions
        # Built-in planning prompt must still be present.
        assert "DQPlan" in instructions

    def test_empty_system_prompt_not_appended(self):
        """When system_prompt is empty (default), instructions must not contain 'Additional instructions'."""
        checks = _make_checks("row_count")
        plan = _make_plan("row_count")
        mock_hook = _make_llm_hook(plan)

        planner = SQLDQPlanner(llm_hook=mock_hook, db_hook=None)
        planner.generate_plan(checks, schema_context="")

        call_kwargs = mock_hook.create_agent.call_args
        instructions = call_kwargs.kwargs["instructions"]
        assert "Additional instructions" not in instructions

    def test_agent_params_forwarded_to_create_agent(self):
        checks = _make_checks("row_count")
        plan = _make_plan("row_count")
        mock_hook = _make_llm_hook(plan)

        planner = SQLDQPlanner(llm_hook=mock_hook, db_hook=None, agent_params={"retries": 3})
        planner.generate_plan(checks, schema_context="")

        call_kwargs = mock_hook.create_agent.call_args
        assert call_kwargs.kwargs.get("retries") == 3

    def test_agent_params_empty_by_default(self):
        checks = _make_checks("row_count")
        plan = _make_plan("row_count")
        mock_hook = _make_llm_hook(plan)

        planner = SQLDQPlanner(llm_hook=mock_hook, db_hook=None)
        planner.generate_plan(checks, schema_context="")

        call_kwargs = mock_hook.create_agent.call_args
        # Only output_type and instructions should be present — no extra kwargs.
        extra = {k: v for k, v in call_kwargs.kwargs.items() if k not in ("output_type", "instructions")}
        assert extra == {}

    def test_prompt_bodies_logged_at_debug_only(self, caplog):
        checks = _make_checks("row_count")
        plan = _make_plan("row_count")
        mock_hook = _make_llm_hook(plan)

        planner = SQLDQPlanner(llm_hook=mock_hook, db_hook=None)

        with caplog.at_level("INFO"):
            planner.generate_plan(checks, schema_context="Table: orders\nColumns: id INT")

        assert "Generating DQ plan with" in caplog.text
        assert "Using system prompt:" not in caplog.text
        assert "Using user message:" not in caplog.text

        caplog.clear()
        with caplog.at_level("DEBUG"):
            planner.generate_plan(checks, schema_context="Table: orders\nColumns: id INT")

        assert "Using system prompt:" in caplog.text
        assert "Using user message:" in caplog.text


class TestSQLDQPlannerExecutePlan:
    def test_returns_flat_check_name_to_value_map(self):
        mock_db_hook = MagicMock(spec=DbApiHook)
        mock_db_hook.get_records.return_value = [{"null_email_count": 5}]

        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="null_check",
                    query="SELECT COUNT(*) AS null_email_count FROM t WHERE email IS NULL",
                    checks=[
                        DQCheck(
                            check_name="null_emails",
                            metric_key="null_email_count",
                            group_id="null_check",
                        )
                    ],
                )
            ]
        )
        planner = SQLDQPlanner(llm_hook=MagicMock(spec=PydanticAIHook), db_hook=mock_db_hook)
        results = planner.execute_plan(plan)

        assert results["null_emails"] == 5

    def test_raises_safety_error_for_unsafe_sql(self):
        mock_db_hook = MagicMock(spec=DbApiHook)
        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="g",
                    query="DROP TABLE customers",
                    checks=[DQCheck(check_name="c", metric_key="c_val", group_id="g")],
                )
            ]
        )
        planner = SQLDQPlanner(llm_hook=MagicMock(spec=PydanticAIHook), db_hook=mock_db_hook)
        with pytest.raises(SQLSafetyError):
            planner.execute_plan(plan)

        mock_db_hook.get_records.assert_not_called()

    def test_raises_when_metric_key_missing_from_result(self):
        mock_db_hook = MagicMock(spec=DbApiHook)
        mock_db_hook.get_records.return_value = [{"wrong_column": 0}]

        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="g",
                    query="SELECT 0 AS wrong_column FROM t",
                    checks=[DQCheck(check_name="c", metric_key="expected_column", group_id="g")],
                )
            ]
        )
        planner = SQLDQPlanner(llm_hook=MagicMock(spec=PydanticAIHook), db_hook=mock_db_hook)
        with pytest.raises(ValueError, match="expected_column"):
            planner.execute_plan(plan)

    def test_handles_tuple_rows_from_get_records(self):
        """get_records may return plain tuples; ensure positional mapping works."""
        mock_db_hook = MagicMock(spec=DbApiHook)
        mock_db_hook.get_records.return_value = [(42,)]

        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="g",
                    query="SELECT COUNT(*) AS row_count FROM t",
                    checks=[DQCheck(check_name="rows", metric_key="row_count", group_id="g")],
                )
            ]
        )
        planner = SQLDQPlanner(llm_hook=MagicMock(spec=PydanticAIHook), db_hook=mock_db_hook)
        results = planner.execute_plan(plan)
        assert results["rows"] == 42

    def test_raises_when_tuple_length_does_not_match_metric_keys(self):
        """Tuple-shaped rows must match the number of expected metric keys."""
        mock_db_hook = MagicMock(spec=DbApiHook)
        mock_db_hook.get_records.return_value = [(1, 2)]

        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="g",
                    query="SELECT COUNT(*) AS row_count FROM t",
                    checks=[DQCheck(check_name="rows", metric_key="row_count", group_id="g")],
                )
            ]
        )

        planner = SQLDQPlanner(llm_hook=MagicMock(spec=PydanticAIHook), db_hook=mock_db_hook)
        with pytest.raises(ValueError, match=r"returned 2 value\(s\)"):
            planner.execute_plan(plan)

    def test_raises_for_unsupported_row_type(self):
        """Unexpected row types should fail fast with a clear error."""
        mock_db_hook = MagicMock(spec=DbApiHook)
        mock_db_hook.get_records.return_value = [1]

        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="g",
                    query="SELECT COUNT(*) AS row_count FROM t",
                    checks=[DQCheck(check_name="rows", metric_key="row_count", group_id="g")],
                )
            ]
        )

        planner = SQLDQPlanner(llm_hook=MagicMock(spec=PydanticAIHook), db_hook=mock_db_hook)
        with pytest.raises(ValueError, match="Unsupported row type"):
            planner.execute_plan(plan)

    def test_raises_when_result_column_order_differs_from_check_order_for_dict_rows(self):
        """Result columns must follow the same order as checks in the plan group."""
        mock_db_hook = MagicMock(spec=DbApiHook)
        mock_db_hook.get_records.return_value = [{"second_metric": 2, "first_metric": 1}]

        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="g",
                    query="SELECT 1 AS first_metric, 2 AS second_metric FROM t",
                    checks=[
                        DQCheck(check_name="first", metric_key="first_metric", group_id="g"),
                        DQCheck(check_name="second", metric_key="second_metric", group_id="g"),
                    ],
                )
            ]
        )

        planner = SQLDQPlanner(llm_hook=MagicMock(spec=PydanticAIHook), db_hook=mock_db_hook)
        with pytest.raises(ValueError, match="unexpected order"):
            planner.execute_plan(plan)

    def test_raises_when_result_column_order_differs_from_check_order_for_tuple_rows(self):
        """When last_description is available, column order must match check order."""
        mock_db_hook = MagicMock(spec=DbApiHook)
        mock_db_hook.get_records.return_value = [(1, 2)]
        mock_db_hook.last_description = [("second_metric",), ("first_metric",)]

        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="g",
                    query="SELECT 1 AS first_metric, 2 AS second_metric FROM t",
                    checks=[
                        DQCheck(check_name="first", metric_key="first_metric", group_id="g"),
                        DQCheck(check_name="second", metric_key="second_metric", group_id="g"),
                    ],
                )
            ]
        )

        planner = SQLDQPlanner(llm_hook=MagicMock(spec=PydanticAIHook), db_hook=mock_db_hook)
        with pytest.raises(ValueError, match="unexpected order"):
            planner.execute_plan(plan)


class TestSQLDQPlannerExecutionBackends:
    """Tests for backend selection and DataFusion execution path."""

    def _simple_plan(self, metric_key: str = "null_count") -> DQPlan:
        return DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="g",
                    query=f"SELECT COUNT(*) AS {metric_key} FROM sales",
                    checks=[DQCheck(check_name="null_check", metric_key=metric_key, group_id="g")],
                )
            ]
        )

    def test_raises_when_neither_db_hook_nor_datasource(self):
        """Both db_hook and datasource_config are absent — must raise ValueError."""
        plan = _make_plan("some_check")
        planner = SQLDQPlanner(llm_hook=MagicMock(spec=PydanticAIHook), db_hook=None)
        with pytest.raises(ValueError, match="db_conn_id|datasource_config"):
            planner.execute_plan(plan)

    @patch("airflow.providers.common.ai.utils.dq_planner.SQLDQPlanner._build_datafusion_engine")
    def test_datafusion_engine_built_when_no_db_hook(self, mock_build_engine):
        """_build_datafusion_engine is called when db_hook is None and datasource_config is set."""
        mock_engine = MagicMock()
        mock_engine.execute_query.return_value = {"null_count": [0]}
        mock_build_engine.return_value = mock_engine

        mock_datasource = MagicMock()
        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=None,
            datasource_config=mock_datasource,
        )
        planner.execute_plan(self._simple_plan())

        mock_build_engine.assert_called_once()
        mock_engine.execute_query.assert_called_once()

    @patch("airflow.providers.common.ai.utils.dq_planner.SQLDQPlanner._build_datafusion_engine")
    def test_datafusion_result_first_value_used(self, mock_build_engine):
        """Column-oriented result {col: [val, ...]} — first element is used as the metric value."""
        mock_engine = MagicMock()
        mock_engine.execute_query.return_value = {"null_count": [7, 99]}  # only first value counts
        mock_build_engine.return_value = mock_engine

        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=None,
            datasource_config=MagicMock(),
        )
        results = planner.execute_plan(self._simple_plan("null_count"))
        assert results["null_check"] == 7

    @patch("airflow.providers.common.ai.utils.dq_planner.SQLDQPlanner._build_datafusion_engine")
    def test_datafusion_empty_result_raises_metric_missing(self, mock_build_engine):
        """An empty DataFusion result triggers the metric-key-missing ValueError."""
        mock_engine = MagicMock()
        mock_engine.execute_query.return_value = {}  # no columns at all
        mock_build_engine.return_value = mock_engine

        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=None,
            datasource_config=MagicMock(),
        )
        with pytest.raises(ValueError, match="null_count"):
            planner.execute_plan(self._simple_plan("null_count"))

    @patch("airflow.providers.common.ai.utils.dq_planner.SQLDQPlanner._build_datafusion_engine")
    def test_datafusion_raises_when_result_column_order_differs_from_check_order(self, mock_build_engine):
        """DataFusion result column order must align with plan check order."""
        mock_engine = MagicMock()
        mock_engine.execute_query.return_value = {"second_metric": [2], "first_metric": [1]}
        mock_build_engine.return_value = mock_engine

        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="g",
                    query="SELECT 1 AS first_metric, 2 AS second_metric FROM t",
                    checks=[
                        DQCheck(check_name="first", metric_key="first_metric", group_id="g"),
                        DQCheck(check_name="second", metric_key="second_metric", group_id="g"),
                    ],
                )
            ]
        )

        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=None,
            datasource_config=MagicMock(),
        )
        with pytest.raises(ValueError, match="unexpected order"):
            planner.execute_plan(plan)

    @patch("airflow.providers.common.ai.utils.dq_planner.SQLDQPlanner._build_datafusion_engine")
    def test_datafusion_engine_built_once_for_multiple_groups(self, mock_build_engine):
        """DataFusion engine is instantiated once even when the plan has multiple groups."""
        mock_engine = MagicMock()
        mock_engine.execute_query.side_effect = [
            {"a_count": [3]},
            {"b_count": [0]},
        ]
        mock_build_engine.return_value = mock_engine

        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="g1",
                    query="SELECT COUNT(*) AS a_count FROM t",
                    checks=[DQCheck(check_name="check_a", metric_key="a_count", group_id="g1")],
                ),
                DQCheckGroup(
                    group_id="g2",
                    query="SELECT COUNT(*) AS b_count FROM t",
                    checks=[DQCheck(check_name="check_b", metric_key="b_count", group_id="g2")],
                ),
            ]
        )
        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=None,
            datasource_config=MagicMock(),
        )
        results = planner.execute_plan(plan)

        mock_build_engine.assert_called_once()
        assert results == {"check_a": 3, "check_b": 0}

    def test_db_path_preferred_when_both_db_hook_and_datasource_set(self):
        """When both db_hook and datasource_config are set, the DB path is used."""
        mock_db_hook = MagicMock(spec=DbApiHook)
        mock_db_hook.get_records.return_value = [{"null_count": 5}]

        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=mock_db_hook,
            datasource_config=MagicMock(),
        )
        results = planner.execute_plan(self._simple_plan("null_count"))
        mock_db_hook.get_records.assert_called_once()
        assert results["null_check"] == 5


class TestSQLDQPlannerDialect:
    @pytest.mark.parametrize(
        ("case", "dialect_arg", "expected"),
        [
            ("explicit", "postgres", "postgres"),
            ("auto", None, "postgres"),
            ("none", None, None),
        ],
    )
    def test_dialect_resolution(self, case, dialect_arg, expected):
        if case == "explicit":
            db_hook = MagicMock(spec=[])  # no dialect_name attribute
        elif case == "auto":
            db_hook = MagicMock()
            db_hook.dialect_name = "postgresql"
        else:
            db_hook = None

        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=db_hook,
            dialect=dialect_arg,
        )
        assert planner._dialect == expected

    @patch("airflow.providers.common.ai.utils.dq_planner._validate_sql")
    def test_dialect_passed_to_validate_sql(self, mock_validate):
        mock_validate.return_value = []
        mock_db_hook = MagicMock(spec=DbApiHook)
        mock_db_hook.get_records.return_value = [{"c": 1}]

        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="g",
                    query="SELECT 1 AS c FROM t",
                    checks=[DQCheck(check_name="x", metric_key="c", group_id="g")],
                )
            ]
        )
        planner = SQLDQPlanner(llm_hook=MagicMock(spec=PydanticAIHook), db_hook=mock_db_hook, dialect="mysql")
        planner.execute_plan(plan)

        mock_validate.assert_called_once()
        _, kwargs = mock_validate.call_args
        assert kwargs.get("dialect") == "mysql"


def _single_group_plan(query: str = "SELECT 1 AS c FROM t") -> DQPlan:
    return DQPlan(
        groups=[
            DQCheckGroup(
                group_id="g",
                query=query,
                checks=[DQCheck(check_name="x", metric_key="c", group_id="g")],
            )
        ]
    )


def _make_agent_returning(plan: DQPlan) -> MagicMock:
    """Mock pydantic-ai agent whose run_sync always returns *plan*."""
    mock_result = MagicMock(spec=["output", "all_messages"])
    mock_result.output = plan
    mock_result.all_messages.return_value = []
    mock_agent = MagicMock(spec=["run_sync"])
    mock_agent.run_sync.return_value = mock_result
    return mock_agent


class TestSQLDQPlannerSQLRetry:
    def _planner_with_agent(self, plan_agent, db_hook=None, max_sql_retries=2) -> SQLDQPlanner:
        """Return a planner with _plan_agent already set (simulates post-generate_plan state)."""
        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=db_hook,
            max_sql_retries=max_sql_retries,
        )
        planner._plan_agent = plan_agent
        planner._plan_all_messages = []
        return planner

    def test_valid_sql_passes_without_retry(self):
        """A query that passes validation is returned immediately unchanged."""
        group = DQCheckGroup(
            group_id="g",
            query="SELECT COUNT(*) AS c FROM t",
            checks=[DQCheck(check_name="x", metric_key="c", group_id="g")],
        )
        planner = self._planner_with_agent(MagicMock())
        result = planner._validate_or_fix_group(group)

        assert result.query == group.query
        planner._plan_agent.run_sync.assert_not_called()

    def test_first_retry_fixes_sql(self):
        """LLM returns a valid query on the first correction attempt."""
        bad_plan = _single_group_plan("DROP TABLE t")
        good_plan = _single_group_plan("SELECT COUNT(*) AS c FROM t")

        mock_agent = _make_agent_returning(good_plan)
        mock_db_hook = MagicMock()
        mock_db_hook.get_records.return_value = [{"c": 1}]

        planner = self._planner_with_agent(mock_agent, db_hook=mock_db_hook)
        results = planner.execute_plan(bad_plan)

        assert results["x"] == 1
        mock_agent.run_sync.assert_called_once()
        # fix_prompt must name the failing group and include the error
        fix_prompt = mock_agent.run_sync.call_args.args[0]
        assert "g" in fix_prompt
        assert (
            "DROP" in fix_prompt
            or "not allowed" in fix_prompt.lower()
            or "SafetyError" in fix_prompt
            or "safety" in fix_prompt.lower()
        )

    def test_second_retry_fixes_sql_when_first_still_fails(self):
        """Planner retries twice; first correction still fails, second succeeds."""
        bad_plan = _single_group_plan("DELETE FROM t")
        still_bad_plan = _single_group_plan("TRUNCATE t")
        good_plan = _single_group_plan("SELECT 1 AS c FROM t")

        # agent returns a new plan on each call
        call_count = 0
        plans = [still_bad_plan, good_plan]

        def side_effect(*args, **kwargs):
            nonlocal call_count
            result = MagicMock()
            result.output = plans[call_count]
            result.all_messages.return_value = []
            call_count += 1
            return result

        mock_agent = MagicMock()
        mock_agent.run_sync.side_effect = side_effect

        mock_db_hook = MagicMock()
        mock_db_hook.get_records.return_value = [{"c": 1}]

        planner = self._planner_with_agent(mock_agent, db_hook=mock_db_hook, max_sql_retries=2)
        results = planner.execute_plan(bad_plan)

        assert results["x"] == 1
        assert mock_agent.run_sync.call_count == 2

    def test_raises_after_all_retries_exhausted(self):
        """SQLSafetyError is raised when every correction attempt also fails."""
        bad_plan = _single_group_plan("DROP TABLE t")
        still_bad_plan = _single_group_plan("DELETE FROM t")

        mock_agent = _make_agent_returning(still_bad_plan)
        mock_db_hook = MagicMock()

        planner = self._planner_with_agent(mock_agent, db_hook=mock_db_hook, max_sql_retries=2)
        with pytest.raises(SQLSafetyError, match="could not be corrected after 2 attempt"):
            planner.execute_plan(bad_plan)

        assert mock_agent.run_sync.call_count == 2

    def test_max_sql_retries_zero_raises_immediately(self):
        """With max_sql_retries=0 the original error is re-raised without wrapping."""
        bad_plan = _single_group_plan("DROP TABLE t")
        mock_db_hook = MagicMock()

        planner = self._planner_with_agent(MagicMock(), db_hook=mock_db_hook, max_sql_retries=0)
        with pytest.raises(SQLSafetyError, match="not allowed") as exc_info:
            planner.execute_plan(bad_plan)

        assert "attempt" not in str(exc_info.value).lower()
        planner._plan_agent.run_sync.assert_not_called()

    def test_no_plan_agent_re_raises_original_error(self):
        """If execute_plan is called without a prior generate_plan, the error propagates."""
        bad_plan = _single_group_plan("DROP TABLE t")
        mock_db_hook = MagicMock()

        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook), db_hook=mock_db_hook, max_sql_retries=2
        )
        # _plan_agent is None — no generate_plan was called
        with pytest.raises(SQLSafetyError):
            planner.execute_plan(bad_plan)

    def test_message_history_updated_on_each_retry(self):
        """_plan_all_messages is refreshed from result.all_messages() after each retry."""
        bad_plan = _single_group_plan("DROP TABLE t")
        good_plan = _single_group_plan("SELECT 1 AS c FROM t")

        sentinel_messages = [object()]  # unique object to verify update

        mock_result = MagicMock()
        mock_result.output = good_plan
        mock_result.all_messages.return_value = sentinel_messages
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        mock_db_hook = MagicMock()
        mock_db_hook.get_records.return_value = [{"c": 1}]

        planner = self._planner_with_agent(mock_agent, db_hook=mock_db_hook)
        planner.execute_plan(bad_plan)

        assert planner._plan_all_messages is sentinel_messages

    def test_retry_skips_when_corrected_plan_missing_group(self):
        """If the LLM omits the target group_id, the loop continues to the next attempt."""
        bad_plan = _single_group_plan("DROP TABLE t")
        # first response has a different group_id
        wrong_group_plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="OTHER_GROUP",
                    query="SELECT 1 AS c FROM t",
                    checks=[DQCheck(check_name="x", metric_key="c", group_id="OTHER_GROUP")],
                )
            ]
        )
        good_plan = _single_group_plan("SELECT 1 AS c FROM t")

        plans = [wrong_group_plan, good_plan]
        call_idx = 0

        def side_effect(*args, **kwargs):
            nonlocal call_idx
            result = MagicMock()
            result.output = plans[call_idx]
            result.all_messages.return_value = []
            call_idx += 1
            return result

        mock_agent = MagicMock()
        mock_agent.run_sync.side_effect = side_effect

        mock_db_hook = MagicMock()
        mock_db_hook.get_records.return_value = [{"c": 1}]

        planner = self._planner_with_agent(mock_agent, db_hook=mock_db_hook, max_sql_retries=2)
        results = planner.execute_plan(bad_plan)

        assert results["x"] == 1
        assert mock_agent.run_sync.call_count == 2


class TestSQLDQPlannerGroupSizes:
    """Tests for _validate_group_sizes — warns but does not fail for oversize groups."""

    def test_no_warning_when_groups_within_limit(self, caplog):
        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="g",
                    query="SELECT 1 AS a, 2 AS b FROM t",
                    checks=[DQCheck(check_name=f"c{i}", metric_key=f"m{i}", group_id="g") for i in range(5)],
                )
            ]
        )
        with caplog.at_level("WARNING"):
            SQLDQPlanner._validate_group_sizes(plan)
        assert "checks (max recommended" not in caplog.text

    def test_warns_when_group_exceeds_limit(self, caplog):
        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="big_group",
                    query="SELECT 1 FROM t",
                    checks=[
                        DQCheck(check_name=f"c{i}", metric_key=f"m{i}", group_id="big_group")
                        for i in range(8)
                    ],
                )
            ]
        )
        with caplog.at_level("WARNING"):
            SQLDQPlanner._validate_group_sizes(plan)
        assert "big_group" in caplog.text
        assert "8 checks" in caplog.text


class TestSQLDQPlannerCollectUnexpected:
    """Tests for the unexpected-value collection feature."""

    def test_collect_unexpected_adds_prompt_section(self):
        """When collect_unexpected=True, the system prompt includes UNEXPECTED VALUE COLLECTION."""
        checks = _make_checks("phone_fmt")
        plan = _make_plan("phone_fmt")
        mock_hook = _make_llm_hook(plan)

        planner = SQLDQPlanner(
            llm_hook=mock_hook, db_hook=None, collect_unexpected=True, unexpected_sample_size=50
        )
        planner.generate_plan(checks, schema_context="")

        call_kwargs = mock_hook.create_agent.call_args
        instructions = call_kwargs.kwargs["instructions"]
        assert "UNEXPECTED VALUE COLLECTION" in instructions
        assert "LIMIT 50" in instructions

    def test_collect_unexpected_false_no_prompt_section(self):
        """When collect_unexpected=False (default), the prompt does NOT include unexpected section."""
        checks = _make_checks("row_count")
        plan = _make_plan("row_count")
        mock_hook = _make_llm_hook(plan)

        planner = SQLDQPlanner(llm_hook=mock_hook, db_hook=None, collect_unexpected=False)
        planner.generate_plan(checks, schema_context="")

        call_kwargs = mock_hook.create_agent.call_args
        instructions = call_kwargs.kwargs["instructions"]
        assert "UNEXPECTED VALUE COLLECTION" not in instructions

    def test_execute_unexpected_queries_returns_results_for_failed_checks(self):
        """Unexpected queries are executed and results returned for failed checks."""
        mock_db_hook = MagicMock()
        mock_db_hook.get_records.return_value = [
            {"id": 1, "phone": "bad-fmt"},
            {"id": 2, "phone": "also-bad"},
        ]

        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="customers_validity_1",
                    query="SELECT COUNT(*) AS invalid_phone_count FROM customers",
                    checks=[
                        DQCheck(
                            check_name="phone_fmt",
                            metric_key="invalid_phone_count",
                            group_id="customers_validity_1",
                            check_category="validity",
                            unexpected_query="SELECT id, phone FROM customers WHERE phone IS NULL LIMIT 100",
                        )
                    ],
                )
            ]
        )

        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=mock_db_hook,
            collect_unexpected=True,
        )
        results = planner.execute_unexpected_queries(plan, failed_check_names={"phone_fmt"})

        assert "phone_fmt" in results
        assert len(results["phone_fmt"].unexpected_records) == 2
        assert results["phone_fmt"].unexpected_records[0] == "1, bad-fmt"

    def test_execute_unexpected_queries_skips_checks_without_unexpected_query(self):
        """Checks without unexpected_query are silently skipped."""
        mock_db_hook = MagicMock()

        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="g",
                    query="SELECT COUNT(*) AS null_count FROM t",
                    checks=[
                        DQCheck(
                            check_name="null_emails",
                            metric_key="null_count",
                            group_id="g",
                            check_category="null_check",
                        )
                    ],
                )
            ]
        )

        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=mock_db_hook,
            collect_unexpected=True,
        )
        results = planner.execute_unexpected_queries(plan, failed_check_names={"null_emails"})

        assert results == {}
        mock_db_hook.get_records.assert_not_called()

    def test_execute_unexpected_queries_skips_non_failed_checks(self):
        """Only checks in failed_check_names are executed."""
        mock_db_hook = MagicMock()

        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="g",
                    query="SELECT 1 AS m FROM t",
                    checks=[
                        DQCheck(
                            check_name="passing_check",
                            metric_key="m",
                            group_id="g",
                            check_category="validity",
                            unexpected_query="SELECT id FROM t LIMIT 100",
                        )
                    ],
                )
            ]
        )

        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=mock_db_hook,
            collect_unexpected=True,
        )
        results = planner.execute_unexpected_queries(plan, failed_check_names=set())

        assert results == {}
        mock_db_hook.get_records.assert_not_called()

    def test_execute_unexpected_queries_skips_unsafe_unexpected_query(self, caplog):
        """Unsafe unexpected queries are skipped with a warning, not re-raised."""
        mock_db_hook = MagicMock()

        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="g",
                    query="SELECT 1 AS m FROM t",
                    checks=[
                        DQCheck(
                            check_name="bad_uq",
                            metric_key="m",
                            group_id="g",
                            check_category="validity",
                            unexpected_query="DROP TABLE customers",
                        )
                    ],
                )
            ]
        )

        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=mock_db_hook,
            collect_unexpected=True,
        )
        with caplog.at_level("WARNING"):
            results = planner.execute_unexpected_queries(plan, failed_check_names={"bad_uq"})

        assert results == {}
        assert "failed safety validation" in caplog.text
        mock_db_hook.get_records.assert_not_called()

    def test_grouping_prompt_includes_category_and_max_checks(self):
        """System prompt contains category list and max-checks-per-group rule."""
        checks = _make_checks("c1")
        plan = _make_plan("c1")
        mock_hook = _make_llm_hook(plan)

        planner = SQLDQPlanner(llm_hook=mock_hook, db_hook=None)
        planner.generate_plan(checks, schema_context="")

        instructions = mock_hook.create_agent.call_args.kwargs["instructions"]
        assert "null_check" in instructions
        assert "uniqueness" in instructions
        assert "validity" in instructions
        assert "numeric_range" in instructions
        assert "string_format" in instructions
        assert "MAX 5 CHECKS PER GROUP" in instructions
        assert "check_category" in instructions


def _make_row_level_plan(check_name: str = "age_valid") -> DQPlan:
    """Helper: minimal DQPlan with one row-level group."""
    return DQPlan(
        groups=[
            DQCheckGroup(
                group_id="row_group",
                query="SELECT age FROM customers",
                checks=[
                    DQCheck(
                        check_name=check_name,
                        metric_key="age",
                        group_id="row_group",
                        row_level=True,
                    )
                ],
            )
        ]
    )


class TestSQLDQPlannerRowLevel:
    """Tests for the row-level execution path in _execute_row_level_group."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_cursor(rows: list, col_names: list[str]) -> MagicMock:
        """Return a mock cursor that yields *rows* on the first fetchmany() call."""
        cursor = MagicMock()
        cursor.description = [(name, None, None, None, None, None, None) for name in col_names]
        # First fetchmany returns the rows; second signals end-of-result.
        cursor.fetchmany.side_effect = [rows, []]
        return cursor

    @staticmethod
    def _make_db_hook(cursor: MagicMock) -> MagicMock:
        """Wrap *cursor* in a mock DbApiHook whose get_conn() returns a mock connection."""
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = cursor
        mock_hook = MagicMock(spec=DbApiHook)
        mock_hook.get_conn.return_value = mock_conn
        return mock_hook

    # ------------------------------------------------------------------
    # DataFusion path
    # ------------------------------------------------------------------

    def test_raises_when_row_level_validator_is_missing(self):
        """Row-level checks without matching validators must fail fast."""
        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=None,
            datasource_config=MagicMock(),
            row_validators={},
        )

        with pytest.raises(ValueError, match="requires row-level validator"):
            planner._execute_row_level_group(
                _make_row_level_plan().groups[0],
                datafusion_engine=MagicMock(),
            )

    def test_datafusion_empty_query_returns_zero_total(self):
        """DataFusion iter_query_row_chunks yielding no batches must produce total=0."""
        mock_engine = MagicMock()
        mock_engine.iter_query_row_chunks.return_value = iter([])

        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=None,
            datasource_config=MagicMock(),
            row_validators={"age_valid": lambda v: v is not None},
        )

        result = planner._execute_row_level_group(
            _make_row_level_plan().groups[0],
            datafusion_engine=mock_engine,
        )

        assert "age_valid" in result
        r = result["age_valid"]
        assert r.total == 0
        assert r.invalid == 0
        assert r.invalid_pct == 0.0
        assert r.sample_violations == []
        assert r.sample_size == 0

    def test_datafusion_rows_processed_per_batch(self):
        """DataFusion path processes rows from multiple RecordBatch dicts correctly."""
        mock_engine = MagicMock()
        # Two batches: batch-1 has 2 rows, batch-2 has 1 row with None age.
        mock_engine.iter_query_row_chunks.return_value = iter(
            [
                {"age": [25, 30]},
                {"age": [None]},
            ]
        )

        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=None,
            datasource_config=MagicMock(),
            row_validators={"age_valid": lambda v: v is not None},
        )

        result = planner._execute_row_level_group(
            _make_row_level_plan().groups[0],
            datafusion_engine=mock_engine,
        )

        assert "age_valid" in result
        r = result["age_valid"]
        assert r.total == 3
        assert r.invalid == 1
        assert r.invalid_pct == pytest.approx(1 / 3)
        assert r.sample_violations == ["None"]

    # ------------------------------------------------------------------
    # DB path — basic cases
    # ------------------------------------------------------------------

    def test_db_empty_query_returns_zero_total(self):
        """Cursor returning no rows must yield RowLevelResult(total=0) per check."""
        cursor = self._make_cursor(rows=[], col_names=["age"])
        mock_db_hook = self._make_db_hook(cursor)

        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=mock_db_hook,
            row_validators={"age_valid": lambda v: v is not None},
        )

        result = planner._execute_row_level_group(
            _make_row_level_plan().groups[0],
            datafusion_engine=None,
        )

        assert "age_valid" in result
        r = result["age_valid"]
        assert r.total == 0
        assert r.invalid == 0
        assert r.invalid_pct == 0.0
        assert r.sample_violations == []
        assert r.sample_size == 0

    def test_db_tuple_rows_use_cursor_description_for_column_names(self):
        """Tuple rows must be mapped via cursor.description, not metric_key position.

        The LLM-generated SELECT includes a leading PK column:
            SELECT id, age FROM customers
        Without cursor.description the fallback would map position-0 (id=1) to 'age',
        silently validating the wrong value.  With description the mapping is correct.
        """
        rows = [(1, 25), (2, None), (3, 30)]
        cursor = self._make_cursor(rows=rows, col_names=["id", "age"])
        mock_db_hook = self._make_db_hook(cursor)

        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=mock_db_hook,
            row_validators={"age_valid": lambda v: v is not None},
        )

        result = planner._execute_row_level_group(
            _make_row_level_plan().groups[0],
            datafusion_engine=None,
        )

        assert "age_valid" in result
        r = result["age_valid"]
        # Row (2, None): age is None → 1 invalid; (1,25) and (3,30) pass.
        assert r.total == 3
        assert r.invalid == 1
        assert r.sample_violations == ["None"]

    # ------------------------------------------------------------------
    # DB path — chunked processing
    # ------------------------------------------------------------------

    def test_db_rows_spread_across_multiple_chunks(self):
        """All rows across multiple fetchmany() calls are counted correctly."""
        cursor = MagicMock()
        cursor.description = [("age", None, None, None, None, None, None)]
        # Simulate three chunks: 2 valid, 1 invalid (None), empty sentinel.
        cursor.fetchmany.side_effect = [
            [(25,), (30,)],
            [(None,)],
            [],
        ]
        mock_db_hook = self._make_db_hook(cursor)

        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=mock_db_hook,
            row_validators={"age_valid": lambda v: v is not None},
        )

        result = planner._execute_row_level_group(
            _make_row_level_plan().groups[0],
            datafusion_engine=None,
        )

        r = result["age_valid"]
        assert r.total == 3
        assert r.invalid == 1
        assert r.sample_violations == ["None"]

    def test_db_violation_sample_capped_at_max(self):
        """sample_violations must never exceed _MAX_VIOLATION_SAMPLES regardless of row count."""
        from airflow.providers.common.ai.utils.dq_planner import _MAX_VIOLATION_SAMPLES

        # Generate more invalid values than the cap.
        num_invalid = _MAX_VIOLATION_SAMPLES + 50
        rows = [(None,)] * num_invalid

        cursor = MagicMock()
        cursor.description = [("age", None, None, None, None, None, None)]
        cursor.fetchmany.side_effect = [rows, []]
        mock_db_hook = self._make_db_hook(cursor)

        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=mock_db_hook,
            row_validators={"age_valid": lambda v: v is not None},
        )

        result = planner._execute_row_level_group(
            _make_row_level_plan().groups[0],
            datafusion_engine=None,
        )

        r = result["age_valid"]
        assert r.total == num_invalid
        assert r.invalid == num_invalid
        # Violation list is capped — must not grow unboundedly.
        assert len(r.sample_violations) == _MAX_VIOLATION_SAMPLES
        assert r.sample_size == len(r.sample_violations)
        assert r.sample_size == _MAX_VIOLATION_SAMPLES

    def test_db_dict_rows_processed_correctly(self):
        """Dict rows (some DB drivers return dicts) are handled without column-name lookup."""
        rows = [{"id": 1, "age": 25}, {"id": 2, "age": None}, {"id": 3, "age": 30}]
        cursor = MagicMock()
        cursor.description = [
            ("id", None, None, None, None, None, None),
            ("age", None, None, None, None, None, None),
        ]
        cursor.fetchmany.side_effect = [rows, []]
        mock_db_hook = self._make_db_hook(cursor)

        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=mock_db_hook,
            row_validators={"age_valid": lambda v: v is not None},
        )

        result = planner._execute_row_level_group(
            _make_row_level_plan().groups[0],
            datafusion_engine=None,
        )

        r = result["age_valid"]
        assert r.total == 3
        assert r.invalid == 1
        assert r.sample_violations == ["None"]

    def test_validator_exception_counted_as_invalid(self):
        """A validator that raises must count the row as invalid, not propagate the exception."""
        rows = [(25,), ("bad_value",), (30,)]
        cursor = self._make_cursor(rows=rows, col_names=["age"])
        mock_db_hook = self._make_db_hook(cursor)

        def strict_validator(v):
            if not isinstance(v, int):
                raise TypeError("expected int")
            return v > 0

        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=mock_db_hook,
            row_validators={"age_valid": strict_validator},
        )

        result = planner._execute_row_level_group(
            _make_row_level_plan().groups[0],
            datafusion_engine=None,
        )

        r = result["age_valid"]
        assert r.total == 3
        assert r.invalid == 1  # only "bad_value" raises; 25 and 30 pass
        assert r.sample_violations == ["bad_value"]


class TestSQLDQPlannerGroupHomogeneity:
    """execute_plan must reject groups that mix row-level and aggregate checks."""

    def _planner_with_db_hook(self, mock_db_hook) -> SQLDQPlanner:
        return SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=mock_db_hook,
            row_validators={"age_valid": lambda v: v is not None},
        )

    def test_mixed_group_raises_value_error(self):
        mock_db_hook = MagicMock(spec=DbApiHook)
        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="mixed_group",
                    query="SELECT COUNT(*) AS total, age FROM customers",
                    checks=[
                        DQCheck(
                            check_name="row_count",
                            metric_key="total",
                            group_id="mixed_group",
                            row_level=False,
                        ),
                        DQCheck(
                            check_name="age_valid",
                            metric_key="age",
                            group_id="mixed_group",
                            row_level=True,
                        ),
                    ],
                )
            ]
        )
        planner = self._planner_with_db_hook(mock_db_hook)
        with pytest.raises(ValueError, match="mixed_group"):
            planner.execute_plan(plan)

    def test_pure_row_level_group_not_rejected(self):
        cursor = MagicMock()
        cursor.description = [("age", None, None, None, None, None, None)]
        cursor.fetchmany.side_effect = [[], []]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = cursor
        mock_db_hook = MagicMock(spec=DbApiHook)
        mock_db_hook.get_conn.return_value = mock_conn

        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="row_group",
                    query="SELECT age FROM customers",
                    checks=[
                        DQCheck(
                            check_name="age_valid",
                            metric_key="age",
                            group_id="row_group",
                            row_level=True,
                        )
                    ],
                )
            ]
        )
        planner = self._planner_with_db_hook(mock_db_hook)
        result = planner.execute_plan(plan)
        assert "age_valid" in result

    def test_pure_aggregate_group_not_rejected(self):
        mock_db_hook = MagicMock(spec=DbApiHook)
        mock_db_hook.get_records.return_value = [{"total": 100}]

        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="agg_group",
                    query="SELECT COUNT(*) AS total FROM customers",
                    checks=[
                        DQCheck(
                            check_name="row_count",
                            metric_key="total",
                            group_id="agg_group",
                            row_level=False,
                        )
                    ],
                )
            ]
        )
        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=mock_db_hook,
        )
        result = planner.execute_plan(plan)
        assert result["row_count"] == 100


class TestIterDbRowChunks:
    """_iter_db_row_chunks must fail fast on driver/query anomalies."""

    @staticmethod
    def _make_planner(cursor: MagicMock) -> SQLDQPlanner:
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = cursor
        mock_hook = MagicMock(spec=DbApiHook)
        mock_hook.get_conn.return_value = mock_conn
        return SQLDQPlanner(llm_hook=MagicMock(spec=PydanticAIHook), db_hook=mock_hook)

    def test_raises_when_cursor_description_is_none(self):
        cursor = MagicMock()
        cursor.description = None
        planner = self._make_planner(cursor)
        with pytest.raises(ValueError, match="cursor.description is None"):
            list(planner._iter_db_row_chunks("SELECT 1"))

    def test_raises_for_unsupported_row_type(self):
        cursor = MagicMock()
        cursor.description = [("col", None, None, None, None, None, None)]
        cursor.fetchmany.side_effect = [[42], []]
        planner = self._make_planner(cursor)
        with pytest.raises(ValueError, match="Unsupported row type"):
            list(planner._iter_db_row_chunks("SELECT 1"))

    def test_raises_when_tuple_length_mismatches_description(self):
        cursor = MagicMock()
        cursor.description = [("a", None, None, None, None, None, None)]
        cursor.fetchmany.side_effect = [[(1, 2)], []]
        planner = self._make_planner(cursor)
        with pytest.raises(ValueError, match=r"1 column\(s\)"):
            list(planner._iter_db_row_chunks("SELECT 1"))


class TestSQLDQPlannerDuplicateCheckName:
    """Duplicate check_name entries in LLM-generated plans must be rejected."""

    def test_raises_when_plan_has_duplicate_check_names(self):
        """A plan where the same check_name appears in two groups must raise ValueError."""
        checks = _make_checks("null_emails")
        # LLM returns check_name "null_emails" in two separate groups (duplicate)
        duplicate_plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="g1",
                    query="SELECT COUNT(*) AS a FROM t",
                    checks=[DQCheck(check_name="null_emails", metric_key="a", group_id="g1")],
                ),
                DQCheckGroup(
                    group_id="g2",
                    query="SELECT COUNT(*) AS b FROM t",
                    checks=[DQCheck(check_name="null_emails", metric_key="b", group_id="g2")],
                ),
            ]
        )
        mock_hook = _make_llm_hook(duplicate_plan)
        planner = SQLDQPlanner(llm_hook=mock_hook, db_hook=None)
        with pytest.raises(ValueError, match="null_emails"):
            planner.generate_plan(checks, schema_context="")

    def test_unique_check_names_do_not_raise(self):
        """A plan with all unique check_names must pass coverage validation."""
        checks = _make_checks("check_a", "check_b")
        valid_plan = _make_plan("check_a", "check_b")
        mock_hook = _make_llm_hook(valid_plan)
        planner = SQLDQPlanner(llm_hook=mock_hook, db_hook=None)
        result = planner.generate_plan(checks, schema_context="")
        assert set(result.check_names) == {"check_a", "check_b"}


class TestSQLDQPlannerValidatorSelectionValidation:
    """Planner must fail closed when LLM validator selection is missing/invalid."""

    def test_raises_when_non_fixed_check_has_null_validator_name(self):
        checks = _make_checks("null_emails")
        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="g",
                    query="SELECT COUNT(*) AS null_email_count FROM t",
                    checks=[
                        DQCheck(
                            check_name="null_emails",
                            metric_key="null_email_count",
                            group_id="g",
                            check_category="row_level",
                            validator_name=None,
                            validator_args={},
                        )
                    ],
                )
            ]
        )
        mock_hook = _make_llm_hook(plan)
        planner = SQLDQPlanner(llm_hook=mock_hook, db_hook=None, max_validator_retries=1)

        with pytest.raises(ValueError, match="validator_name is null or 'none'"):
            planner.generate_plan(checks, schema_context="")

    def test_raises_when_non_fixed_check_has_none_validator_name(self):
        checks = _make_checks("null_emails")
        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="g",
                    query="SELECT COUNT(*) AS null_email_count FROM t",
                    checks=[
                        DQCheck(
                            check_name="null_emails",
                            metric_key="null_email_count",
                            group_id="g",
                            check_category="row_level",
                            validator_name="none",
                            validator_args={},
                        )
                    ],
                )
            ]
        )
        mock_hook = _make_llm_hook(plan)
        planner = SQLDQPlanner(llm_hook=mock_hook, db_hook=None, max_validator_retries=1)

        with pytest.raises(ValueError, match="validator_name is null or 'none'"):
            planner.generate_plan(checks, schema_context="")

    def test_aggregate_check_with_none_validator_is_accepted(self):
        """Aggregate checks may have no validator — 'none' is a valid pass-through."""
        checks = _make_checks("null_emails")
        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="g",
                    query="SELECT COUNT(*) AS null_email_count FROM t",
                    checks=[
                        DQCheck(
                            check_name="null_emails",
                            metric_key="null_email_count",
                            group_id="g",
                            check_category="null_check",
                            validator_name=None,
                            validator_args={},
                        )
                    ],
                )
            ]
        )
        mock_hook = _make_llm_hook(plan)
        planner = SQLDQPlanner(llm_hook=mock_hook, db_hook=None, max_validator_retries=1)

        result = planner.generate_plan(checks, schema_context="")
        assert result.check_names == ["null_emails"]

    def test_fixed_validator_check_can_keep_null_validator_name(self):
        checks = _make_checks("null_emails")
        plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="g",
                    query="SELECT COUNT(*) AS null_email_count FROM t",
                    checks=[
                        DQCheck(
                            check_name="null_emails",
                            metric_key="null_email_count",
                            group_id="g",
                            validator_name=None,
                            validator_args={},
                        )
                    ],
                )
            ]
        )
        mock_hook = _make_llm_hook(plan)
        planner = SQLDQPlanner(
            llm_hook=mock_hook,
            db_hook=None,
            max_validator_retries=1,
            fixed_validators={"null_emails": lambda v: v == 0},
        )

        result = planner.generate_plan(checks, schema_context="")
        assert result.check_names == ["null_emails"]


class TestSQLDQPlannerRowLevelMetricKeyValidation:
    """Row-level queries not returning the required metric_key column must fail fast."""

    def _make_row_level_group(self, metric_key: str = "email") -> DQCheckGroup:
        return DQCheckGroup(
            group_id="g",
            query=f"SELECT {metric_key} FROM customers",
            checks=[
                DQCheck(
                    check_name="null_check",
                    metric_key=metric_key,
                    group_id="g",
                    row_level=True,
                )
            ],
        )

    def _make_row_validator(self):
        def _v(v):
            return v is not None

        _v._row_level = True
        _v._max_invalid_pct = 0.05
        return _v

    def test_raises_when_row_does_not_contain_metric_key(self):
        """First chunk with wrong column name should raise ValueError naming the missing key."""
        mock_hook = MagicMock(spec=DbApiHook)
        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=mock_hook,
            row_validators={"null_check": self._make_row_validator()},
        )
        group = self._make_row_level_group(metric_key="email")
        # Iterator returns row with "wrong_col" instead of "email"
        with patch.object(planner, "_iter_db_row_chunks", return_value=iter([[{"wrong_col": "value"}]])):
            with pytest.raises(ValueError, match="email"):
                planner._execute_row_level_group(group, None)

    def test_correct_metric_key_processes_without_error(self):
        """First chunk with the correct metric_key column should process all rows."""
        mock_hook = MagicMock(spec=DbApiHook)
        planner = SQLDQPlanner(
            llm_hook=MagicMock(spec=PydanticAIHook),
            db_hook=mock_hook,
            row_validators={"null_check": self._make_row_validator()},
        )
        group = self._make_row_level_group(metric_key="email")
        with patch.object(
            planner,
            "_iter_db_row_chunks",
            return_value=iter([[{"email": "a@b.com"}, {"email": None}, {"email": "c@d.com"}]]),
        ):
            results = planner._execute_row_level_group(group, None)

        assert "null_check" in results
        assert results["null_check"].total == 3
        assert results["null_check"].invalid == 1  # None row


class TestIterDataFusionRowChunks:
    """_iter_datafusion_row_chunks must validate column lengths within each batch."""

    def test_raises_when_column_lengths_are_inconsistent(self):
        """A batch where columns have different lengths should raise ValueError."""
        mock_engine = MagicMock()
        mock_engine.iter_query_row_chunks.return_value = iter(
            [{"id": [1, 2], "email": [None]}]  # id has 2 values, email has 1
        )
        planner = SQLDQPlanner(llm_hook=MagicMock(spec=PydanticAIHook), db_hook=None)
        with pytest.raises(ValueError, match="inconsistent column lengths"):
            list(planner._iter_datafusion_row_chunks(mock_engine, "SELECT id, email FROM t"))

    def test_consistent_column_lengths_yield_correct_rows(self):
        """Batches with equal-length columns should be transposed to row dicts correctly."""
        mock_engine = MagicMock()
        mock_engine.iter_query_row_chunks.return_value = iter(
            [{"id": [1, 2], "email": ["a@b.com", "c@d.com"]}]
        )
        planner = SQLDQPlanner(llm_hook=MagicMock(spec=PydanticAIHook), db_hook=None)
        rows = list(planner._iter_datafusion_row_chunks(mock_engine, "SELECT id, email FROM t"))

        assert rows == [[{"id": 1, "email": "a@b.com"}, {"id": 2, "email": "c@d.com"}]]

    def test_empty_col_dict_is_skipped(self):
        """Empty column dicts should not produce any output rows."""
        mock_engine = MagicMock()
        mock_engine.iter_query_row_chunks.return_value = iter([{}])
        planner = SQLDQPlanner(llm_hook=MagicMock(spec=PydanticAIHook), db_hook=None)
        rows = list(planner._iter_datafusion_row_chunks(mock_engine, "SELECT 1"))
        assert rows == []


class TestExtractTableNames:
    def test_single_table(self):
        schema = "Table: orders\nColumns: id INT, amount DOUBLE"
        assert _extract_table_names(schema) == ["orders"]

    def test_multiple_tables(self):
        schema = (
            "Table: orders\nColumns: id INT, amount DOUBLE\n\nTable: customers\nColumns: id INT, name VARCHAR"
        )
        assert _extract_table_names(schema) == ["orders", "customers"]

    def test_empty_string(self):
        assert _extract_table_names("") == []

    def test_no_table_prefix(self):
        assert _extract_table_names("Columns: id INT") == []


class TestTableNameConstraintInPrompt:
    def test_constraint_included_when_schema_has_tables(self):
        checks = _make_checks("row_count")
        plan = _make_plan("row_count")
        mock_hook = _make_llm_hook(plan)

        planner = SQLDQPlanner(llm_hook=mock_hook, db_hook=None)
        planner.generate_plan(checks, schema_context="Table: sales_data\nColumns: id INT")

        instructions = mock_hook.create_agent.call_args.kwargs["instructions"]
        assert "TABLE NAME CONSTRAINT" in instructions
        assert "sales_data" in instructions

    def test_constraint_not_included_when_schema_empty(self):
        checks = _make_checks("row_count")
        plan = _make_plan("row_count")
        mock_hook = _make_llm_hook(plan)

        planner = SQLDQPlanner(llm_hook=mock_hook, db_hook=None)
        planner.generate_plan(checks, schema_context="")

        instructions = mock_hook.create_agent.call_args.kwargs["instructions"]
        assert "TABLE NAME CONSTRAINT" not in instructions

    def test_constraint_lists_multiple_tables(self):
        checks = _make_checks("row_count")
        plan = _make_plan("row_count")
        mock_hook = _make_llm_hook(plan)

        schema = "Table: orders\nColumns: id INT\n\nTable: customers\nColumns: id INT"
        planner = SQLDQPlanner(llm_hook=mock_hook, db_hook=None)
        planner.generate_plan(checks, schema_context=schema)

        instructions = mock_hook.create_agent.call_args.kwargs["instructions"]
        assert "orders, customers" in instructions
