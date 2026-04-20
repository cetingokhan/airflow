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

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from airflow.providers.common.ai.decorators.llm_data_quality import (
    _LLMDQDecoratedOperator,
)
from airflow.providers.common.ai.hooks.pydantic_ai import PydanticAIHook
from airflow.providers.common.ai.utils.dq_models import DQCheck, DQCheckGroup, DQCheckInput, DQPlan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHECKS = [
    DQCheckInput(name="null_emails", description="Check for null email addresses"),
    DQCheckInput(name="dup_ids", description="Check for duplicate customer IDs"),
]


def _make_plan() -> DQPlan:
    return DQPlan(
        groups=[
            DQCheckGroup(
                group_id="null_check",
                query="SELECT COUNT(*) AS null_email_count FROM customers WHERE email IS NULL",
                checks=[
                    DQCheck(
                        check_name="null_emails",
                        metric_key="null_email_count",
                        group_id="null_check",
                    )
                ],
            ),
            DQCheckGroup(
                group_id="uniqueness",
                query=(
                    "SELECT COUNT(*) AS dup_id_count FROM ("
                    "SELECT id FROM customers GROUP BY id HAVING COUNT(*) > 1) sub"
                ),
                checks=[
                    DQCheck(
                        check_name="dup_ids",
                        metric_key="dup_id_count",
                        group_id="uniqueness",
                    )
                ],
            ),
        ]
    )


def _make_op(callable_fn=None, **kwargs) -> _LLMDQDecoratedOperator:
    if callable_fn is None:

        def callable_fn():
            return _CHECKS

    return _LLMDQDecoratedOperator(
        task_id="test_dq",
        python_callable=callable_fn,
        llm_conn_id="pydantic_ai_default",
        db_conn_id="postgres_default",
        **kwargs,
    )


@contextmanager
def _patched_runtime(*, plan: DQPlan, results_map: dict, cache_value: str | None = None):
    """Patch runtime collaborators and yield planner/variable mocks."""
    with (
        patch("airflow.providers.common.ai.operators.llm_data_quality.Variable", autospec=True) as mock_var,
        patch(
            "airflow.providers.common.ai.operators.llm_data_quality.SQLDQPlanner",
            autospec=True,
        ) as mock_planner_cls,
        patch(
            "airflow.providers.common.ai.operators.llm_data_quality.get_db_hook",
            autospec=True,
        ),
    ):
        mock_var.get.return_value = cache_value
        mock_planner = mock_planner_cls.return_value
        mock_planner.build_schema_context.return_value = ""
        mock_planner.generate_plan.return_value = plan
        mock_planner.execute_plan.return_value = results_map
        yield mock_var, mock_planner_cls, mock_planner


def _run_op(callable_fn, plan, results_map, **op_kwargs) -> dict:
    with _patched_runtime(plan=plan, results_map=results_map):
        op = _make_op(callable_fn, **op_kwargs)
        op.llm_hook = MagicMock(spec=PydanticAIHook)
        return op.execute(context={})


class TestLLMDQDecoratedOperator:
    def test_custom_operator_name(self):
        assert _LLMDQDecoratedOperator.custom_operator_name == "@task.llm_dq"

    def test_callable_return_value_becomes_checks(self):
        """The list returned by the callable is assigned to op.checks."""
        plan = _make_plan()
        results_map = {"null_emails": 0, "dup_ids": 0}

        def my_checks():
            return _CHECKS

        with _patched_runtime(plan=plan, results_map=results_map):
            op = _make_op(my_checks)
            op.llm_hook = MagicMock(spec=PydanticAIHook)
            op.execute(context={})

            assert op.checks == _CHECKS

    def test_happy_path_all_checks_pass(self):
        checks_with_validators = [
            DQCheckInput(
                name="null_emails", description="Check for null email addresses", validator=lambda v: v == 0
            ),
            DQCheckInput(
                name="dup_ids", description="Check for duplicate customer IDs", validator=lambda v: v == 0
            ),
        ]
        plan = _make_plan()
        result = _run_op(
            lambda: checks_with_validators,
            plan,
            {"null_emails": 0, "dup_ids": 0},
        )
        assert result["passed"] is True

    def test_raises_on_invalid_return_value(self):
        """TypeError when the callable returns a non-list or empty value."""
        op = _make_op(lambda: "not a list")
        with pytest.raises(TypeError, match="non-empty list"):
            op.execute(context={})

    @pytest.mark.parametrize(
        "return_value",
        [[], None, 42, ""],
        ids=["empty-list", "none", "int", "empty-string"],
    )
    def test_raises_on_falsy_return(self, return_value):
        op = _make_op(lambda: return_value)
        with pytest.raises(TypeError, match="non-empty list"):
            op.execute(context={})

    def test_merges_op_kwargs_into_callable(self):
        """op_kwargs are passed to the callable when building the checks list."""
        results_map = {"row_count": 5000}
        single_check_plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="g1",
                    query="SELECT COUNT(*) AS row_count FROM orders",
                    checks=[DQCheck(check_name="row_count", metric_key="row_count", group_id="g1")],
                )
            ]
        )

        def my_checks(min_rows):
            return [
                DQCheckInput(
                    name="row_count",
                    description=f"Orders must have at least {min_rows} rows.",
                ),
            ]

        with _patched_runtime(plan=single_check_plan, results_map=results_map):
            op = _make_op(
                my_checks,
                op_kwargs={"min_rows": 1000},
            )
            op.llm_hook = MagicMock(spec=PydanticAIHook)
            op.execute(context={"task_instance": MagicMock()})
            assert "1000" in op.checks[0].description

    def test_dry_run_returns_plan_dict(self):
        """dry_run=True returns a unified dict with plan, passed=None, results=None."""
        plan = _make_plan()
        result = _run_op(lambda: _CHECKS, plan, {}, dry_run=True)
        assert isinstance(result, dict)
        assert result["passed"] is None
        assert result["results"] is None
        assert "plan" in result
        assert "groups" in result["plan"]
