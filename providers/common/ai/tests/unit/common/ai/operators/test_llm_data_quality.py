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

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import ANY, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from airflow.sdk.types import Context

from airflow.providers.common.ai.hooks.pydantic_ai import PydanticAIHook
from airflow.providers.common.ai.operators.llm_data_quality import (
    LLMDataQualityOperator,
    _compute_plan_hash,
)
from airflow.providers.common.ai.utils.dq_models import DQCheck, DQCheckGroup, DQPlan, RowLevelResult
from airflow.providers.common.ai.utils.dq_validation import (
    default_registry,
    null_pct_check,
    register_validator,
    row_count_check,
)
from airflow.providers.common.compat.sdk import AirflowException

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROMPTS = {
    "null_emails": "Check for null email addresses",
    "dup_ids": "Check for duplicate customer IDs",
}


def _make_context():
    """Return a minimal Airflow context with a mock TaskInstance."""
    mock_ti = MagicMock()
    return cast("Context", {"ti": mock_ti})


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


def _make_operator(**overrides: Any) -> LLMDataQualityOperator:
    defaults: dict[str, Any] = dict(
        task_id="test_dq",
        prompts=_PROMPTS,
        llm_conn_id="pydantic_ai_default",
        db_conn_id="postgres_default",
    )
    defaults.update(overrides)
    op = LLMDataQualityOperator(**defaults)
    op.llm_hook = MagicMock(spec=PydanticAIHook)
    return op


class TestLLMDataQualityOperatorInit:
    def test_requires_llm_conn_id(self):
        with pytest.raises(TypeError):
            LLMDataQualityOperator(
                task_id="test_dq",
                prompts=_PROMPTS,
                db_conn_id="postgres_default",
            )

    def test_template_fields(self):
        op = _make_operator()
        assert hasattr(op, "template_fields")
        assert isinstance(op.template_fields, (list, tuple))
        assert set(op.template_fields) >= {"prompts", "system_prompt", "agent_params"}

    def test_empty_prompts_raises_value_error(self):
        with pytest.raises(ValueError, match="prompts must not be empty"):
            _make_operator(prompts={})


class TestComputePlanHash:
    def test_same_prompts_same_version_yields_same_hash(self):
        h1 = _compute_plan_hash(_PROMPTS, "v1")
        h2 = _compute_plan_hash(_PROMPTS, "v1")
        assert h1 == h2

    def test_different_version_yields_different_hash(self):
        h1 = _compute_plan_hash(_PROMPTS, "v1")
        h2 = _compute_plan_hash(_PROMPTS, "v2")
        assert h1 != h2

    def test_different_prompts_yield_different_hash(self):
        other = {"a": "check A"}
        assert _compute_plan_hash(_PROMPTS, None) != _compute_plan_hash(other, None)

    def test_hash_is_order_independent(self):
        prompts_a = {"a": "x", "b": "y"}
        prompts_b = {"b": "y", "a": "x"}
        assert _compute_plan_hash(prompts_a, None) == _compute_plan_hash(prompts_b, None)

    def test_hash_length_within_variable_key_limit(self):
        from airflow.providers.common.ai.operators.llm_data_quality import _PLAN_VARIABLE_KEY_MAX_LEN

        h = _compute_plan_hash(_PROMPTS, "a" * 300)
        assert len(h) <= _PLAN_VARIABLE_KEY_MAX_LEN

    def test_long_version_tag_preserves_digest_uniqueness(self):
        # Before the fix, a 300-char version_tag caused the full key to be sliced
        # to _PLAN_VARIABLE_KEY_MAX_LEN, truncating the 16-char digest away entirely.
        # Two different prompt payloads with the same long tag would then produce
        # identical cache keys, causing a stale plan to be served.
        long_tag = "x" * 300
        other = {"a": "completely different description"}
        h1 = _compute_plan_hash(_PROMPTS, long_tag)
        h2 = _compute_plan_hash(other, long_tag)
        assert h1 != h2, "Different prompts must produce different hashes even with a long version_tag"


class TestLLMDataQualityOperatorCache:
    @patch("airflow.providers.common.ai.operators.llm_data_quality.Variable", autospec=True)
    @patch("airflow.providers.common.ai.operators.llm_data_quality.SQLDQPlanner", autospec=True)
    @patch("airflow.providers.common.ai.operators.llm_data_quality.get_db_hook", autospec=True)
    def test_cache_miss_calls_generate_and_sets_variable(
        self, mock_get_db_hook, mock_planner_cls, mock_variable
    ):
        plan = _make_plan()
        mock_variable.get.return_value = None  # cache miss
        mock_planner = mock_planner_cls.return_value
        mock_planner.build_schema_context.return_value = ""
        mock_planner.generate_plan.return_value = plan
        mock_planner.execute_plan.return_value = {"null_emails": 0, "dup_ids": 0}

        op = _make_operator()
        op.execute(context=_make_context())

        mock_planner.generate_plan.assert_called_once_with(_PROMPTS, "")
        mock_variable.set.assert_called_once_with(ANY, plan.model_dump_json())

    @patch("airflow.providers.common.ai.operators.llm_data_quality.Variable", autospec=True)
    @patch("airflow.providers.common.ai.operators.llm_data_quality.SQLDQPlanner", autospec=True)
    @patch("airflow.providers.common.ai.operators.llm_data_quality.get_db_hook", autospec=True)
    def test_cache_hit_skips_generate(self, mock_get_db_hook, mock_planner_cls, mock_variable):
        plan = _make_plan()
        mock_variable.get.return_value = plan.model_dump_json()  # cache hit
        mock_planner = mock_planner_cls.return_value
        mock_planner.build_schema_context.return_value = ""
        mock_planner.execute_plan.return_value = {"null_emails": 0, "dup_ids": 0}

        op = _make_operator()
        op.execute(context=_make_context())

        mock_planner.generate_plan.assert_not_called()
        mock_variable.set.assert_not_called()


class TestLLMDataQualityOperatorExecute:
    def _run_operator(self, plan, results_map, validators=None):
        with (
            patch(
                "airflow.providers.common.ai.operators.llm_data_quality.Variable", autospec=True
            ) as mock_var,
            patch(
                "airflow.providers.common.ai.operators.llm_data_quality.SQLDQPlanner", autospec=True
            ) as mock_planner_cls,
            patch("airflow.providers.common.ai.operators.llm_data_quality.get_db_hook", autospec=True),
        ):
            mock_var.get.return_value = None
            mock_planner = mock_planner_cls.return_value
            mock_planner.build_schema_context.return_value = ""
            mock_planner.generate_plan.return_value = plan
            mock_planner.execute_plan.return_value = results_map

            op = _make_operator(validators=validators or {})
            return op.execute(context=_make_context())

    def test_happy_path_all_checks_pass(self):
        plan = _make_plan()
        result = self._run_operator(
            plan,
            {"null_emails": 0, "dup_ids": 0},
            validators={
                "null_emails": lambda v: v == 0,
                "dup_ids": lambda v: v == 0,
            },
        )
        assert result["passed"] is True
        assert all(r["passed"] for r in result["results"])
        assert "plan" in result
        assert "groups" in result["plan"]

    def test_raises_airflow_exception_when_check_fails(self):
        plan = _make_plan()
        with pytest.raises(AirflowException, match="null_emails"):
            self._run_operator(
                plan,
                {"null_emails": 100, "dup_ids": 0},
                validators={"null_emails": lambda v: v == 0},
            )

    def test_failure_message_names_failing_check(self):
        plan = _make_plan()
        with pytest.raises(AirflowException) as exc_info:
            self._run_operator(
                plan,
                {"null_emails": 50, "dup_ids": 3},
                validators={
                    "null_emails": lambda v: v == 0,
                    "dup_ids": lambda v: v == 0,
                },
            )
        msg = str(exc_info.value)
        assert "null_emails" in msg
        assert "dup_ids" in msg

    def test_no_validators_all_pass_by_default(self):
        plan = _make_plan()
        result = self._run_operator(plan, {"null_emails": 999, "dup_ids": 999})
        assert result["passed"] is True

    def test_builtin_validator_factory_works_as_validator(self):
        plan = _make_plan()
        with pytest.raises(AirflowException):
            self._run_operator(
                plan,
                {"null_emails": 0.10, "dup_ids": 0},
                validators={"null_emails": null_pct_check(max_pct=0.05)},
            )

    def test_builtin_row_count_check_passes(self):
        plan = _make_plan()
        result = self._run_operator(
            plan,
            {"null_emails": 0, "dup_ids": 0},
            validators={"null_emails": row_count_check(min_count=0)},
        )
        assert result["passed"] is True

    def test_validator_exception_marks_check_failed(self):
        """If a validator raises, the check is marked failed with the exception message."""
        plan = _make_plan()
        with pytest.raises(AirflowException) as exc_info:
            self._run_operator(
                plan,
                {"null_emails": "not-a-number", "dup_ids": 0},
                validators={"null_emails": null_pct_check(max_pct=0.05)},
            )
        assert "null_emails" in str(exc_info.value)

    def test_failure_reason_uses_validator_display(self):
        name = "_test_failure_reason_display"
        try:

            @register_validator(name)
            def _factory(*, max_pct: float):
                return lambda v: float(v) <= max_pct

            plan = _make_plan()
            with pytest.raises(AirflowException) as exc_info:
                self._run_operator(
                    plan,
                    {"null_emails": 0.2, "dup_ids": 0},
                    validators={"null_emails": _factory(max_pct=0.1)},
                )
            assert f"{name}(max_pct=0.1) returned False" in str(exc_info.value)
        finally:
            default_registry.unregister(name)

    def test_failure_pushes_results_to_xcom_before_raising(self):
        """When checks fail, results are pushed to XCom before the exception is raised."""
        plan = _make_plan()
        with (
            patch(
                "airflow.providers.common.ai.operators.llm_data_quality.Variable", autospec=True
            ) as mock_var,
            patch(
                "airflow.providers.common.ai.operators.llm_data_quality.SQLDQPlanner", autospec=True
            ) as mock_planner_cls,
            patch("airflow.providers.common.ai.operators.llm_data_quality.get_db_hook", autospec=True),
        ):
            mock_var.get.return_value = None
            mock_planner = mock_planner_cls.return_value
            mock_planner.build_schema_context.return_value = ""
            mock_planner.generate_plan.return_value = plan
            mock_planner.execute_plan.return_value = {"null_emails": 100, "dup_ids": 0}

            op = _make_operator(validators={"null_emails": lambda v: v == 0})
            ctx = _make_context()

            with pytest.raises(AirflowException):
                op.execute(context=ctx)

            ctx["ti"].xcom_push.assert_called_once()
            call_kwargs = ctx["ti"].xcom_push.call_args
            assert call_kwargs.kwargs["key"] == "return_value"
            pushed = call_kwargs.kwargs["value"]
            assert pushed["passed"] is False
            assert "plan" in pushed
            assert "groups" in pushed["plan"]
            assert any(r["check_name"] == "null_emails" and not r["passed"] for r in pushed["results"])

    @patch("airflow.providers.common.ai.operators.llm_data_quality.Variable", autospec=True)
    @patch("airflow.providers.common.ai.operators.llm_data_quality.SQLDQPlanner", autospec=True)
    @patch("airflow.providers.common.ai.operators.llm_data_quality.get_db_hook", autospec=True)
    def test_dry_run_returns_plan_dict_without_executing(
        self, mock_get_db_hook, mock_planner_cls, mock_variable
    ):
        plan = _make_plan()
        mock_variable.get.return_value = None
        mock_planner = mock_planner_cls.return_value
        mock_planner.build_schema_context.return_value = ""
        mock_planner.generate_plan.return_value = plan

        op = _make_operator(dry_run=True)
        result = op.execute(context=_make_context())

        mock_planner.execute_plan.assert_not_called()
        assert isinstance(result, dict)
        assert result["passed"] is None
        assert result["results"] is None
        assert "plan" in result
        assert "groups" in result["plan"]
        assert len(result["plan"]["groups"]) == 2
        assert result["plan"]["groups"][0]["group_id"] == "null_check"

    @patch("airflow.providers.common.ai.operators.llm_data_quality.Variable", autospec=True)
    @patch("airflow.providers.common.ai.operators.llm_data_quality.SQLDQPlanner", autospec=True)
    @patch("airflow.providers.common.ai.operators.llm_data_quality.get_db_hook", autospec=True)
    def test_dry_run_still_caches_plan(self, mock_get_db_hook, mock_planner_cls, mock_variable):
        plan = _make_plan()
        mock_variable.get.return_value = None
        mock_planner = mock_planner_cls.return_value
        mock_planner.build_schema_context.return_value = ""
        mock_planner.generate_plan.return_value = plan

        op = _make_operator(dry_run=True)
        op.execute(context=_make_context())

        mock_variable.set.assert_called_once()


class TestLLMDataQualityOperatorRequireApproval:
    """Tests for require_approval behaviour — HITL gates execution before any SQL runs."""

    @patch("airflow.providers.common.ai.operators.llm_data_quality.Variable", autospec=True)
    @patch("airflow.providers.common.ai.operators.llm_data_quality.SQLDQPlanner", autospec=True)
    @patch("airflow.providers.common.ai.operators.llm_data_quality.get_db_hook", autospec=True)
    def test_require_approval_defers_before_execution(
        self, mock_get_db_hook, mock_planner_cls, mock_variable
    ):
        """When require_approval=True, defer_for_approval is called and execute_plan is not."""
        plan = _make_plan()
        mock_variable.get.return_value = None
        mock_planner = mock_planner_cls.return_value
        mock_planner.build_schema_context.return_value = ""
        mock_planner.generate_plan.return_value = plan

        op = _make_operator(require_approval=True)
        with patch.object(op, "defer_for_approval") as mock_defer:
            op.execute(context=_make_context())

        mock_defer.assert_called_once()
        # Second positional arg must be the JSON-serialised plan.
        deferred_output = mock_defer.call_args.args[1]
        DQPlan.model_validate_json(deferred_output)  # must be valid JSON
        mock_planner.execute_plan.assert_not_called()

    @patch("airflow.providers.common.ai.operators.llm_data_quality.Variable", autospec=True)
    @patch("airflow.providers.common.ai.operators.llm_data_quality.SQLDQPlanner", autospec=True)
    @patch("airflow.providers.common.ai.operators.llm_data_quality.get_db_hook", autospec=True)
    def test_dry_run_with_require_approval_returns_plan_dict_no_defer(
        self, mock_get_db_hook, mock_planner_cls, mock_variable
    ):
        """dry_run=True must short-circuit before require_approval is evaluated."""
        plan = _make_plan()
        mock_variable.get.return_value = None
        mock_planner = mock_planner_cls.return_value
        mock_planner.build_schema_context.return_value = ""
        mock_planner.generate_plan.return_value = plan

        op = _make_operator(dry_run=True, require_approval=True)
        with patch.object(op, "defer_for_approval") as mock_defer:
            result = op.execute(context=_make_context())

        mock_defer.assert_not_called()
        mock_planner.execute_plan.assert_not_called()
        assert isinstance(result, dict)
        assert result["passed"] is None
        assert result["results"] is None
        assert "plan" in result
        assert "groups" in result["plan"]

    @patch("airflow.providers.common.ai.operators.llm_data_quality.SQLDQPlanner", autospec=True)
    @patch("airflow.providers.common.ai.operators.llm_data_quality.get_db_hook", autospec=True)
    def test_execute_complete_runs_checks_after_approval(self, mock_get_db_hook, mock_planner_cls):
        """execute_complete deserializes the plan and executes checks on approval."""
        plan = _make_plan()
        mock_planner = mock_planner_cls.return_value
        mock_planner.execute_plan.return_value = {"null_emails": 0, "dup_ids": 0}

        op = _make_operator(require_approval=True)
        approval_event = {
            "chosen_options": ["Approve"],
            "params_input": {},
            "responded_by_user": "alice",
        }
        result = op.execute_complete(
            context=_make_context(),
            generated_output=plan.model_dump_json(),
            event=approval_event,
        )

        mock_planner.execute_plan.assert_called_once()
        assert result["passed"] is True
        assert "plan" in result
        assert "groups" in result["plan"]

    @patch("airflow.providers.common.ai.operators.llm_data_quality.SQLDQPlanner", autospec=True)
    @patch("airflow.providers.common.ai.operators.llm_data_quality.get_db_hook", autospec=True)
    def test_execute_complete_raises_on_rejection(self, mock_get_db_hook, mock_planner_cls):
        """execute_complete raises HITLRejectException when the reviewer rejects."""
        from airflow.providers.standard.exceptions import HITLRejectException

        plan = _make_plan()
        op = _make_operator(require_approval=True)
        rejection_event = {
            "chosen_options": ["Reject"],
            "params_input": {},
            "responded_by_user": "alice",
        }
        with pytest.raises(HITLRejectException):
            op.execute_complete(
                context=_make_context(),
                generated_output=plan.model_dump_json(),
                event=rejection_event,
            )

        mock_planner_cls.return_value.execute_plan.assert_not_called()


class TestLLMDataQualityOperatorSystemPromptAndAgentParams:
    """Verify that LLMOperator's inherited system_prompt / agent_params are forwarded to SQLDQPlanner."""

    def _run_with_planner_spy(self, op: LLMDataQualityOperator, plan: DQPlan):
        """Execute operator and return the kwargs that SQLDQPlanner was constructed with."""
        with (
            patch(
                "airflow.providers.common.ai.operators.llm_data_quality.Variable", autospec=True
            ) as mock_var,
            patch(
                "airflow.providers.common.ai.operators.llm_data_quality.SQLDQPlanner", autospec=True
            ) as mock_planner_cls,
            patch("airflow.providers.common.ai.operators.llm_data_quality.get_db_hook", autospec=True),
        ):
            mock_var.get.return_value = None
            mock_planner = mock_planner_cls.return_value
            mock_planner.build_schema_context.return_value = ""
            mock_planner.generate_plan.return_value = plan
            mock_planner.execute_plan.return_value = {k: 0 for k in plan.check_names}

            op.execute(context=_make_context())
            return mock_planner_cls.call_args.kwargs

    def test_system_prompt_forwarded_to_planner(self):
        plan = _make_plan()
        op = _make_operator(system_prompt="Use ANSI SQL only.")
        kwargs = self._run_with_planner_spy(op, plan)
        assert kwargs["system_prompt"] == "Use ANSI SQL only."

    def test_default_empty_system_prompt_forwarded(self):
        plan = _make_plan()
        op = _make_operator()  # system_prompt defaults to ""
        kwargs = self._run_with_planner_spy(op, plan)
        assert kwargs["system_prompt"] == ""

    def test_agent_params_forwarded_to_planner(self):
        plan = _make_plan()
        op = _make_operator(agent_params={"retries": 5, "model_settings": {"temperature": 0.0}})
        kwargs = self._run_with_planner_spy(op, plan)
        assert kwargs["agent_params"] == {"retries": 5, "model_settings": {"temperature": 0.0}}

    def test_default_empty_agent_params_forwarded(self):
        plan = _make_plan()
        op = _make_operator()  # agent_params defaults to {}
        kwargs = self._run_with_planner_spy(op, plan)
        assert kwargs["agent_params"] == {}


class TestLLMDataQualityOperatorDbHook:
    @patch(
        "airflow.providers.common.ai.operators.llm_data_quality.get_db_hook",
        side_effect=ValueError("Connection 'x' does not provide a DbApiHook."),
    )
    @patch("airflow.providers.common.ai.operators.llm_data_quality.Variable", autospec=True)
    @patch("airflow.providers.common.ai.operators.llm_data_quality.SQLDQPlanner", autospec=True)
    def test_raises_value_error_for_non_dbapi_hook(self, mock_planner_cls, mock_variable, mock_get_db_hook):
        op = _make_operator(db_conn_id="bad_conn")
        with pytest.raises(ValueError, match="DbApiHook"):
            op.execute(context=_make_context())

    def test_none_db_conn_id_returns_none_hook(self):
        op = _make_operator(db_conn_id=None)
        assert op.db_hook is None


class TestLLMDataQualityOperatorCollectUnexpected:
    """Tests for collect_unexpected and unexpected_sample_size parameters."""

    def test_collect_unexpected_in_template_fields(self):
        op = _make_operator()
        assert "collect_unexpected" in op.template_fields
        assert "unexpected_sample_size" in op.template_fields
        assert "row_level_sample_size" in op.template_fields

    def test_collect_unexpected_defaults_false(self):
        op = _make_operator()
        assert op.collect_unexpected is False
        assert op.unexpected_sample_size == 100

    def test_collect_unexpected_forwarded_to_planner(self):
        plan = _make_plan()

        with (
            patch(
                "airflow.providers.common.ai.operators.llm_data_quality.Variable", autospec=True
            ) as mock_var,
            patch(
                "airflow.providers.common.ai.operators.llm_data_quality.SQLDQPlanner", autospec=True
            ) as mock_planner_cls,
            patch("airflow.providers.common.ai.operators.llm_data_quality.get_db_hook", autospec=True),
        ):
            mock_var.get.return_value = None
            mock_planner = mock_planner_cls.return_value
            mock_planner.build_schema_context.return_value = ""
            mock_planner.generate_plan.return_value = plan
            mock_planner.execute_plan.return_value = {"null_emails": 0, "dup_ids": 0}

            op = _make_operator(collect_unexpected=True, unexpected_sample_size=50)
            op.execute(context=_make_context())

            planner_kwargs = mock_planner_cls.call_args.kwargs
            assert planner_kwargs["collect_unexpected"] is True
            assert planner_kwargs["unexpected_sample_size"] == 50

    def test_unexpected_results_in_output_when_check_fails(self):
        from airflow.providers.common.ai.utils.dq_models import UnexpectedResult

        plan = _make_plan()

        with (
            patch(
                "airflow.providers.common.ai.operators.llm_data_quality.Variable", autospec=True
            ) as mock_var,
            patch(
                "airflow.providers.common.ai.operators.llm_data_quality.SQLDQPlanner", autospec=True
            ) as mock_planner_cls,
            patch("airflow.providers.common.ai.operators.llm_data_quality.get_db_hook", autospec=True),
        ):
            mock_var.get.return_value = None
            mock_planner = mock_planner_cls.return_value
            mock_planner.build_schema_context.return_value = ""
            mock_planner.generate_plan.return_value = plan
            mock_planner.execute_plan.return_value = {"null_emails": 100, "dup_ids": 0}
            mock_planner.execute_unexpected_queries.return_value = {
                "null_emails": UnexpectedResult(
                    check_name="null_emails",
                    unexpected_records=["1, None"],
                    sample_size=100,
                )
            }

            op = _make_operator(
                collect_unexpected=True,
                validators={"null_emails": lambda v: v == 0},
            )
            ctx = _make_context()
            with pytest.raises(AirflowException, match="null_emails"):
                op.execute(context=ctx)

            # execute_unexpected_queries must have been called with the failed check name
            mock_planner.execute_unexpected_queries.assert_called_once()
            call_args = mock_planner.execute_unexpected_queries.call_args
            assert "null_emails" in call_args.args[1]  # failed_check_names

    def test_collect_unexpected_false_skips_unexpected_queries(self):
        """When collect_unexpected=False, execute_unexpected_queries is never called."""
        plan = _make_plan()

        with (
            patch(
                "airflow.providers.common.ai.operators.llm_data_quality.Variable", autospec=True
            ) as mock_var,
            patch(
                "airflow.providers.common.ai.operators.llm_data_quality.SQLDQPlanner", autospec=True
            ) as mock_planner_cls,
            patch("airflow.providers.common.ai.operators.llm_data_quality.get_db_hook", autospec=True),
        ):
            mock_var.get.return_value = None
            mock_planner = mock_planner_cls.return_value
            mock_planner.build_schema_context.return_value = ""
            mock_planner.generate_plan.return_value = plan
            mock_planner.execute_plan.return_value = {"null_emails": 100, "dup_ids": 0}

            op = _make_operator(
                collect_unexpected=False,
                validators={"null_emails": lambda v: v == 0},
            )
            with pytest.raises(AirflowException):
                op.execute(context=_make_context())

            mock_planner.execute_unexpected_queries.assert_not_called()


# ---------------------------------------------------------------------------
# Row-level helpers
# ---------------------------------------------------------------------------


def _make_row_level_plan() -> DQPlan:
    """Return a minimal plan whose single check has ``row_level=True``."""
    return DQPlan(
        groups=[
            DQCheckGroup(
                group_id="email_row_level",
                query="SELECT id, email FROM customers",
                checks=[
                    DQCheck(
                        check_name="email_format",
                        metric_key="email",
                        group_id="email_row_level",
                        row_level=True,
                    )
                ],
            )
        ]
    )


def _make_row_level_validator(max_invalid_pct: float):
    """Return a plain callable stamped with row-level introspection attributes."""

    def _check(v: Any) -> bool:
        return bool(v)

    _check._max_invalid_pct = max_invalid_pct  # type: ignore[attr-defined]
    _check._row_level = True  # type: ignore[attr-defined]
    _check._validator_name = "_inline_row_level"  # type: ignore[attr-defined]
    return _check


class TestLLMDataQualityOperatorRowLevel:
    """Tests for row-level check evaluation, serialisation, and planner forwarding."""

    def _run_row_level(self, row_result: Any, max_invalid_pct: float) -> dict[str, Any]:
        plan = _make_row_level_plan()
        with (
            patch(
                "airflow.providers.common.ai.operators.llm_data_quality.Variable", autospec=True
            ) as mock_var,
            patch(
                "airflow.providers.common.ai.operators.llm_data_quality.SQLDQPlanner", autospec=True
            ) as mock_planner_cls,
            patch("airflow.providers.common.ai.operators.llm_data_quality.get_db_hook", autospec=True),
        ):
            mock_var.get.return_value = None
            mock_planner = mock_planner_cls.return_value
            mock_planner.build_schema_context.return_value = ""
            mock_planner.generate_plan.return_value = plan
            mock_planner.execute_plan.return_value = {"email_format": row_result}

            op = _make_operator(
                prompts={"email_format": "Validate email column row by row"},
                validators={"email_format": _make_row_level_validator(max_invalid_pct)},
            )
            return op.execute(context=_make_context())

    def test_row_level_passes_when_under_threshold(self):

        row_result = RowLevelResult(
            check_name="email_format",
            total=100,
            invalid=3,
            invalid_pct=0.03,
            sample_violations=[],
            sample_size=10,
        )
        result = self._run_row_level(row_result, max_invalid_pct=0.05)
        assert result["passed"] is True

    def test_row_level_fails_when_over_threshold(self):

        row_result = RowLevelResult(
            check_name="email_format",
            total=100,
            invalid=10,
            invalid_pct=0.10,
            sample_violations=["bad@", "noemail"],
            sample_size=10,
        )
        with pytest.raises(AirflowException):
            self._run_row_level(row_result, max_invalid_pct=0.05)

    def test_row_level_failure_message_contains_counts_and_threshold(self):

        row_result = RowLevelResult(
            check_name="email_format",
            total=200,
            invalid=30,
            invalid_pct=0.15,
            sample_violations=[],
            sample_size=10,
        )
        with pytest.raises(AirflowException) as exc_info:
            self._run_row_level(row_result, max_invalid_pct=0.05)
        msg = str(exc_info.value)
        assert "30" in msg
        assert "200" in msg
        assert "0.05" in msg or "5.0000%" in msg

    def test_row_level_boundary_at_threshold_passes(self):

        row_result = RowLevelResult(
            check_name="email_format",
            total=100,
            invalid=5,
            invalid_pct=0.05,
            sample_violations=[],
            sample_size=5,
        )
        result = self._run_row_level(row_result, max_invalid_pct=0.05)
        assert result["passed"] is True

    def test_collect_row_validators_returns_only_row_level(self):
        name = "_test_rl_collect_op"
        try:

            @register_validator(name, llm_context="ROW-LEVEL check.", row_level=True)
            def _rl_factory():
                return lambda v: bool(v)

            op = _make_operator(
                prompts={"email_format": "Validate email", "null_emails": "Check nulls"},
                validators={
                    "email_format": _rl_factory(),
                    "null_emails": null_pct_check(max_pct=0.0),
                },
            )
            row_validators = op._collect_row_validators()
            assert "email_format" in row_validators
            assert "null_emails" not in row_validators
        finally:
            default_registry.unregister(name)

    def test_row_level_sample_size_forwarded_to_planner(self):
        plan = _make_plan()
        with (
            patch(
                "airflow.providers.common.ai.operators.llm_data_quality.Variable", autospec=True
            ) as mock_var,
            patch(
                "airflow.providers.common.ai.operators.llm_data_quality.SQLDQPlanner", autospec=True
            ) as mock_planner_cls,
            patch("airflow.providers.common.ai.operators.llm_data_quality.get_db_hook", autospec=True),
        ):
            mock_var.get.return_value = None
            mock_planner = mock_planner_cls.return_value
            mock_planner.build_schema_context.return_value = ""
            mock_planner.generate_plan.return_value = plan
            mock_planner.execute_plan.return_value = {k: 0 for k in plan.check_names}

            op = _make_operator(row_level_sample_size=5000)
            op.execute(context=_make_context())

            planner_kwargs = mock_planner_cls.call_args.kwargs
            assert planner_kwargs["row_level_sample_size"] == 5000

    def test_row_level_result_serialized_as_dict_in_output(self):

        row_result = RowLevelResult(
            check_name="email_format",
            total=50,
            invalid=1,
            invalid_pct=0.02,
            sample_violations=["bad"],
            sample_size=10,
        )
        result = self._run_row_level(row_result, max_invalid_pct=0.05)
        assert result["passed"] is True
        check_result = result["results"][0]
        value = check_result["value"]
        assert isinstance(value, dict)
        assert value["total"] == 50
        assert value["invalid"] == 1
        assert value["invalid_pct"] == 0.02
        assert value["sample_violations"] == ["bad"]
        assert value["sample_size"] == 10

    def test_dry_run_markdown_contains_row_level_section(self):
        plan = _make_row_level_plan()
        op = _make_operator(
            prompts={"email_format": "Validate email column row by row"},
            validators={"email_format": _make_row_level_validator(0.0)},
        )
        md = op._build_dry_run_markdown(plan)
        assert "Row-Level Checks" in md
        assert "email_row_level" in md

    def test_row_level_without_validator_fails(self):
        """Row-level checks must fail when no validator is registered."""
        row_result = RowLevelResult(
            check_name="email_format",
            total=100,
            invalid=90,
            invalid_pct=0.90,
            sample_violations=["bad@"],
            sample_size=10,
        )
        plan = _make_row_level_plan()
        with (
            patch(
                "airflow.providers.common.ai.operators.llm_data_quality.Variable", autospec=True
            ) as mock_var,
            patch(
                "airflow.providers.common.ai.operators.llm_data_quality.SQLDQPlanner", autospec=True
            ) as mock_planner_cls,
            patch("airflow.providers.common.ai.operators.llm_data_quality.get_db_hook", autospec=True),
        ):
            mock_var.get.return_value = None
            mock_planner = mock_planner_cls.return_value
            mock_planner.build_schema_context.return_value = ""
            mock_planner.generate_plan.return_value = plan
            mock_planner.execute_plan.return_value = {"email_format": row_result}

            op = _make_operator(
                prompts={"email_format": "Validate email column row by row"},
                validators={},
            )
            with pytest.raises(AirflowException, match="email_format"):
                op.execute(context=_make_context())


class TestValidateResultsMissingKey:
    """_validate_results must raise ValueError when results_map is missing a check key."""

    def test_raises_value_error_when_planner_returns_partial_results(self):
        plan = _make_plan()

        with (
            patch(
                "airflow.providers.common.ai.operators.llm_data_quality.Variable", autospec=True
            ) as mock_var,
            patch(
                "airflow.providers.common.ai.operators.llm_data_quality.SQLDQPlanner", autospec=True
            ) as mock_planner_cls,
            patch("airflow.providers.common.ai.operators.llm_data_quality.get_db_hook", autospec=True),
        ):
            mock_var.get.return_value = None
            mock_planner = mock_planner_cls.return_value
            mock_planner.build_schema_context.return_value = ""
            mock_planner.generate_plan.return_value = plan
            # Planner only returns one of the two required check results.
            mock_planner.execute_plan.return_value = {"null_emails": 0}

            op = _make_operator(validators={"null_emails": lambda v: v == 0})
            with pytest.raises(ValueError, match="dup_ids"):
                op.execute(context=_make_context())


class TestComputePlanHashExtendedInputs:
    """Non-prompt inputs that must influence cache key generation."""

    def test_different_collect_unexpected_yields_different_hash(self):
        h1 = _compute_plan_hash(_PROMPTS, None, collect_unexpected=False)
        h2 = _compute_plan_hash(_PROMPTS, None, collect_unexpected=True)
        assert h1 != h2

    def test_same_collect_unexpected_yields_same_hash(self):
        h1 = _compute_plan_hash(_PROMPTS, None, collect_unexpected=True)
        h2 = _compute_plan_hash(_PROMPTS, None, collect_unexpected=True)
        assert h1 == h2

    def test_different_row_level_sample_size_yields_different_hash(self):
        h1 = _compute_plan_hash(_PROMPTS, None, row_level_sample_size=None)
        h2 = _compute_plan_hash(_PROMPTS, None, row_level_sample_size=10_000)
        assert h1 != h2

    def test_same_row_level_sample_size_yields_same_hash(self):
        h1 = _compute_plan_hash(_PROMPTS, None, row_level_sample_size=5000)
        h2 = _compute_plan_hash(_PROMPTS, None, row_level_sample_size=5000)
        assert h1 == h2

    def test_none_and_missing_sample_size_equivalent(self):
        h1 = _compute_plan_hash(_PROMPTS, None)
        h2 = _compute_plan_hash(_PROMPTS, None, row_level_sample_size=None)
        assert h1 == h2

    def test_different_schema_context_yields_different_hash(self):
        h1 = _compute_plan_hash(_PROMPTS, None, schema_context="Table: a\nColumns: id INT")
        h2 = _compute_plan_hash(_PROMPTS, None, schema_context="Table: b\nColumns: email TEXT")
        assert h1 != h2

    def test_same_schema_context_yields_same_hash(self):
        schema = "Table: customers\nColumns: id INT, email TEXT"
        h1 = _compute_plan_hash(_PROMPTS, None, schema_context=schema)
        h2 = _compute_plan_hash(_PROMPTS, None, schema_context=schema)
        assert h1 == h2

    def test_empty_and_missing_schema_context_equivalent(self):
        h1 = _compute_plan_hash(_PROMPTS, None)
        h2 = _compute_plan_hash(_PROMPTS, None, schema_context="")
        assert h1 == h2

    def test_different_unexpected_sample_size_with_collect_unexpected_yields_different_hash(self):
        h1 = _compute_plan_hash(_PROMPTS, None, collect_unexpected=True, unexpected_sample_size=50)
        h2 = _compute_plan_hash(_PROMPTS, None, collect_unexpected=True, unexpected_sample_size=200)
        assert h1 != h2

    def test_same_unexpected_sample_size_yields_same_hash(self):
        h1 = _compute_plan_hash(_PROMPTS, None, collect_unexpected=True, unexpected_sample_size=100)
        h2 = _compute_plan_hash(_PROMPTS, None, collect_unexpected=True, unexpected_sample_size=100)
        assert h1 == h2

    def test_schema_context_combined_with_other_params(self):
        """All hashing inputs together produce a unique key."""
        h1 = _compute_plan_hash(
            _PROMPTS, "v1", collect_unexpected=True, unexpected_sample_size=50, schema_context="schema A"
        )
        h2 = _compute_plan_hash(
            _PROMPTS, "v1", collect_unexpected=True, unexpected_sample_size=50, schema_context="schema B"
        )
        assert h1 != h2

    def test_different_validator_contexts_yield_different_hash(self):
        h1 = _compute_plan_hash(_PROMPTS, None, validator_contexts="ctx A")
        h2 = _compute_plan_hash(_PROMPTS, None, validator_contexts="ctx B")
        assert h1 != h2

    def test_same_validator_contexts_yield_same_hash(self):
        h1 = _compute_plan_hash(_PROMPTS, None, validator_contexts="same")
        h2 = _compute_plan_hash(_PROMPTS, None, validator_contexts="same")
        assert h1 == h2

    def test_different_row_validator_thresholds_yield_different_hash(self):
        h1 = _compute_plan_hash(
            _PROMPTS,
            None,
            row_validator_thresholds={"email_format": 0.01, "iban_valid": 0.0},
        )
        h2 = _compute_plan_hash(
            _PROMPTS,
            None,
            row_validator_thresholds={"email_format": 0.05, "iban_valid": 0.0},
        )
        assert h1 != h2

    def test_threshold_order_is_hash_stable(self):
        h1 = _compute_plan_hash(
            _PROMPTS,
            None,
            row_validator_thresholds={"a": 0.1, "b": 0.2},
        )
        h2 = _compute_plan_hash(
            _PROMPTS,
            None,
            row_validator_thresholds={"b": 0.2, "a": 0.1},
        )
        assert h1 == h2


class TestLLMDataQualityOperatorMissingMaxInvalidPct:
    """Row-level validator without _max_invalid_pct defaults to 0.0 and logs a warning."""

    def _run_row_level(self, invalid_pct: float, validator):
        from airflow.providers.common.ai.utils.dq_models import RowLevelResult

        row_plan = DQPlan(
            groups=[
                DQCheckGroup(
                    group_id="g",
                    query="SELECT email FROM customers",
                    checks=[
                        DQCheck(
                            check_name="email_valid",
                            metric_key="email",
                            group_id="g",
                            row_level=True,
                        )
                    ],
                )
            ]
        )
        row_result = RowLevelResult(
            check_name="email_valid",
            total=100,
            invalid=int(invalid_pct * 100),
            invalid_pct=invalid_pct,
            sample_violations=[],
            sample_size=10,
        )

        with (
            patch(
                "airflow.providers.common.ai.operators.llm_data_quality.Variable", autospec=True
            ) as mock_var,
            patch(
                "airflow.providers.common.ai.operators.llm_data_quality.SQLDQPlanner", autospec=True
            ) as mock_planner_cls,
            patch("airflow.providers.common.ai.operators.llm_data_quality.get_db_hook", autospec=True),
        ):
            mock_var.get.return_value = None
            mock_planner = mock_planner_cls.return_value
            mock_planner.build_schema_context.return_value = ""
            mock_planner.generate_plan.return_value = row_plan
            mock_planner.execute_plan.return_value = {"email_valid": row_result}

            op = _make_operator(
                prompts={"email_valid": "check emails"},
                validators={"email_valid": validator},
            )
            return op.execute(context=_make_context())

    def test_zero_invalid_pct_passes_with_missing_max_invalid_pct(self):
        """With no _max_invalid_pct, threshold is 0.0 — zero invalid rows should pass."""
        validator = lambda v: v is not None
        validator._row_level = True
        # Deliberately no _max_invalid_pct attribute
        result = self._run_row_level(0.0, validator)
        assert result["passed"] is True

    def test_nonzero_invalid_pct_fails_with_missing_max_invalid_pct(self):
        """With no _max_invalid_pct, threshold is 0.0 — any invalid row should fail."""
        from airflow.providers.common.compat.sdk import AirflowException

        validator = lambda v: v is not None
        validator._row_level = True
        # Deliberately no _max_invalid_pct attribute
        with pytest.raises(AirflowException):
            self._run_row_level(0.01, validator)
