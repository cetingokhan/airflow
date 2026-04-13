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
"""Operator for generating and executing data-quality checks from natural language using LLMs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any

from airflow.providers.common.ai.operators.llm import LLMOperator
from airflow.providers.common.ai.utils.db_schema import get_db_hook
from airflow.providers.common.ai.utils.dq_models import (
    DQCheckFailedError,
    DQCheckGroup,
    DQCheckResult,
    DQPlan,
    DQReport,
    RowLevelResult,
    UnexpectedResult,
)
from airflow.providers.common.ai.utils.dq_validation import default_registry
from airflow.providers.common.compat.sdk import Variable

try:
    from airflow.providers.common.ai.utils.dq_planner import SQLDQPlanner
except ImportError as e:
    from airflow.providers.common.compat.sdk import AirflowOptionalProviderFeatureException

    raise AirflowOptionalProviderFeatureException(e)

if TYPE_CHECKING:
    from airflow.providers.common.sql.config import DataSourceConfig
    from airflow.providers.common.sql.hooks.sql import DbApiHook
    from airflow.sdk import Context

_PLAN_VARIABLE_PREFIX = "dq_plan_"
_PLAN_VARIABLE_KEY_MAX_LEN = 200  # stay well under Airflow Variable key length limit


def _describe_validator(validator: Callable[[Any], bool]) -> str:
    """Return a human-readable validator label for failure messages."""
    display = getattr(validator, "_validator_display", None)
    if isinstance(display, str) and display:
        return display
    validator_name = getattr(validator, "_validator_name", None)
    if isinstance(validator_name, str) and validator_name:
        return validator_name
    validator_name = getattr(validator, "__name__", None)
    if isinstance(validator_name, str) and validator_name:
        return validator_name
    return repr(validator)


class LLMDataQualityOperator(LLMOperator):
    """
    Generate and execute data-quality checks from natural language descriptions.

    Each entry in ``prompts`` describes **one** data-quality expectation.
    The LLM groups related checks into optimised SQL queries, executes them
    against the target database, and validates each metric against the
    corresponding entry in ``validators``.  The task fails if any check
    does not pass, gating downstream tasks on data quality.

    Generated SQL plans are cached in Airflow
    :class:`~airflow.models.variable.Variable` to avoid repeat LLM calls.
    Set ``dry_run=True`` to preview the plan without executing it — the
    serialised plan dict is returned without running any SQL.
    Set ``require_approval=True`` to gate execution on human review via the
    HITL interface: the plan is presented to the reviewer first, and SQL
    checks run only after approval.  ``dry_run`` and ``require_approval``
    are independent — enabling both returns the plan dict without any
    approval prompt.

    :param prompts: Mapping of ``{check_name: natural_language_description}``.
        Each key must be unique.  Use one check per key; the operator enforces
        a strict one-key → one-check mapping.
    :param llm_conn_id: Connection ID for the LLM provider.
    :param model_id: Model identifier (e.g. ``"openai:gpt-4o"``).
        Overrides the model stored in the connection's extra field.
    :param system_prompt: Additional instructions appended to the planning prompt.
    :param agent_params: Additional keyword arguments passed to the pydantic-ai
        ``Agent`` constructor (e.g. ``retries``, ``model_settings``).
    :param db_conn_id: Connection ID for the database to run checks against.
        Must resolve to a :class:`~airflow.providers.common.sql.hooks.sql.DbApiHook`.
    :param table_names: Tables to include in the LLM's schema context.
    :param schema_context: Manual schema description; bypasses DB introspection.
    :param validators: Mapping of ``{check_name: callable}`` where each callable
        receives the raw metric value and returns ``True`` (pass) or ``False`` (fail).
        Keys must be a subset of ``prompts.keys()``.
        Use built-in factories from
        :mod:`~airflow.providers.common.ai.utils.dq_validation` or plain lambdas::

            from airflow.providers.common.ai.utils.dq_validation import null_pct_check

            validators = {
                "email_nulls": null_pct_check(max_pct=0.05),
                "row_check": lambda v: v >= 1000,
            }

    :param dialect: SQL dialect override (``postgres``, ``mysql``, etc.).
        Auto-detected from *db_conn_id* when not set.
    :param datasource_config: DataFusion datasource for object-storage schema.
    :param dry_run: When ``True``, generate and cache the plan but skip execution.
        Returns the serialised plan dict instead of a :class:`~airflow.providers.common.ai.utils.dq_models.DQReport`.
    :param prompt_version: Optional version tag included in the plan cache key.
        Bump this to invalidate cached plans when prompts change semantically
        without changing their text.
    :param collect_unexpected: When ``True``, the LLM generates an
        ``unexpected_query`` for validity / string-format checks.
        If any of those checks fail, the unexpected query is executed and
        the resulting sample rows are included in the report.
    :param unexpected_sample_size: Maximum number of violating rows to return
        per failed check.  Default ``100``.
    :param row_level_sample_size: Maximum number of rows to fetch per row-level
        check.  ``None`` (default) performs a full table scan — every row is
        fetched and validated.  A positive integer is passed to the LLM as a
        ``LIMIT`` clause on the generated SELECT, bounding execution time and
        memory usage at the cost of sampling coverage.
    :param require_approval: When ``True``, the operator defers after generating
        and caching the DQ plan.  The plan SQL is surfaced in the HITL interface
        for human review; checks run only after the reviewer approves.  Inherited
        from :class:`~airflow.providers.common.ai.operators.llm.LLMOperator`.
        ``dry_run=True`` takes precedence — combining both flags returns the plan
        dict immediately without requesting approval.
    """

    template_fields: Sequence[str] = (
        *LLMOperator.template_fields,
        "prompts",
        "db_conn_id",
        "table_names",
        "schema_context",
        "prompt_version",
        "collect_unexpected",
        "unexpected_sample_size",
        "row_level_sample_size",
    )

    def __init__(
        self,
        *,
        prompts: dict[str, str],
        db_conn_id: str | None = None,
        table_names: list[str] | None = None,
        schema_context: str | None = None,
        validators: dict[str, Callable[[Any], bool]] | None = None,
        dialect: str | None = None,
        datasource_config: DataSourceConfig | None = None,
        prompt_version: str | None = None,
        dry_run: bool = False,
        collect_unexpected: bool = False,
        unexpected_sample_size: int = 100,
        row_level_sample_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("output_type", None)
        kwargs.setdefault("prompt", "LLMDataQualityOperator")
        super().__init__(**kwargs)

        self.prompts = prompts
        self.db_conn_id = db_conn_id
        self.table_names = table_names
        self.schema_context = schema_context
        self.validators = validators or {}
        self.dialect = dialect
        self.datasource_config = datasource_config
        self.prompt_version = prompt_version
        self.dq_dry_run = dry_run
        self.collect_unexpected = collect_unexpected
        self.unexpected_sample_size = unexpected_sample_size
        self.row_level_sample_size = row_level_sample_size

        self._validate_prompts()
        self._validate_validator_keys()

    def execute(self, context: Context) -> dict[str, Any]:
        """
        Generate the DQ plan (or load from cache), then execute or defer for approval.

        When ``dry_run=True`` the serialised plan dict is returned immediately —
        no SQL is executed and no approval is requested.
        When ``require_approval=True`` the task defers, presenting the plan to a
        human reviewer; data-quality checks run only after the reviewer approves.

        :returns: Dict with keys ``plan``, ``passed``, and ``results``.  On success
            ``passed=True`` and ``results`` is a list of per-check result dicts.
            For row-level checks the ``value`` entry in each result dict is itself
            a dict with keys ``total``, ``invalid``, ``invalid_pct``, and
            ``sample_violations`` rather than a raw scalar.
            When ``dry_run=True`` ``passed=None`` and ``results=None`` — no SQL
            is executed.  The ``plan`` key is always present in all modes.
        :raises DQCheckFailedError: If any data-quality check fails threshold validation.
        :raises TaskDeferred: When ``require_approval=True``, defers for human review
            before executing the checks.
        """
        planner = self._build_planner()

        schema_ctx = planner.build_schema_context(
            table_names=self.table_names, schema_context=self.schema_context
        )

        self.log.info("Using schema context:\n%s", schema_ctx)

        plan = self._load_or_generate_plan(planner, schema_ctx)

        if self.dq_dry_run:
            self.log.info(
                "dry_run=True — skipping execution. Plan contains %d group(s), %d check(s).",
                len(plan.groups),
                len(plan.check_names),
            )
            for group in plan.groups:
                self.log.info(
                    "Group: %s\nChecks: %s\nSQL Query:\n%s\n",
                    group.group_id,
                    ", ".join(c.check_name for c in group.checks),
                    group.query,
                )
            return {"plan": plan.model_dump(), "passed": None, "results": None}

        if self.require_approval:
            # Defer BEFORE execution — approval gates the SQL checks.
            self.defer_for_approval(  # type: ignore[misc]
                context,
                plan.model_dump_json(),
                body=self._build_dry_run_markdown(plan),
            )
            return {}  # type: ignore[return-value]  # pragma: no cover

        return self._run_checks_and_report(context, planner, plan)

    def _build_planner(self) -> SQLDQPlanner:
        """Construct a :class:`~airflow.providers.common.ai.utils.dq_planner.SQLDQPlanner` from operator config."""
        return SQLDQPlanner(
            llm_hook=self.llm_hook,
            db_hook=self.db_hook,
            dialect=self.dialect,
            datasource_config=self.datasource_config,
            system_prompt=self.system_prompt,
            agent_params=self.agent_params,
            collect_unexpected=self.collect_unexpected,
            unexpected_sample_size=self.unexpected_sample_size,
            validator_contexts=self.validator_contexts,
            row_validators=self._collect_row_validators(),
            row_level_sample_size=self.row_level_sample_size,
        )

    @cached_property
    def validator_contexts(self) -> str:
        """Return validator-specific LLM context rendered from configured validators."""
        return default_registry.build_llm_context(self.validators)

    def _run_checks_and_report(
        self,
        context: Context,
        planner: SQLDQPlanner,
        plan: DQPlan,
    ) -> dict[str, Any]:
        """
        Execute *plan* against the database, apply validators, and return the serialised report.

        :raises DQCheckFailedError: If any data-quality check fails.
        """
        results_map = planner.execute_plan(plan)
        check_results = self._validate_results(results_map, plan)

        # Collect unexpected rows for failed validity/format checks.
        if self.collect_unexpected:
            failed_names = {r.check_name for r in check_results if not r.passed}
            if failed_names:
                unexpected_map = planner.execute_unexpected_queries(plan, failed_names)
                self._attach_unexpected(check_results, unexpected_map)

        report = DQReport.build(check_results)

        output: dict[str, Any] = {
            "plan": plan.model_dump(),
            "passed": report.passed,
            "results": [
                {
                    "check_name": r.check_name,
                    "metric_key": r.metric_key,
                    # RowLevelResult is not JSON-serialisable; convert to a plain dict.
                    "value": (
                        {
                            "total": r.value.total,
                            "invalid": r.value.invalid,
                            "invalid_pct": r.value.invalid_pct,
                            "sample_violations": r.value.sample_violations,
                            "sample_size": r.value.sample_size,
                        }
                        if isinstance(r.value, RowLevelResult)
                        else r.value
                    ),
                    "passed": r.passed,
                    "failure_reason": r.failure_reason,
                    **(
                        {
                            "unexpected_records": r.unexpected.unexpected_records,
                            "unexpected_sample_size": r.unexpected.sample_size,
                        }
                        if r.unexpected
                        else {}
                    ),
                }
                for r in report.results
            ],
        }

        if not report.passed:
            # Push results to XCom before failing so downstream tasks
            # (e.g. with trigger_rule=all_done) can still inspect them.
            context["ti"].xcom_push(key="return_value", value=output)
            raise DQCheckFailedError(report.failure_summary)

        self.log.info("All %d data-quality check(s) passed.", len(report.results))
        return output

    def _build_dry_run_markdown(self, plan: DQPlan) -> str:
        """
        Build a structured markdown summary of the DQ plan for the HITL review body.

        Aggregate groups and row-level groups are rendered in separate sections so
        reviewers can immediately distinguish SQL-aggregate checks from per-row
        validation logic.
        """
        aggregate_groups = [g for g in plan.groups if not any(c.row_level for c in g.checks)]
        row_level_groups = [g for g in plan.groups if any(c.row_level for c in g.checks)]

        total_checks = len(plan.check_names)
        agg_count = sum(len(g.checks) for g in aggregate_groups)
        row_count = sum(len(g.checks) for g in row_level_groups)

        lines: list[str] = [
            "# LLM Data Quality Plan",
            "",
            "| | |",
            "|---|---|",
            f"| **Plan hash** | `{plan.plan_hash or 'N/A'}` |",
            f"| **Total checks** | {total_checks} |",
            f"| **Aggregate checks** | {agg_count} ({len(aggregate_groups)} group{'s' if len(aggregate_groups) != 1 else ''}) |",
            f"| **Row-level checks** | {row_count} ({len(row_level_groups)} group{'s' if len(row_level_groups) != 1 else ''}) |",
            "",
        ]

        if aggregate_groups:
            lines += [
                "---",
                "",
                "## Aggregate Checks",
                "",
                "> Each group runs as a **single SQL query**. "
                "Result columns are matched to check names by metric key.",
                "",
            ]
            for group in aggregate_groups:
                lines += self._render_aggregate_group(group)

        if row_level_groups:
            lines += [
                "---",
                "",
                "## Row-Level Checks",
                "",
                "> Row-level checks fetch **raw column values** and apply Python-side "
                "validation per row. The threshold controls the maximum allowed fraction "
                "of invalid rows before the check fails.",
                "",
            ]
            for group in row_level_groups:
                lines += self._render_row_level_group(group)

        return "\n".join(lines).rstrip()

    def _render_aggregate_group(self, group: DQCheckGroup) -> list[str]:
        """Render one aggregate SQL group as a markdown subsection."""
        lines: list[str] = [
            f"### `{group.group_id}`",
            "",
            "| Check name | Metric key | Category |",
            "|---|---|---|",
        ]
        for check in group.checks:
            category = check.check_category or "—"
            lines.append(f"| `{check.check_name}` | `{check.metric_key}` | {category} |")

        lines += [
            "",
            "```sql",
            group.query.strip(),
            "```",
            "",
        ]

        # Unexpected queries — only show when present.
        unexpected = [(c.check_name, c.unexpected_query) for c in group.checks if c.unexpected_query]
        if unexpected:
            lines += ["<details><summary>Unexpected-row queries</summary>", ""]
            for check_name, uq in unexpected:
                lines += [
                    f"**`{check_name}`**",
                    "",
                    "```sql",
                    (uq or "").strip(),
                    "```",
                    "",
                ]
            lines += ["</details>", ""]

        return lines

    def _render_row_level_group(self, group: DQCheckGroup) -> list[str]:
        """Render one row-level group as a markdown subsection with threshold info."""
        lines: list[str] = [
            f"### `{group.group_id}`",
            "",
            "| Check name | Metric key | Max invalid % |",
            "|---|---|---|",
        ]
        for check in group.checks:
            validator = self.validators.get(check.check_name)
            max_pct = (
                self._resolve_row_level_max_invalid_pct(
                    check.check_name,
                    validator,
                    default_when_missing=None,
                    warn_on_missing=False,
                )
                if validator is not None
                else None
            )
            threshold_str = f"{max_pct:.2%}" if max_pct is not None else "—"
            lines.append(f"| `{check.check_name}` | `{check.metric_key}` | {threshold_str} |")

        lines += [
            "",
            "```sql",
            group.query.strip(),
            "```",
            "",
        ]
        return lines

    def _load_or_generate_plan(self, planner: SQLDQPlanner, schema_ctx: str) -> DQPlan:
        """Return a cached plan when available, otherwise generate and cache a new one."""
        if not isinstance(self.prompts, dict):
            raise TypeError("prompts must be a dict[str, str] before generating a DQ plan.")

        row_validator_thresholds = self._collect_row_validator_thresholds()

        plan_hash = _compute_plan_hash(
            self.prompts,
            self.prompt_version,
            self.collect_unexpected,
            self.row_level_sample_size,
            schema_context=schema_ctx,
            unexpected_sample_size=self.unexpected_sample_size,
            validator_contexts=self.validator_contexts,
            row_validator_thresholds=row_validator_thresholds,
        )
        variable_key = f"{_PLAN_VARIABLE_PREFIX}{plan_hash}"

        cached_json = Variable.get(variable_key, None)
        if cached_json is not None:
            self.log.info("DQ plan cache hit — key: %r", variable_key)
            plan = DQPlan.model_validate_json(cached_json)
            if not plan.plan_hash:
                plan.plan_hash = plan_hash
            return plan

        self.log.info("DQ plan cache miss — generating via LLM (key: %r).", variable_key)
        plan = planner.generate_plan(self.prompts, schema_ctx)
        plan.plan_hash = plan_hash
        Variable.set(variable_key, plan.model_dump_json())
        return plan

    def _resolve_row_level_max_invalid_pct(
        self,
        check_name: str,
        validator: Callable[[Any], bool],
        *,
        default_when_missing: float | None,
        warn_on_missing: bool,
    ) -> float | None:
        """Return row-level threshold as ``float`` or raise a clear ``ValueError``."""
        if not hasattr(validator, "_max_invalid_pct"):
            if warn_on_missing:
                self.log.warning(
                    "Row-level validator for check %r has no '_max_invalid_pct' attribute — "
                    "defaulting threshold to 0.0%%. Every invalid row will fail the check.",
                    check_name,
                )
            return default_when_missing

        raw_max_pct = getattr(validator, "_max_invalid_pct")
        try:
            return float(raw_max_pct)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Row-level validator for check {check_name!r} has invalid _max_invalid_pct "
                f"value {raw_max_pct!r}; expected a numeric value."
            ) from exc

    def _validate_results(
        self,
        results_map: dict[str, Any],
        plan: DQPlan,
    ) -> list[DQCheckResult]:
        """
        Apply validators to each metric value and return per-check results.

        For aggregate checks each validator callable receives the raw metric
        value returned by the database.  For row-level checks, where *value* is
        a :class:`~airflow.providers.common.ai.utils.dq_models.RowLevelResult`,
        the pass/fail decision compares ``invalid_pct`` against the validator's
        ``_max_invalid_pct`` attribute (defaulting to ``0.0`` when absent).
        Aggregate checks without a registered validator are logged and marked
        as passed; row-level checks require an explicit validator and fail
        when none is provided.

        :param results_map: ``{check_name: metric_value_or_RowLevelResult}`` as
            returned by
            :meth:`~airflow.providers.common.ai.utils.dq_planner.SQLDQPlanner.execute_plan`.
        :param plan: The DQ plan whose groups and checks drive iteration order.
        :returns: Per-check
            :class:`~airflow.providers.common.ai.utils.dq_models.DQCheckResult`
            list in plan-group order.
        :raises ValueError: If *results_map* is missing a key for any check in *plan*.
        """
        check_results: list[DQCheckResult] = []

        for group in plan.groups:
            for check in group.checks:
                if check.check_name not in results_map:
                    raise ValueError(
                        f"Planner did not return a result for check {check.check_name!r} "
                        f"(group {group.group_id!r}). Available keys: {sorted(results_map)}"
                    )
                value = results_map[check.check_name]
                validator = self.validators.get(check.check_name)

                passed = True
                failure_reason: str | None = None

                if isinstance(value, RowLevelResult):
                    if validator is None:
                        self.log.error(
                            "No validator found for row-level check %r (metric key: %r). "
                            "Row-level checks require an explicit validator.",
                            check.check_name,
                            check.metric_key,
                        )
                        passed = False
                        failure_reason = (
                            "Row-level check requires a registered row-level validator, "
                            "but none was provided."
                        )
                    else:
                        max_pct = self._resolve_row_level_max_invalid_pct(
                            check.check_name,
                            validator,
                            default_when_missing=0.0,
                            warn_on_missing=True,
                        )
                        passed = value.invalid_pct <= (max_pct if max_pct is not None else 0.0)
                        if not passed:
                            failure_reason = (
                                f"Row-level check failed: {value.invalid}/{value.total} rows invalid "
                                f"({value.invalid_pct:.4%}), threshold {max_pct:.4%}"
                            )
                elif validator is not None:
                    try:
                        passed = bool(validator(value))
                    except Exception as exc:
                        passed = False
                        failure_reason = str(exc)

                    if not passed and failure_reason is None:
                        failure_reason = f"{_describe_validator(validator)} returned False"
                else:
                    self.log.warning(
                        "No validator found for check %r (metric key: %r). Marking as passed by default.",
                        check.check_name,
                        check.metric_key,
                    )

                check_results.append(
                    DQCheckResult(
                        check_name=check.check_name,
                        metric_key=check.metric_key,
                        value=value,
                        passed=passed,
                        failure_reason=failure_reason,
                    )
                )

        return check_results

    def _collect_row_validators(self) -> dict[str, Callable[[Any], bool]]:
        """Return the subset of validators that are row-level."""
        return {name: fn for name, fn in self.validators.items() if default_registry.is_row_level(fn)}

    def _collect_row_validator_thresholds(self) -> dict[str, float | str | None]:
        """Return a deterministic ``{check_name: threshold}`` map for row-level validators."""
        thresholds: dict[str, float | str | None] = {}
        for name, validator in self._collect_row_validators().items():
            raw_threshold = getattr(validator, "_max_invalid_pct", None)
            if isinstance(raw_threshold, int | float):
                thresholds[name] = float(raw_threshold)
            elif raw_threshold is None:
                thresholds[name] = None
            else:
                thresholds[name] = str(raw_threshold)
        return thresholds

    @staticmethod
    def _attach_unexpected(
        check_results: list[DQCheckResult],
        unexpected_map: dict[str, UnexpectedResult],
    ) -> None:
        """Attach :class:`UnexpectedResult` objects to their corresponding check results."""
        for result in check_results:
            unexpected = unexpected_map.get(result.check_name)
            if unexpected is not None:
                result.unexpected = unexpected

    def _validate_validator_keys(self) -> None:
        """
        Raise :class:`ValueError` if any validator key is absent from *prompts*.

        Skips validation when *prompts* is not yet a dict — this happens when the
        operator is constructed via the ``@task.llm_dq`` decorator and *prompts*
        is set to ``SET_DURING_EXECUTION`` at init time.  The decorator calls this
        method again after the callable has populated ``self.prompts``.
        """
        if not isinstance(self.prompts, dict):
            return
        unknown = set(self.validators.keys()) - set(self.prompts.keys())
        if unknown:
            raise ValueError(
                f"validators contains keys that are not present in prompts: {sorted(unknown)}. "
                "Every validator key must correspond to a prompt key."
            )

    def _validate_prompts(self) -> None:
        """
        Raise :class:`ValueError` if *prompts* is an empty dict.

        Skips validation when *prompts* is not yet a dict (decorator use-case).
        """
        if isinstance(self.prompts, dict) and not self.prompts:
            raise ValueError("prompts must not be empty. Provide at least one check_name → description pair.")

    @cached_property
    def db_hook(self) -> DbApiHook | None:
        """Return a DbApiHook when *db_conn_id* is configured, or ``None``."""
        if not self.db_conn_id:
            return None
        return get_db_hook(self.db_conn_id)

    def execute_complete(
        self, context: Context, generated_output: str, event: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Resume after human approval and execute the data-quality checks.

        Called automatically by Airflow when the HITL trigger fires.  The
        ``generated_output`` is the JSON-serialised
        :class:`~airflow.providers.common.ai.utils.dq_models.DQPlan` that was
        deferred for review.

        :param context: Airflow task context.
        :param generated_output: JSON string of the approved
            :class:`~airflow.providers.common.ai.utils.dq_models.DQPlan`.
        :param event: Trigger event payload from the HITL reviewer.
        :raises HITLRejectException: If the reviewer rejected the plan.
        :raises HITLTimeoutError: If the approval timed out.
        :raises DQCheckFailedError: If any data-quality check fails after approval.
        """
        approved_json = super().execute_complete(context, generated_output, event)
        plan = DQPlan.model_validate_json(approved_json)
        planner = self._build_planner()
        return self._run_checks_and_report(context, planner, plan)


def _compute_plan_hash(
    prompts: dict[str, str],
    prompt_version: str | None,
    collect_unexpected: bool = False,
    row_level_sample_size: int | None = None,
    schema_context: str = "",
    unexpected_sample_size: int = 100,
    validator_contexts: str = "",
    row_validator_thresholds: dict[str, float | str | None] | None = None,
) -> str:
    """
    Return a short, stable hash of the inputs that determine a unique DQ plan.

    Includes *prompts*, *prompt_version*, *collect_unexpected*,
    *row_level_sample_size*, *schema_context*, *unexpected_sample_size*,
    *validator_contexts*, and *row_validator_thresholds*.
    Sorted serialisation ensures the hash is order-independent.
    The result is prefixed with the version tag so cache keys are human-readable.
    """
    payload = json.dumps(sorted(prompts.items()), sort_keys=True)
    if schema_context:
        payload += f":schema={hashlib.sha256(schema_context.encode()).hexdigest()[:16]}"
    if validator_contexts:
        payload += f":validator_contexts={hashlib.sha256(validator_contexts.encode()).hexdigest()[:16]}"
    if row_validator_thresholds:
        payload += ":row_thresholds=" + json.dumps(
            row_validator_thresholds, sort_keys=True, separators=(",", ":")
        )
    if collect_unexpected:
        payload += f":unexpected=1:unexpected_sample={unexpected_sample_size}"
    if row_level_sample_size is not None:
        payload += f":row_sample={row_level_sample_size}"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    version_tag = prompt_version or "default"
    max_tag_len = _PLAN_VARIABLE_KEY_MAX_LEN - len(digest) - 1  # -1 for the "_" separator
    return f"{version_tag[:max_tag_len]}_{digest}"
