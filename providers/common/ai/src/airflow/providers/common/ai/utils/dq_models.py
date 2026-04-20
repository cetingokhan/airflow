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
"""
Pydantic models for the LLM data quality plan and result reporting.

``DQPlan`` is the structured output type requested from the LLM.  The remaining
dataclasses hold execution results and are never serialised back to the model.
``DQCheckInput`` represents a single user-supplied check definition for the operator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, computed_field

from airflow.providers.common.compat.sdk import AirflowException


class DQCheckInput:
    """
    A single user-supplied data-quality check definition.

    The LLM selects the appropriate validator automatically from the registry;
    supply *validator* only when you want to force a specific callable and bypass
    the LLM suggestion entirely.

    :param name: Unique check identifier.  Used as ``check_name`` throughout the
        plan and results.
    :param description: Natural-language description of the expectation.  This is
        what the LLM reads to produce the SQL and select a validator.
    :param validator: Optional fixed validator callable.  When provided the LLM is
        **not** asked to suggest a validator for this check; the supplied callable is
        used directly.  Accepts plain lambdas or any factory-produced callable from
        :mod:`~airflow.providers.common.ai.utils.dq_validation`.
    """

    # Allow Airflow's template engine to render Jinja expressions inside
    # ``name`` and ``description`` when the operator renders template_fields.
    template_fields: tuple[str, ...] = ("name", "description")

    __slots__ = ("description", "name", "validator")

    def __init__(
        self,
        *,
        name: str,
        description: str,
        validator: Callable[[Any], bool] | None = None,
    ) -> None:
        if not name or not name.strip():
            raise ValueError("DQCheckInput.name must not be empty.")
        if not description or not description.strip():
            raise ValueError("DQCheckInput.description must not be empty.")
        self.name = name
        self.description = description
        self.validator = validator

    @classmethod
    def coerce(cls, value: Any) -> DQCheckInput:
        """
        Accept a :class:`DQCheckInput` instance or a plain dict and return a :class:`DQCheckInput`.

        Plain dicts must contain ``name`` and ``description`` keys; ``validator`` is
        optional.

        :raises TypeError: If *value* is neither a :class:`DQCheckInput` nor a ``dict``.
        :raises KeyError: If *value* is a dict missing required keys.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(
                name=value["name"],
                description=value["description"],
                validator=value.get("validator"),
            )
        raise TypeError(
            f"Expected DQCheckInput or dict, got {type(value).__name__!r}. "
            "Each check must have 'name' and 'description' fields."
        )

    def __repr__(self) -> str:
        has_validator = self.validator is not None
        return (
            f"DQCheckInput(name={self.name!r}, description={self.description!r}, validator={has_validator})"
        )


class DQCheck(BaseModel):
    """
    A single data-quality check produced by the LLM.

    :param check_name: Matches the ``name`` supplied by the user in ``checks``.
    :param metric_key: Column alias used in the generated SQL (e.g. ``null_email_count``).
        The operator reads ``row[metric_key]`` from the query result.
    :param group_id: Logical bucket for grouping checks into a single SQL query
        (e.g. ``customers_null_check_1``, ``orders_validity_1``).
    :param check_category: Semantic category assigned by the LLM based on the check
        description.  Used to sub-group checks within a table so that checks of
        different natures (e.g. null-checks vs. regex validity) land in separate
        SQL queries.  Allowed values: ``null_check``, ``uniqueness``, ``validity``,
        ``numeric_range``, ``row_count``, ``string_format``, ``row_level``.
    :param unexpected_query: Optional SELECT query that returns the top N rows
        violating the check condition.  Populated by the LLM only for
        validity / regex / format checks when ``collect_unexpected`` is enabled.
        Pure aggregate checks (null percentage, row count, duplicates) leave
        this as ``None``.
    :param validator_name: Registered validator name chosen by the LLM from the
        available registry.  ``None`` or ``"none"`` when the check has a user-fixed
        validator or when no suitable validator was found.
    :param validator_args: Keyword arguments that the LLM suggests passing to the
        validator factory (e.g. ``{"max_pct": 0.05}``).  Empty dict when no
        validator was selected.
    """

    check_name: str = Field(description="Matches the check name supplied by the user in the checks list.")
    metric_key: str = Field(
        description="Column alias used in the generated SQL (e.g. null_email_count). The operator reads row[metric_key] from the query result."
    )
    group_id: str = Field(
        description="Logical bucket for grouping checks into a single SQL query (e.g. customers_null_check_1, orders_validity_1)."
    )
    check_category: str = Field(
        default="",
        description=(
            "Semantic category of this check. One of: null_check, uniqueness, validity, "
            "numeric_range, row_count, string_format, row_level. Used to sub-group checks within a table "
            "so that checks of different natures land in separate SQL queries."
        ),
    )
    unexpected_query: str | None = Field(
        default=None,
        description=(
            "Optional SELECT query that returns the top N rows violating this check. "
            "Populated only for validity/regex/format checks when unexpected collection "
            "is enabled. Pure aggregate checks leave this as null."
        ),
    )
    row_level: bool = Field(
        default=False,
        description=(
            "When True, the query returns raw column values (no aggregation). "
            "The planner applies the Python-side validator to every returned row "
            "and aggregates the results into a RowLevelResult."
        ),
    )
    validator_name: str | None = Field(
        default=None,
        description=(
            "Registered validator name chosen by the LLM from the available registry. "
            "Set to null when the check has a user-fixed validator or set to 'none' when "
            "no suitable validator was found."
        ),
    )
    validator_args: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Keyword arguments to pass to the validator factory "
            '(e.g. {"max_pct": 0.05}). Empty dict when no validator was selected.'
        ),
    )


class DQCheckGroup(BaseModel):
    """
    A group of related :class:`DQCheck` items that share one SQL query.

    :param group_id: Matches :attr:`DQCheck.group_id` for each member check.
    :param query: A single SELECT statement whose result columns correspond to
        the ``metric_key`` values of all member checks.
    :param checks: The checks whose metrics can be extracted from ``query``'s result.
    """

    group_id: str = Field(description="Matches DQCheck.group_id for each member check.")
    query: str = Field(
        description="A single SELECT statement whose result columns correspond to the metric_key values of all member checks."
    )
    checks: list[DQCheck] = Field(
        description="The checks whose metrics can be extracted from query's result."
    )


class DQPlan(BaseModel):
    """
    Complete data-quality execution plan returned by the LLM.

    :param groups: All SQL check groups.  Together, the checks inside every group
        must cover every check name in the user's ``checks`` list exactly once.
    :param plan_hash: SHA-256 fingerprint of the checks list (set by the operator
        after generation — not populated by the LLM).
    """

    groups: list[DQCheckGroup] = Field(
        description="All SQL check groups. Together, the checks inside every group must cover every check name in the user's checks list exactly once."
    )
    plan_hash: str = Field(
        default="",
        description="SHA-256 fingerprint of the checks list (set by the operator after generation — not populated by the LLM).",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def check_names(self) -> list[str]:
        """Flat list of all ``check_name`` values across every group."""
        return [check.check_name for group in self.groups for check in group.checks]


@dataclass
class RowLevelResult:
    """
    Aggregated outcome of applying a row-level validator to every row.

    Produced by
    :meth:`~airflow.providers.common.ai.utils.dq_planner.SQLDQPlanner._execute_row_level_group`
    when a :class:`DQCheck` has ``row_level=True``.

    :param check_name: Corresponds to the user-supplied check name.
    :param total: Total number of rows evaluated.
    :param invalid: Number of rows for which the validator returned ``False``.
    :param invalid_pct: Fraction of invalid rows (``invalid / total``), or ``0.0``
        when *total* is zero.
    :param sample_violations: Sampled failing values as strings, capped by an
        internal planner limit.
    :param sample_size: Number of sampled failing values (equivalent to
        ``len(sample_violations)``).
    """

    check_name: str
    total: int
    invalid: int
    invalid_pct: float
    sample_violations: list[str]
    sample_size: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of result metrics (excludes ``check_name``)."""
        return {
            "total": self.total,
            "invalid": self.invalid,
            "invalid_pct": self.invalid_pct,
            "sample_violations": self.sample_violations,
            "sample_size": self.sample_size,
        }


@dataclass
class UnexpectedResult:
    """
    Sample of rows that violate a data-quality check.

    Populated only when ``collect_unexpected=True`` and the check has an
    ``unexpected_query`` that returned results.

    :param check_name: Corresponds to the user prompt key.
    :param unexpected_records: Flat list of violating values as strings.
    :param sample_size: Maximum number of rows requested (LIMIT value).
    """

    check_name: str
    unexpected_records: list[str]
    sample_size: int


@dataclass
class DQCheckResult:
    """
    Result of executing and validating a single :class:`DQCheck`.

    :param check_name: Corresponds to the user prompt key.
    :param metric_key: SQL column alias whose value was extracted.
    :param value: Raw value returned by the database for this metric.
    :param passed: ``True`` if every applicable validator approved the value.
    :param failure_reason: Human-readable description of why validation failed,
        or ``None`` when ``passed`` is ``True``.
    :param unexpected: Sample of violating rows when unexpected collection is
        enabled, or ``None``.
    """

    check_name: str
    metric_key: str
    value: Any
    passed: bool
    failure_reason: str | None = None
    unexpected: UnexpectedResult | None = None


@dataclass
class DQReport:
    """
    Aggregated outcome of all data-quality checks in a single operator run.

    :param results: One :class:`DQCheckResult` per prompt key.
    :param passed: ``True`` only when every result passed.
    :param failure_summary: Multi-line string listing every failed check;
        empty when all checks pass.
    """

    results: list[DQCheckResult] = field(default_factory=list)
    passed: bool = True
    failure_summary: str = ""

    @classmethod
    def build(cls, results: list[DQCheckResult]) -> DQReport:
        """Construct a :class:`DQReport` from a list of individual results."""
        failed = [r for r in results if not r.passed]
        if not failed:
            return cls(results=results, passed=True, failure_summary="")

        lines = [f"Data quality checks failed ({len(failed)}/{len(results)}):"]
        for r in failed:
            reason = r.failure_reason or "validator returned False"
            line = f"  - {r.check_name} (metric: {r.metric_key}, value: {r.value!r}): {reason}"
            if r.unexpected and r.unexpected.unexpected_records:
                line += f" [{len(r.unexpected.unexpected_records)} unexpected row(s) sampled]"
            lines.append(line)
        return cls(results=results, passed=False, failure_summary="\n".join(lines))


class DQCheckFailedError(AirflowException):
    """Raised when one or more data-quality checks fail threshold validation."""
