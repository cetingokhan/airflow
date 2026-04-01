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
"""Example DAGs demonstrating LLMDataQualityOperator usage."""

from __future__ import annotations

import re

from airflow.providers.common.ai.operators.llm_data_quality import LLMDataQualityOperator
from airflow.providers.common.ai.utils.dq_validation import (
    duplicate_pct_check,
    exact_check,
    null_pct_check,
    register_validator,
    row_count_check,
)
from airflow.providers.common.compat.sdk import dag
from airflow.providers.common.sql.config import DataSourceConfig
from airflow.providers.standard.operators.hitl import ApprovalOperator


# [START howto_operator_llm_dq_s3_parquet]
@dag
def example_llm_dq_s3_parquet():
    """
    Run data-quality checks on a Parquet dataset stored in Amazon S3.

    DataFusion is used to register the S3 source so the LLM can introspect
    its schema and generate the appropriate SQL.

    Connections required:
    - ``pydanticai_default``: Pydantic AI connection with Bedrock.
    - ``aws_default``: AWS connection with S3 read permissions.
    """
    datasource_config = DataSourceConfig(
        conn_id="aws_default",
        table_name="sales_events",
        uri="s3://my-data-lake/events/sales/year=2025/",
        format="parquet",
    )

    LLMDataQualityOperator(
        task_id="validate_sales_events",
        llm_conn_id="pydanticai_default",
        datasource_config=datasource_config,
        prompt_version="v1",
        prompts={
            "null_event_id": "Check the percentage of rows where event_id is NULL",
            "invalid_revenue": "Count rows where revenue is negative or NULL",
            "stale_data": "Count rows where event_timestamp is older than 90 days",
            "duplicate_events": "Calculate the percentage of duplicate event_id values",
            "min_row_count": "Count the total number of rows in the dataset",
        },
        validators={
            "null_event_id": null_pct_check(max_pct=0.0),
            "invalid_revenue": exact_check(expected=0),
            "stale_data": lambda v: int(v) < 1000,
            "duplicate_events": duplicate_pct_check(max_pct=0.0),
            "min_row_count": row_count_check(min_count=1_000_000),
        },
    )


# [END howto_operator_llm_dq_s3_parquet]

example_llm_dq_s3_parquet()


_DQ_PROMPTS = {
    "null_order_id": "Check the percentage of rows where order_id is NULL",
    "negative_amount": "Count rows where order_amount is negative or NULL",
    "duplicate_orders": "Calculate the percentage of duplicate order_id values",
    "min_row_count": "Count the total number of rows in the orders table",
}

_DQ_VALIDATORS = {
    "null_order_id": null_pct_check(max_pct=0.0),
    "negative_amount": exact_check(expected=0),
    "duplicate_orders": duplicate_pct_check(max_pct=0.0),
    "min_row_count": row_count_check(min_count=10_000),
}

_DQ_COMMON_KWARGS = dict(
    llm_conn_id="pydanticai_default",
    db_conn_id="postgres_default",
    table_names=["orders"],
    prompt_version="v1",
    prompts=_DQ_PROMPTS,
    collect_unexpected=True,
)


# [START howto_operator_llm_dq_with_human_approval]
@dag
def example_llm_dq_with_approval():
    """
    Generate a DQ plan, let a human review it, then execute the checks.

    Workflow:

    - ``preview_dq_plan`` runs with ``dry_run=True`` to generate (and cache) the SQL
      plan via the LLM, but does **not** execute it. It returns the serialised
      plan dict via XCom so the approver can review the generated SQL.
    - ``approve_dq_plan`` pauses the DAG and shows the generated SQL to the user.
      Selecting "Approve" proceeds to execution; selecting "Reject" skips it.
    - ``run_dq_checks`` executes the cached plan against the database and validates
      each metric. Because the plan was already cached by ``preview_dq_plan``, the
      LLM is **not** called a second time.

    Connections required:

    - ``pydanticai_default``: Pydantic AI connection (any supported LLM provider).
    - ``postgres_default``: PostgreSQL connection for the target database.
    """
    preview_dq_plan = LLMDataQualityOperator(
        task_id="preview_dq_plan",
        dry_run=True,
        **_DQ_COMMON_KWARGS,
    )

    approve_dq_plan = ApprovalOperator(
        task_id="approve_dq_plan",
        subject="Review the generated DQ plan before execution",
        body=(
            "The LLM generated the following SQL plan for the data-quality checks.\n"
            "Please review it and approve or reject.\n\n"
            "Plan:\n\n"
            "```json\n"
            "{{ ti.xcom_pull(task_ids='preview_dq_plan') | tojson(indent=2) }}\n"
            "```"
        ),
    )

    run_dq_checks = LLMDataQualityOperator(
        task_id="run_dq_checks",
        validators=_DQ_VALIDATORS,
        **_DQ_COMMON_KWARGS,
    )

    preview_dq_plan >> approve_dq_plan >> run_dq_checks


# [END howto_operator_llm_dq_with_human_approval]

example_llm_dq_with_approval()


# ------------------------------------------------------------------
# Custom validator with LLM context
# ------------------------------------------------------------------


# [START howto_operator_llm_dq_custom_validator]
@register_validator(
    "freshness_check",
    llm_context=(
        "Compute hours elapsed since the most recent row in the table. "
        "SQL pattern: EXTRACT(EPOCH FROM (NOW() - MAX(timestamp_col))) / 3600.0. "
        "Returns a DOUBLE representing hours elapsed."
    ),
    check_category="freshness",
)
def freshness_check(*, max_hours: float):
    """Return a validator that passes when the metric value is at most *max_hours*."""

    def _check(value):
        try:
            return float(value) <= max_hours
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"freshness_check(max_hours={max_hours!r}): expected a numeric value, got {value!r}"
            ) from exc

    return _check


@dag
def example_llm_dq_custom_validators():
    """
    Run data-quality checks using both built-in and custom validators.

    Demonstrates the ``register_validator`` decorator which attaches
    ``llm_context`` to a custom validator factory.  The operator
    automatically injects the context into the LLM system prompt so the
    model produces SQL that returns the metric format the validator expects.

    Connections required:

    - ``pydanticai_default``: Pydantic AI connection (any supported LLM provider).
    - ``postgres_default``: PostgreSQL connection for the target database.
    """
    LLMDataQualityOperator(
        task_id="validate_orders_with_custom_checks",
        llm_conn_id="pydanticai_default",
        db_conn_id="postgres_default",
        table_names=["orders"],
        prompt_version="v2",
        prompts={
            "null_order_id": "Check the percentage of rows where order_id is NULL",
            "min_row_count": "Count the total number of rows in the orders table",
            "order_freshness": "Check how many hours since the last order was placed",
        },
        validators={
            "null_order_id": null_pct_check(max_pct=0.01),
            "min_row_count": row_count_check(min_count=10_000),
            "order_freshness": freshness_check(max_hours=24.0),
        },
    )


# [END howto_operator_llm_dq_custom_validator]

example_llm_dq_custom_validators()


# [START howto_operator_llm_dq_dry_run_standalone]
@dag
def example_llm_dq_dry_run_standalone():
    """
    Preview the generated SQL plan without executing any checks.

    ``dry_run=True`` generates and caches the DQ plan via the LLM but skips
    all SQL execution.  The serialised plan dict (containing SQL queries,
    check names, and metric keys) is returned as XCom so you can inspect
    the generated queries before running them against production data.

    Connections required:

    - ``pydanticai_default``: Pydantic AI connection (any supported LLM provider).
    - ``postgres_default``: PostgreSQL connection (schema introspection only;
      no data is read).
    """
    LLMDataQualityOperator(
        task_id="preview_products_plan",
        llm_conn_id="pydanticai_default",
        db_conn_id="postgres_default",
        table_names=["products"],
        dry_run=True,
        prompt_version="v1",
        prompts={
            "null_sku": "Check the percentage of rows where sku is NULL",
            "negative_price": "Count rows where price is negative or NULL",
            "dup_sku": "Calculate the percentage of duplicate sku values",
        },
        validators={
            "null_sku": null_pct_check(max_pct=0.0),
            "negative_price": exact_check(expected=0),
            "dup_sku": duplicate_pct_check(max_pct=0.0),
        },
    )


# [END howto_operator_llm_dq_dry_run_standalone]

example_llm_dq_dry_run_standalone()


# [START howto_operator_llm_dq_require_approval_builtin]
@dag
def example_llm_dq_require_approval_builtin():
    """
    Gate SQL execution on human review using the built-in HITL mechanism.

    Unlike the manual three-step flow (``dry_run`` → ``ApprovalOperator`` →
    execute), ``require_approval=True`` keeps human-in-the-loop entirely
    within a single operator.  After generating and caching the DQ plan the
    task defers; checks run only after the reviewer approves via the
    Airflow UI.  The plan SQL is shown to the reviewer in a structured
    markdown summary.

    Connections required:

    - ``pydanticai_default``: Pydantic AI connection (any supported LLM provider).
    - ``postgres_default``: PostgreSQL connection for the target database.
    """
    LLMDataQualityOperator(
        task_id="validate_users_with_approval",
        llm_conn_id="pydanticai_default",
        db_conn_id="postgres_default",
        table_names=["users"],
        require_approval=True,
        prompt_version="v1",
        prompts={
            "null_username": "Check the percentage of rows where username is NULL",
            "null_email": "Check the percentage of rows where email is NULL",
            "dup_email": "Calculate the percentage of duplicate email values",
            "min_users": "Count the total number of rows in the users table",
        },
        validators={
            "null_username": null_pct_check(max_pct=0.0),
            "null_email": null_pct_check(max_pct=0.0),
            "dup_email": duplicate_pct_check(max_pct=0.0),
            "min_users": row_count_check(min_count=1000),
        },
    )


# [END howto_operator_llm_dq_require_approval_builtin]

example_llm_dq_require_approval_builtin()


# [START howto_operator_llm_dq_custom_row_level_validator]
@register_validator(
    "email_format_check",
    llm_context=(
        "ROW-LEVEL: SELECT id, email FROM table — no aggregation, no WHERE filter. "
        "Returns one row per record.  The Python validator checks each email value "
        "against a basic format regex (local-part @ domain . tld)."
    ),
    check_category="row_level",
    row_level=True,
)
def email_format_check(*, max_invalid_pct: float = 0.0):
    """Row-level validator that checks email addresses against a basic format regex."""
    _pattern = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

    def _check(value):
        if value is None:
            return False
        return bool(_pattern.match(str(value)))

    return _check


@dag
def example_llm_dq_custom_row_level_validator():
    """
    Apply a custom row-level validator to verify email address formatting.

    ``email_format_check`` is registered with ``row_level=True``, which tells
    the LLM to generate a plain ``SELECT id, email FROM table`` query with no
    aggregation.  The planner fetches every row (or up to
    ``row_level_sample_size`` rows), applies the regex validator per value,
    and aggregates the results into a
    :class:`~airflow.providers.common.ai.utils.dq_models.RowLevelResult`.

    Use this pattern for any domain-specific, per-row validation logic that
    cannot be expressed as a single SQL aggregate metric.

    Connections required:

    - ``pydanticai_default``: Pydantic AI connection (any supported LLM provider).
    - ``postgres_default``: PostgreSQL connection for the target database.
    """
    LLMDataQualityOperator(
        task_id="validate_email_format",
        llm_conn_id="pydanticai_default",
        db_conn_id="postgres_default",
        table_names=["customers"],
        prompt_version="v1",
        row_level_sample_size=100_000,
        prompts={
            "email_format": "Validate that the email column contains properly formatted email addresses",
            "null_email": "Check the percentage of rows where email is NULL",
        },
        validators={
            "email_format": email_format_check(max_invalid_pct=0.01),
            "null_email": null_pct_check(max_pct=0.01),
        },
    )


# [END howto_operator_llm_dq_custom_row_level_validator]

example_llm_dq_custom_row_level_validator()
