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

import logging
import re

from airflow.providers.common.ai.operators.llm_data_quality import LLMDataQualityOperator
from airflow.providers.common.ai.utils.dq_models import DQCheckInput
from airflow.providers.common.ai.utils.dq_validation import (
    exact_check,
    null_pct_check,
    register_validator,
)
from airflow.providers.common.compat.sdk import dag, task
from airflow.providers.common.sql.config import DataSourceConfig

# ------------------------------------------------------------------
# Module-level custom row-level validator used across multiple DAGs
# ------------------------------------------------------------------


# [START howto_operator_llm_dq_email_format_validator]
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


# [END howto_operator_llm_dq_email_format_validator]


# [START howto_operator_llm_dq_datafusion_classic]
@dag
def example_llm_dq_datafusion_classic():
    """
    Run data-quality checks on a DataFusion (S3 Parquet) dataset using the classic operator.

    ``null_session_id`` and ``duplicate_sessions`` have no fixed validator; the LLM
    selects an appropriate validator from the built-in registry automatically.
    ``invalid_click_count`` uses a fixed ``exact_check`` validator so the LLM is
    not asked to suggest one for that check.

    Connections required:

    - ``pydanticai_default``: Pydantic AI connection (Bedrock, OpenAI, etc.).
    - ``aws_default``: AWS connection with S3 read permissions.
    """
    datasource_config = DataSourceConfig(
        conn_id="aws_default",
        table_name="web_clicks",
        uri="s3://my-data-lake/web/clicks/year=2025/",
        format="parquet",
    )

    LLMDataQualityOperator(
        task_id="validate_web_clicks",
        llm_conn_id="pydanticai_default",
        datasource_config=datasource_config,
        prompt_version="v1",
        checks=[
            DQCheckInput(
                name="null_session_id",
                description="Check the percentage of rows where session_id is NULL",
            ),
            DQCheckInput(
                name="invalid_click_count",
                description="Count rows where click_count is negative",
                validator=exact_check(expected=0),
            ),
            DQCheckInput(
                name="duplicate_sessions",
                description="Calculate the percentage of duplicate session_id values",
            ),
        ],
    )


# [END howto_operator_llm_dq_datafusion_classic]

example_llm_dq_datafusion_classic()


# [START howto_operator_llm_dq_postgres_seed_classic]
@dag
def example_llm_dq_postgres_seed_classic():
    """
    Run data-quality checks on seeded PostgreSQL tables (``orders`` and ``customers``).

    The tables are defined in ``001_schema_and_seed.sql``.  Most checks have no
    fixed validator — the LLM selects an appropriate one from the built-in registry.
    ``negative_amount`` uses a plain ``lambda`` as a lightweight one-off validator.
    ``customer_email_format`` is a row-level check: the planner fetches each row
    and applies the ``email_format_check`` regex per value, aggregating the outcome
    into a :class:`~airflow.providers.common.ai.utils.dq_models.RowLevelResult`.

    Connections required:

    - ``pydanticai_default``: Pydantic AI connection.
    - ``postgres_default``: PostgreSQL connection pointing at the seeded database.
    """
    LLMDataQualityOperator(
        task_id="validate_orders_and_customers",
        llm_conn_id="pydanticai_default",
        db_conn_id="postgres_default",
        table_names=["orders", "customers"],
        prompt_version="v1",
        row_level_sample_size=50_000,
        checks=[
            DQCheckInput(
                name="null_order_id",
                description="Check the percentage of rows in orders where order_id is NULL",
            ),
            DQCheckInput(
                name="duplicate_orders",
                description="Calculate the percentage of duplicate order_id values in orders",
            ),
            DQCheckInput(
                name="null_customer_email",
                description="Check the percentage of rows in customers where email is NULL",
            ),
            DQCheckInput(
                name="min_order_count",
                description="Count the total number of rows in the orders table",
            ),
            DQCheckInput(
                name="negative_amount",
                description="Count rows in orders where order_amount is negative or NULL",
                validator=lambda v: int(v) == 0,
            ),
            DQCheckInput(
                name="customer_email_format",
                description="Validate that each email in the customers table matches a valid email format",
                validator=email_format_check(max_invalid_pct=0.01),
            ),
        ],
    )


# [END howto_operator_llm_dq_postgres_seed_classic]

example_llm_dq_postgres_seed_classic()


# [START howto_operator_llm_dq_task_decorator]
@dag
def example_llm_dq_task_decorator():
    """
    Combine a ``@task``-decorated step with a ``@task.llm_dq``-decorated DQ task.

    ``notify_start`` is a plain ``@task`` that runs first to signal an upstream
    condition (e.g. loading configuration or triggering an upstream pipeline).
    ``validate_users`` is decorated with ``@task.llm_dq``: the function body
    returns the checks list, and the decorator handles LLM plan generation, plan
    caching, SQL execution, and metric validation.  Both checks have no fixed
    validator — the LLM selects appropriate ones from the built-in registry.

    Connections required:

    - ``pydanticai_default``: Pydantic AI connection.
    - ``postgres_default``: PostgreSQL connection for the ``users`` table.
    """

    @task
    def notify_start() -> dict:
        """Log the start of the validation run and return a run metadata dict."""
        log = logging.getLogger(__name__)
        log.info("Starting DQ validation for the users table.")
        return {"status": "started"}

    @task.llm_dq(
        llm_conn_id="pydanticai_default",
        db_conn_id="postgres_default",
        table_names=["users"],
        prompt_version="v1",
    )
    def validate_users():
        return [
            DQCheckInput(
                name="null_username",
                description="Check the percentage of rows where username is NULL",
            ),
            DQCheckInput(
                name="null_email",
                description="Check the percentage of rows where email is NULL",
            ),
        ]

    notify_start() >> validate_users()


# [END howto_operator_llm_dq_task_decorator]

example_llm_dq_task_decorator()


# [START howto_operator_llm_dq_simple_builtin_approval]
@dag
def example_llm_dq_simple_builtin_approval():
    """
    Gate DQ check execution on human review using the built-in HITL mechanism.

    ``require_approval=True`` defers the task after plan generation and surfaces
    the generated SQL plan to a human reviewer in the Airflow UI.  The two checks
    have no fixed validator — the LLM selects appropriate ones automatically.
    Checks run only after the reviewer approves.

    Connections required:

    - ``pydanticai_default``: Pydantic AI connection.
    - ``postgres_default``: PostgreSQL connection for the ``products`` table.
    """
    LLMDataQualityOperator(
        task_id="validate_products_with_approval",
        llm_conn_id="pydanticai_default",
        db_conn_id="postgres_default",
        table_names=["products"],
        require_approval=True,
        prompt_version="v1",
        checks=[
            {"name": "null_sku", "description": "Check the percentage of rows where sku is NULL"},
            {"name": "negative_price", "description": "Count rows where price is negative or NULL"},
        ],
    )


# [END howto_operator_llm_dq_simple_builtin_approval]

example_llm_dq_simple_builtin_approval()


# [START howto_operator_llm_dq_dry_run_preview]
@dag
def example_llm_dq_dry_run_preview():
    """
    Preview the generated SQL plan without executing any checks.

    ``dry_run=True`` generates and caches the DQ plan via the LLM but skips all
    SQL execution.  The serialised plan dict (containing SQL queries, check names,
    and metric keys) is returned via XCom so you can inspect the generated SQL
    before running it against production data.

    Connections required:

    - ``pydanticai_default``: Pydantic AI connection.
    - ``postgres_default``: PostgreSQL connection (schema introspection only;
      no data is read).
    """
    LLMDataQualityOperator(
        task_id="preview_orders_plan",
        llm_conn_id="pydanticai_default",
        db_conn_id="postgres_default",
        table_names=["orders"],
        dry_run=True,
        prompt_version="v1",
        checks=[
            DQCheckInput(
                name="null_order_id",
                description="Check the percentage of rows where order_id is NULL",
            ),
            DQCheckInput(
                name="min_order_count",
                description="Count the total number of rows in the orders table",
            ),
        ],
    )


# [END howto_operator_llm_dq_dry_run_preview]

example_llm_dq_dry_run_preview()


# [START howto_operator_llm_dq_with_human_approval]
@dag
def example_llm_dq_with_human_approval():
    """
    Three-step pipeline: dry-run plan generation → human approval → execution.

    The first task generates and caches the SQL plan with ``dry_run=True``.
    An ``ApprovalOperator`` gates execution so a reviewer can inspect the
    generated SQL before it touches production data.  The second
    ``LLMDataQualityOperator`` reuses the cached plan (same ``prompt_version``
    and checks) and executes the actual data-quality queries.

    Connections required:

    - ``pydanticai_default``: Pydantic AI connection.
    - ``postgres_default``: PostgreSQL connection for the ``orders`` table.
    """
    from airflow.providers.standard.operators.hitl import ApprovalOperator

    checks = [
        DQCheckInput(
            name="null_order_id",
            description="Check the percentage of rows where order_id is NULL",
        ),
        DQCheckInput(
            name="min_order_count",
            description="Count the total number of rows in the orders table",
        ),
    ]

    generate_plan = LLMDataQualityOperator(
        task_id="generate_plan",
        llm_conn_id="pydanticai_default",
        db_conn_id="postgres_default",
        table_names=["orders"],
        dry_run=True,
        prompt_version="v1",
        checks=checks,
    )

    approve = ApprovalOperator(task_id="approve_plan")

    execute_checks = LLMDataQualityOperator(
        task_id="execute_checks",
        llm_conn_id="pydanticai_default",
        db_conn_id="postgres_default",
        table_names=["orders"],
        prompt_version="v1",
        checks=checks,
    )

    generate_plan >> approve >> execute_checks


# [END howto_operator_llm_dq_with_human_approval]

example_llm_dq_with_human_approval()


# [START howto_operator_llm_dq_custom_row_level]
@dag
def example_llm_dq_custom_row_level():
    """
    Run data-quality checks that include a custom row-level validator.

    ``email_format_check`` is registered at module level with ``row_level=True``,
    which instructs the LLM to generate a plain ``SELECT id, email FROM table``
    query without aggregation.  The planner fetches up to ``row_level_sample_size``
    rows and applies the regex validator per value, aggregating the result into a
    :class:`~airflow.providers.common.ai.utils.dq_models.RowLevelResult`.

    ``null_email`` uses a fixed ``null_pct_check`` validator.
    ``min_customer_count`` has no fixed validator — the LLM selects one.

    Use this pattern whenever per-row domain-specific validation logic cannot be
    expressed as a single SQL aggregate metric.

    Connections required:

    - ``pydanticai_default``: Pydantic AI connection.
    - ``postgres_default``: PostgreSQL connection for the ``customers`` table.
    """
    LLMDataQualityOperator(
        task_id="validate_customer_emails",
        llm_conn_id="pydanticai_default",
        db_conn_id="postgres_default",
        table_names=["customers"],
        prompt_version="v1",
        row_level_sample_size=100_000,
        checks=[
            DQCheckInput(
                name="email_format",
                description="Validate that each email in the customers table is a properly formatted email address",
                validator=email_format_check(max_invalid_pct=0.01),
            ),
            DQCheckInput(
                name="null_email",
                description="Check the percentage of rows where email is NULL",
                validator=null_pct_check(max_pct=0.01),
            ),
            DQCheckInput(
                name="min_customer_count",
                description="Count the total number of rows in the customers table",
            ),
        ],
    )


# [END howto_operator_llm_dq_custom_row_level]

example_llm_dq_custom_row_level()
