 .. Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

 ..   http://www.apache.org/licenses/LICENSE-2.0

 .. Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. _howto/operator:llm_data_quality:

``LLMDataQualityOperator``
==========================

Use :class:`~airflow.providers.common.ai.operators.llm_data_quality.LLMDataQualityOperator`
to generate and execute data-quality checks from natural language using an LLM.

Each entry in ``checks`` describes **one** data-quality expectation as a
:class:`~airflow.providers.common.ai.utils.dq_models.DQCheckInput` object.
The LLM groups related checks into optimised SQL queries, executes them against the
target database, and validates each metric.  The task fails if any check does not
pass, gating downstream tasks on data quality.

.. seealso::
    :ref:`Connection configuration <howto/connection:pydanticai>`

Basic Usage
-----------

Provide a ``checks`` list and a target ``db_conn_id``. The operator introspects the
schema automatically when ``table_names`` is provided:

.. code-block:: python

    from airflow.providers.common.ai.operators.llm_data_quality import LLMDataQualityOperator
    from airflow.providers.common.ai.utils.dq_models import DQCheckInput
    from airflow.providers.common.ai.utils.dq_validation import null_pct_check, row_count_check

    LLMDataQualityOperator(
        task_id="validate_orders",
        llm_conn_id="pydanticai_default",
        db_conn_id="postgres_default",
        table_names=["orders", "customers"],
        checks=[
            DQCheckInput(
                name="row_count",
                description="The orders table must contain at least 1000 rows.",
                validator=row_count_check(min_count=1000),
            ),
            DQCheckInput(
                name="email_nulls",
                description="No more than 5% of customer email addresses should be null.",
                validator=null_pct_check(max_pct=0.05),
            ),
        ],
    )

Validators
----------

Each :class:`~airflow.providers.common.ai.utils.dq_models.DQCheckInput` can carry
an optional ``validator`` — a callable that receives the raw metric value returned
by the generated SQL and returns ``True`` (pass) or ``False`` (fail).

Built-in Factories
~~~~~~~~~~~~~~~~~~

:mod:`~airflow.providers.common.ai.utils.dq_validation` ships ready-made factories
for the most common thresholds:

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Factory
     - Passes when …
     - Example
   * - ``null_pct_check(max_pct=…)``
     - metric ≤ ``max_pct``
     - ``null_pct_check(max_pct=0.05)``
   * - ``row_count_check(min_count=…)``
     - metric ≥ ``min_count``
     - ``row_count_check(min_count=1000)``
   * - ``duplicate_pct_check(max_pct=…)``
     - metric ≤ ``max_pct``
     - ``duplicate_pct_check(max_pct=0.0)``
   * - ``between_check(min_val=…, max_val=…)``
     - ``min_val`` ≤ metric ≤ ``max_val``
     - ``between_check(min_val=0.0, max_val=1.0)``
   * - ``exact_check(expected=…)``
     - metric == ``expected``
     - ``exact_check(expected=0)``

You can also use plain lambdas for one-off conditions::

    DQCheckInput(
        name="stale_rows",
        description="Count rows older than 30 days",
        validator=lambda v: int(v) < 1000,
    )

Aggregate checks without a validator are marked as **passed** — metrics are
still collected and included in the report, but no threshold is enforced.
Row-level checks are stricter: each row-level check must have a corresponding
row-level validator, otherwise execution fails fast.

Custom Validators with ``register_validator``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the :func:`~airflow.providers.common.ai.utils.dq_validation.register_validator`
decorator to attach an ``llm_context`` hint to your validator factory.  The
operator injects the hint into the LLM system prompt so the model generates SQL
that returns the metric format your validator expects:

.. exampleinclude:: /../../ai/src/airflow/providers/common/ai/example_dags/example_llm_data_quality.py
    :language: python
    :start-after: [START howto_operator_llm_dq_email_format_validator]
    :end-before: [END howto_operator_llm_dq_email_format_validator]

With Schema Introspection
-------------------------

Pass ``db_conn_id`` and ``table_names`` so the operator can introspect the live
database schema. The LLM receives real column names and types, producing more
accurate SQL:

.. code-block:: python

    from airflow.providers.common.ai.operators.llm_data_quality import LLMDataQualityOperator
    from airflow.providers.common.ai.utils.dq_models import DQCheckInput
    from airflow.providers.common.ai.utils.dq_validation import (
        duplicate_pct_check,
        exact_check,
        null_pct_check,
        row_count_check,
    )

    LLMDataQualityOperator(
        task_id="validate_customers",
        llm_conn_id="pydanticai_default",
        db_conn_id="postgres_default",
        table_names=["customers"],
        prompt_version="v1",
        checks=[
            DQCheckInput(
                name="null_emails",
                description="Check the percentage of rows where email is NULL",
                validator=null_pct_check(max_pct=0.01),
            ),
            DQCheckInput(
                name="duplicate_ids",
                description="Calculate the percentage of duplicate customer IDs",
                validator=duplicate_pct_check(max_pct=0.0),
            ),
            DQCheckInput(
                name="negative_balance",
                description="Count rows where account_balance is negative",
                validator=exact_check(expected=0),
            ),
            DQCheckInput(
                name="min_customers",
                description="Count the total number of customer rows",
                validator=row_count_check(min_count=10_000),
            ),
        ],
    )

With Manual Schema Context
~~~~~~~~~~~~~~~~~~~~~~~~~~

When you cannot or do not want to connect to the database at plan-generation
time, pass a manual schema description via ``schema_context``:

.. code-block:: python

    LLMDataQualityOperator(
        task_id="validate_with_manual_schema",
        llm_conn_id="pydanticai_default",
        db_conn_id="postgres_default",
        schema_context=(
            "Table: orders(id INT, customer_id INT, amount DECIMAL, created_at TIMESTAMP)\n"
            "Table: customers(id INT, email TEXT, country TEXT)"
        ),
        checks=[
            DQCheckInput(
                name="null_amount",
                description="Check the percentage of orders with a NULL amount",
                validator=null_pct_check(max_pct=0.0),
            ),
        ],
    )

With Object Storage
-------------------

Use ``datasource_config`` to validate data stored in object storage
(e.g., Amazon S3, Azure Blob Storage) via
`DataFusion <https://datafusion.apache.org/>`_.
The operator uses :class:`~airflow.providers.common.sql.config.DataSourceConfig`
to register the object storage source as a table so the LLM can incorporate it
in the schema context.

.. exampleinclude:: /../../ai/src/airflow/providers/common/ai/example_dags/example_llm_data_quality.py
    :language: python
    :start-after: [START howto_operator_llm_dq_datafusion_classic]
    :end-before: [END howto_operator_llm_dq_datafusion_classic]

Row-Level Checks
----------------

Row-level validation lets you apply a Python predicate **per row** rather than
against a single aggregate metric.  Instead of executing a ``COUNT`` or
``AVG``, the LLM generates a plain ``SELECT`` statement and the operator
fetches the result set, applies your validator to every value in turn, and
aggregates the outcomes into a
:class:`~airflow.providers.common.ai.utils.dq_models.RowLevelResult`.

To enable row-level mode for a validator, register it with ``row_level=True``
via the :func:`~airflow.providers.common.ai.utils.dq_validation.register_validator`
decorator::

    @register_validator(
        "my_format_check",
        llm_context="ROW-LEVEL: SELECT id, col FROM table — no aggregation.",
        check_category="row_level",
        row_level=True,
    )
    def my_format_check(*, max_invalid_pct: float = 0.0):
        def _check(value):
            ...  # return True (valid) or False (invalid) for each row value
        return _check

The ``llm_context`` hint tells the LLM to generate a non-aggregating
``SELECT`` covering every row.  The ``row_level=True`` flag marks the
validator in the registry so the planner executes it in row-level mode.

``RowLevelResult`` output
~~~~~~~~~~~~~~~~~~~~~~~~~

For row-level checks, the ``value`` field in the output results dict is a
serialised :class:`~airflow.providers.common.ai.utils.dq_models.RowLevelResult`
dict with the following keys:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Type
     - Description
   * - ``total``
     - ``int``
     - Total number of rows evaluated.
   * - ``invalid``
     - ``int``
     - Number of rows for which the validator returned ``False``.
   * - ``invalid_pct``
     - ``float``
     - Fraction of invalid rows (``invalid / total``), or ``0.0`` when *total*
       is zero.
   * - ``sample_violations``
     - ``list[str]``
     - String representations of sampled failing values (capped internally).
   * - ``sample_size``
     - ``int``
     - Number of returned sampled violations (``len(sample_violations)``).

The check **passes** when ``invalid_pct ≤ max_invalid_pct`` (the
``_max_invalid_pct`` attribute on the validator callable, defaulting to
``0.0``).

If a row-level check is present in the generated plan but no matching
row-level validator is configured, the operator raises ``ValueError`` and
stops execution.

``row_level_sample_size``
~~~~~~~~~~~~~~~~~~~~~~~~~

By default the operator fetches every row for row-level checks (full table
scan).  Set ``row_level_sample_size`` to a positive integer to add a ``LIMIT``
to the generated ``SELECT``, bounding execution time and memory usage at the
cost of sampling coverage:

.. code-block:: python

    LLMDataQualityOperator(
        task_id="validate_email_format", row_level_sample_size=100_000, ...  # sample up to 100 k rows per check
    )

Example
~~~~~~~

.. exampleinclude:: /../../ai/src/airflow/providers/common/ai/example_dags/example_llm_data_quality.py
    :language: python
    :start-after: [START howto_operator_llm_dq_custom_row_level]
    :end-before: [END howto_operator_llm_dq_custom_row_level]

Plan Caching
------------

Generated SQL plans are cached in Airflow
:class:`~airflow.models.variable.Variable` to avoid repeat LLM calls on every
DAG run.

Cache key
~~~~~~~~~

The cache key is derived from a SHA-256 digest of the sorted checks list
plus all planning-affecting inputs: ``prompt_version``, schema context,
unexpected-row collection settings, row-level sample size, validator LLM
context, and row-level validator thresholds. The key format is::

    dq_plan_{version_tag}_{sha256[:16]}

Because it is order-independent, reordering checks does **not**
invalidate the cache.

Invalidating the Cache
~~~~~~~~~~~~~~~~~~~~~~

Bump ``prompt_version`` whenever the intent of a check changes, even if the
text is the same:

.. code-block:: python

    from airflow.providers.common.ai.utils.dq_models import DQCheckInput
    from airflow.providers.common.ai.utils.dq_validation import row_count_check

    LLMDataQualityOperator(
        task_id="validate_orders",
        prompt_version="v2",  # was "v1" — forces a new LLM call
        llm_conn_id="pydanticai_default",
        db_conn_id="postgres_default",
        checks=[
            DQCheckInput(
                name="row_count",
                description="The orders table must contain at least 5000 rows.",
                validator=row_count_check(min_count=5000),
            ),
        ],
    )

Dry Run Mode
------------

Set ``dry_run=True`` to generate and cache the SQL plan without executing it.
The operator returns the serialised plan dict, which you can inspect or log
before running against production data:

.. exampleinclude:: /../../ai/src/airflow/providers/common/ai/example_dags/example_llm_data_quality.py
    :language: python
    :start-after: [START howto_operator_llm_dq_dry_run_preview]
    :end-before: [END howto_operator_llm_dq_dry_run_preview]

You can also combine ``dry_run=True`` with
:class:`~airflow.providers.standard.operators.hitl.ApprovalOperator` to build
an explicit three-step review pipeline where a human approves the generated
SQL before it is executed:

.. exampleinclude:: /../../ai/src/airflow/providers/common/ai/example_dags/example_llm_data_quality.py
    :language: python
    :start-after: [START howto_operator_llm_dq_with_human_approval]
    :end-before: [END howto_operator_llm_dq_with_human_approval]

Because the ``dry_run`` run already cached the plan under the same key, the
execution run skips the LLM entirely and reads from the
:class:`~airflow.models.variable.Variable` cache.

Human-in-the-Loop Approval
--------------------------

Set ``require_approval=True`` to gate execution on human review.  When the
plan has been generated and cached, the task defers and presents a formatted
SQL-plan markdown to the reviewer.  Data-quality checks only run **after** the
reviewer approves.  If the reviewer rejects, the task raises
``HITLRejectException`` without executing any SQL.

``dry_run`` and ``require_approval`` are independent:

- ``dry_run=True`` — return the plan dict immediately, no approval prompt,
  no SQL execution.
- ``require_approval=True`` — generate the plan, defer for human review,
  then execute checks after approval.
- Both ``True`` — behaves as ``dry_run=True``; the approval prompt is
  suppressed.

.. code-block:: python

    from airflow.providers.common.ai.utils.dq_models import DQCheckInput
    from airflow.providers.common.ai.utils.dq_validation import row_count_check

    LLMDataQualityOperator(
        task_id="validate_with_approval",
        llm_conn_id="pydanticai_default",
        db_conn_id="postgres_default",
        table_names=["orders"],
        checks=[
            DQCheckInput(
                name="row_count",
                description="The orders table must have at least 1000 rows.",
                validator=row_count_check(min_count=1000),
            ),
        ],
        require_approval=True,
        allow_modifications=False,
    )

A full runnable example:

.. exampleinclude:: /../../ai/src/airflow/providers/common/ai/example_dags/example_llm_data_quality.py
    :language: python
    :start-after: [START howto_operator_llm_dq_simple_builtin_approval]
    :end-before: [END howto_operator_llm_dq_simple_builtin_approval]

See :class:`~airflow.providers.common.ai.operators.llm.LLMOperator` for the
full set of HITL parameters (``require_approval``, ``approval_timeout``,
``allow_modifications``).

TaskFlow Decorator
------------------

The ``@task.llm_dq`` decorator lets you write a function that returns a
``list[DQCheckInput]``. The decorator handles LLM plan generation, plan caching,
SQL execution, and metric validation:

.. code-block:: python

    from airflow.providers.common.ai.utils.dq_models import DQCheckInput
    from airflow.providers.common.ai.utils.dq_validation import (
        null_pct_check,
        row_count_check,
    )
    from airflow.sdk import dag, task


    @dag
    def validate_pipeline():
        @task.llm_dq(
            llm_conn_id="pydanticai_default",
            db_conn_id="postgres_default",
            table_names=["orders", "customers"],
        )
        def dq_checks(ds=None):
            return [
                DQCheckInput(
                    name="row_count",
                    description=f"The orders table must have at least 1000 rows as of {ds}.",
                    validator=row_count_check(min_count=1000),
                ),
                DQCheckInput(
                    name="email_nulls",
                    description="No more than 5% of customer emails should be null.",
                    validator=null_pct_check(max_pct=0.05),
                ),
            ]

        dq_checks()


    validate_pipeline()

The function body can use Airflow context variables, XCom pulls, or any
runtime information to build the checks list dynamically.

Logging
-------

The operator and its underlying planner log the following:

- The schema context used (at INFO level).
- Whether the plan was loaded from cache or freshly generated (at INFO level).
- LLM run summary after plan generation: model name, token usage (input,
  output, total), request count, and tool call sequence (at INFO level).
- Per-group SQL queries and check names in ``dry_run`` mode (at INFO level).
- The number of checks that passed after execution (at INFO level).
- A failure summary raised as :class:`~airflow.exceptions.AirflowException`
  when any check fails, listing each failing check name and reason.

See :ref:`AgentOperator — Logging <howto/operator:agent>` for details on the
underlying LLM call log format (model name, token usage, request count).
