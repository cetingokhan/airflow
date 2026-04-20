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
Built-in and custom validator factories for :class:`~airflow.providers.common.ai.operators.llm_data_quality.LLMDataQualityOperator`.

Each factory returns a ``Callable[[Any], bool]`` and can be passed as the
``validator`` argument of a :class:`~airflow.providers.common.ai.utils.dq_models.DQCheckInput`
to force a specific validator for that check, bypassing LLM selection.
Factories are intentionally decoupled from the operator so they can be tested and
composed independently.

Custom validators registered with :func:`register_validator` are exposed to the LLM
via the validator catalog so the model can select them automatically.

Usage::

    from airflow.providers.common.ai.utils.dq_models import DQCheckInput
    from airflow.providers.common.ai.utils.dq_validation import (
        null_pct_check,
        row_count_check,
        register_validator,
    )

    # Fixed validators — LLM is not asked to select a validator for these checks.
    checks = [
        DQCheckInput(
            name="email_nulls",
            description="Check for null emails",
            validator=null_pct_check(max_pct=0.05),
        ),
        DQCheckInput(
            name="min_customers",
            description="Ensure at least 1000 rows",
            validator=row_count_check(min_count=1000),
        ),
    ]


    # Custom validator with LLM context — LLM can select this automatically.
    @register_validator(
        "freshness_check",
        llm_context=(
            "Compute hours since the most recent row. "
            "SQL pattern: EXTRACT(EPOCH FROM (NOW() - MAX(ts_col))) / 3600.0. "
            "Returns a DOUBLE representing hours elapsed."
        ),
        check_category="freshness",
    )
    def freshness_check(*, max_hours: float):
        def _check(value):
            return float(value) <= max_hours

        return _check
"""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidatorEntry:
    """
    Metadata for a registered validator factory.

    :param factory: Callable that returns a ``Callable[[Any], bool]`` validator.
    :param llm_context: Optional hint injected into the LLM system prompt so
        the model knows what SQL metric format this validator expects.
    :param check_category: Optional custom check category.  When set, the LLM
        is instructed to use this category for grouping.
    :param row_level: When ``True`` the LLM is instructed to generate a plain
        ``SELECT pk, col FROM table`` (no aggregation).  The planner fetches
        every row and applies the validator callable to each column value,
        then reports ``{total, invalid, invalid_pct, sample_violations}``.
    """

    factory: Callable[..., Callable[[Any], bool]]
    llm_context: str = ""
    check_category: str = ""
    row_level: bool = False


class ValidatorRegistry:
    """
    Registry for reusable validator factories with optional LLM context.

    Validators registered here can carry an ``llm_context`` string that the
    operator automatically injects into the LLM system prompt, guiding the
    model to produce SQL that returns the metric format the validator expects.

    A module-level :data:`default_registry` instance is available.  Use the
    convenience decorator :func:`register_validator` to register into it.
    """

    def __init__(self) -> None:
        self._entries: dict[str, ValidatorEntry] = {}

    def register(
        self,
        name: str,
        *,
        llm_context: str = "",
        check_category: str = "",
        row_level: bool = False,
    ) -> Callable[[Callable[..., Callable[[Any], bool]]], Callable[..., Callable[[Any], bool]]]:
        """
        Return a decorator that registers a validator factory under *name*.

        :param name: Unique name for this validator.
        :param llm_context: SQL generation hint injected into the LLM prompt.
        :param check_category: Custom check category for LLM grouping.
        :param row_level: When ``True``, the LLM generates a plain SELECT
            returning raw row values instead of an aggregate query.  The
            planner applies the validator to each row and aggregates results.
        :raises ValueError: If *name* is already registered.
        """
        if name in self._entries:
            raise ValueError(
                f"Validator {name!r} is already registered. "
                "Use a different name or unregister the existing one first."
            )

        def _decorator(
            factory: Callable[..., Callable[[Any], bool]],
        ) -> Callable[..., Callable[[Any], bool]]:
            # Wrap the factory so every closure it returns carries introspection
            # attributes used by the operator and planner.
            @functools.wraps(factory)
            def _wrapped_factory(*args: Any, **kwargs: Any) -> Callable[[Any], bool]:
                closure = factory(*args, **kwargs)
                arg_parts = [repr(a) for a in args]
                kwarg_parts = [f"{k}={v!r}" for k, v in sorted(kwargs.items())]
                call_str = f"{name}({', '.join(arg_parts + kwarg_parts)})"
                if not hasattr(closure, "_validator_name"):
                    closure._validator_name = name  # type: ignore[attr-defined]
                if not hasattr(closure, "_row_level"):
                    closure._row_level = row_level  # type: ignore[attr-defined]
                if not hasattr(closure, "_validator_display"):
                    closure._validator_display = call_str  # type: ignore[attr-defined]
                for k, v in sorted(kwargs.items()):
                    if not hasattr(closure, f"_{k}"):
                        setattr(closure, f"_{k}", v)  # e.g. _max_pct, _min_count
                return closure

            _wrapped_factory._validator_name = name  # type: ignore[attr-defined]
            _wrapped_factory._llm_context = llm_context  # type: ignore[attr-defined]
            _wrapped_factory._check_category = check_category  # type: ignore[attr-defined]
            _wrapped_factory._row_level = row_level  # type: ignore[attr-defined]
            _wrapped_factory._validator_display = name  # type: ignore[attr-defined]
            _wrapped_factory.__name__ = factory.__name__
            _wrapped_factory.__qualname__ = factory.__qualname__
            _wrapped_factory.__doc__ = factory.__doc__

            self._entries[name] = ValidatorEntry(
                factory=_wrapped_factory,
                llm_context=llm_context,
                check_category=check_category,
                row_level=row_level,
            )
            return _wrapped_factory

        return _decorator

    def get(self, name: str) -> ValidatorEntry:
        """
        Return the :class:`ValidatorEntry` for *name*.

        :raises KeyError: If *name* is not registered.
        """
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(
                f"Validator {name!r} is not registered. Available validators: {sorted(self._entries)}"
            ) from None

    def list_validators(self) -> list[str]:
        """Return sorted list of all registered validator names."""
        return sorted(self._entries)

    def is_row_level(self, validator: Callable[[Any], bool]) -> bool:
        """
        Return ``True`` when *validator* was produced by a row-level factory.

        Checks the ``_row_level`` attribute set by the factory closure and,
        as a fallback, the registry entry for the factory name.
        """
        if hasattr(validator, "_row_level"):
            return bool(validator._row_level)
        factory_name: str | None = getattr(validator, "_validator_name", None)
        if factory_name and factory_name in self._entries:
            return self._entries[factory_name].row_level
        return False

    def build_llm_context(self, validators: dict[str, Callable[[Any], bool]]) -> str:
        """
        Collect LLM context strings from all validators that carry one.

        Aggregate and row-level validators are emitted in separate sections so
        the LLM knows which checks require raw-row SELECTs vs. aggregate queries.

        Checks three sources in order for each validator callable:

        1. Registry entry (if the callable's factory was registered).
        2. ``_llm_context`` attribute on the callable itself.
        3. ``llm_context`` attribute on the callable itself.

        :param validators: The ``{check_name: callable}`` dict from the operator.
        :returns: Combined context string ready for injection into the system prompt,
            or empty string if no validator carries context.
        """
        aggregate_lines: list[str] = []
        row_level_lines: list[str] = []

        for check_name, validator in validators.items():
            context = self._resolve_llm_context(validator)
            if not context:
                continue
            if self.is_row_level(validator):
                row_level_lines.append(f"  - {check_name}: {context}")
            else:
                aggregate_lines.append(f"  - {check_name}: {context}")

        if not aggregate_lines and not row_level_lines:
            return ""

        parts: list[str] = []
        if aggregate_lines:
            parts.append(
                "\nCUSTOM VALIDATOR CONTEXT:\n"
                "  The following checks have specific metric requirements.\n"
                "  Generate SQL that returns values matching these descriptions:\n"
                + "\n".join(aggregate_lines)
            )
        if row_level_lines:
            parts.append(
                "\nROW-LEVEL CHECKS:\n"
                "  The following checks require ROW-LEVEL validation.\n"
                "  For each, generate a SELECT that returns the primary key column(s)\n"
                "  and the column(s) to validate — do NOT aggregate.\n"
                "  Set check.row_level = true on these DQCheck entries.\n"
                "  The Python-side validator will inspect each returned value:\n" + "\n".join(row_level_lines)
            )
        return "\n".join(parts) + "\n"

    def _resolve_llm_context(self, validator: Callable[[Any], bool]) -> str:
        """Resolve LLM context from registry entries or callable attributes."""
        # Check registry by factory name attribute.
        factory_name: str | None = getattr(validator, "_validator_name", None)
        if factory_name and factory_name in self._entries:
            entry = self._entries[factory_name]
            if entry.llm_context:
                return entry.llm_context

        # Fallback: attribute on the callable itself.
        for attr in ("_llm_context", "llm_context"):
            context = getattr(validator, attr, None)
            if context and isinstance(context, str):
                return context

        return ""

    def unregister(self, name: str) -> None:
        """
        Remove a validator from the registry.

        :raises KeyError: If *name* is not registered.
        """
        try:
            del self._entries[name]
        except KeyError:
            raise KeyError(f"Validator {name!r} is not registered.") from None


default_registry = ValidatorRegistry()


def register_validator(
    name: str,
    *,
    llm_context: str = "",
    check_category: str = "",
    row_level: bool = False,
) -> Callable[[Callable[..., Callable[[Any], bool]]], Callable[..., Callable[[Any], bool]]]:
    """
    Register a validator factory in the :data:`default_registry`.

    Use as a decorator on a factory function::

        @register_validator(
            "freshness_check",
            llm_context="Compute hours since MAX(updated_at). Returns DOUBLE.",
            check_category="freshness",
        )
        def freshness_check(*, max_hours: float):
            def _check(value):
                return float(value) <= max_hours

            return _check

    For row-level validation (e.g. TCKN formula)::

        @register_validator(
            "tckn_check",
            llm_context="ROW-LEVEL: SELECT pk, tckn_col FROM table. No aggregation.",
            check_category="row_level",
            row_level=True,
        )
        def tckn_check(*, max_invalid_pct: float = 0.0):
            def _check_row(value): ...

            return _check_row

    :param name: Unique name for this validator.
    :param llm_context: SQL generation hint injected into the LLM prompt.
    :param check_category: Custom check category for LLM grouping.
    :param row_level: When ``True``, the LLM generates a plain SELECT returning
        raw column values.  The planner validates each row with the callable.
    """
    return default_registry.register(
        name, llm_context=llm_context, check_category=check_category, row_level=row_level
    )


@register_validator(
    "null_pct_check",
    llm_context="Returns null percentage as a float between 0.0 and 1.0. SQL pattern: COUNT(CASE WHEN col IS NULL THEN 1 END) * 1.0 / COUNT(*).",
    check_category="null_check",
)
def null_pct_check(*, max_pct: float) -> Callable[[Any], bool]:
    """
    Return a validator that passes when ``value <= max_pct``.

    :param max_pct: Maximum allowed null percentage (0.0 – 1.0).
    :raises TypeError: When the metric value cannot be converted to ``float``.
    """

    def _check(value: Any) -> bool:
        try:
            return float(value) <= max_pct
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"null_pct_check(max_pct={max_pct!r}): expected a numeric value, got {value!r}"
            ) from exc

    return _check


@register_validator(
    "row_count_check",
    llm_context="Returns an integer row count. SQL pattern: COUNT(*).",
    check_category="row_count",
)
def row_count_check(*, min_count: int) -> Callable[[Any], bool]:
    """
    Return a validator that passes when ``value >= min_count``.

    :param min_count: Minimum required row count.
    :raises TypeError: When the metric value cannot be converted to ``int``.
    """

    def _check(value: Any) -> bool:
        try:
            return int(value) >= min_count
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"row_count_check(min_count={min_count!r}): expected an integer value, got {value!r}"
            ) from exc

    return _check


@register_validator(
    "duplicate_pct_check",
    llm_context="Returns duplicate percentage as a float between 0.0 and 1.0. SQL pattern: (COUNT(*) - COUNT(DISTINCT col)) * 1.0 / COUNT(*).",
    check_category="uniqueness",
)
def duplicate_pct_check(*, max_pct: float) -> Callable[[Any], bool]:
    """
    Return a validator that passes when ``value <= max_pct``.

    :param max_pct: Maximum allowed duplicate percentage (0.0 – 1.0).
    :raises TypeError: When the metric value cannot be converted to ``float``.
    """

    def _check(value: Any) -> bool:
        try:
            return float(value) <= max_pct
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"duplicate_pct_check(max_pct={max_pct!r}): expected a numeric value, got {value!r}"
            ) from exc

    return _check


@register_validator(
    "between_check",
    llm_context="Returns a numeric value that will be compared against inclusive bounds. SQL should return a single DOUBLE or INTEGER metric.",
    check_category="numeric_range",
)
def between_check(*, min_val: float, max_val: float) -> Callable[[Any], bool]:
    """
    Return a validator that passes when ``min_val <= value <= max_val``.

    :param min_val: Inclusive lower bound.
    :param max_val: Inclusive upper bound.
    :raises ValueError: When *min_val* > *max_val*.
    :raises TypeError: When the metric value cannot be converted to ``float``.
    """
    if min_val > max_val:
        raise ValueError(f"between_check: min_val ({min_val!r}) must be <= max_val ({max_val!r})")

    def _check(value: Any) -> bool:
        try:
            return min_val <= float(value) <= max_val
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"between_check(min_val={min_val!r}, max_val={max_val!r}): "
                f"expected a numeric value, got {value!r}"
            ) from exc

    return _check


@register_validator(
    "exact_check",
    llm_context="Returns a value that must exactly equal an expected constant. SQL should return a single scalar metric.",
    check_category="validity",
)
def exact_check(*, expected: Any) -> Callable[[Any], bool]:
    """
    Return a validator that passes when ``value == expected``.

    .. note::
        Comparison uses Python's ``==`` operator without type coercion.
        ``Decimal(0) == 0`` passes (Python numeric promotion), but
        ``"0" == 0`` does not.  The behaviour depends on the DB driver's
        Python type for the returned column.

    :param expected: The exact value the metric must equal.
    """

    def _check(value: Any) -> bool:
        return value == expected

    return _check


class DQValidationToolset:
    """
    Validator catalog manager that exposes registered validators to the LLM.

    Serves two purposes:

    1. **Prompt generation** — :meth:`build_system_prompt_section` produces a
       text block that is injected into the LLM system prompt so the model knows
       which validators are available, their parameters, and what SQL metric each
       one expects.

    2. **Suggestion validation** — :meth:`validate_suggestion` verifies that the
       name and arguments proposed by the LLM can actually instantiate a callable
       without error, before the plan is accepted.

    :param registry: Validator registry to expose.  Defaults to
        :data:`default_registry`.
    """

    def __init__(self, registry: ValidatorRegistry | None = None) -> None:
        self._registry: ValidatorRegistry = registry if registry is not None else default_registry

    def build_system_prompt_section(self) -> str:
        """
        Return a system-prompt block listing all registered validators.

        The block is appended to the SQL-planning prompt and tells the LLM
        which validators exist, their parameter signatures, and what SQL
        metric each one expects.  The LLM fills ``validator_name`` and
        ``validator_args`` on each :class:`~airflow.providers.common.ai.utils.dq_models.DQCheck`
        based on this catalogue.
        """
        lines: list[str] = [
            "",
            "AVAILABLE VALIDATORS:",
            "  For each DQCheck, select the most appropriate validator from the list below.",
            "  Fill the validator_name field with the exact name and validator_args with the",
            "  required keyword arguments as a JSON object.",
            '  If no validator is appropriate for a check, set validator_name to "none".',
            "  If the check has a [FIXED VALIDATOR] annotation, leave validator_name as null.",
            "",
        ]

        aggregate_entries: list[str] = []
        row_level_entries: list[str] = []

        for name in self._registry.list_validators():
            entry = self._registry.get(name)
            sig_str = self._format_signature(entry.factory)
            category = entry.check_category or "—"
            block = (
                f'  - name: "{name}"\n'
                f"    category: {category}\n"
                f"    row_level: {str(entry.row_level).lower()}\n"
                f"    parameters: {sig_str}\n"
                f"    description: {entry.llm_context or '(no description)'}"
            )
            if entry.row_level:
                row_level_entries.append(block)
            else:
                aggregate_entries.append(block)

        if aggregate_entries:
            lines.append("  Aggregate validators (SQL returns a single scalar metric):")
            lines.extend(aggregate_entries)
            lines.append("")

        if row_level_entries:
            lines.append("  Row-level validators (SQL returns raw column values per row, no aggregation):")
            lines.extend(row_level_entries)
            lines.append("")

        lines += [
            "  IMPORTANT: validator_args must contain ONLY the keyword argument names shown",
            "  above — no extra keys, no positional args.  The argument values must match the",
            "  expected Python types (float, int, etc.).",
            "",
        ]
        return "\n".join(lines)

    def validate_suggestion(
        self,
        check_name: str,
        validator_name: str,
        validator_args: dict[str, Any],
    ) -> tuple[bool, str]:
        """
        Verify that *validator_name* exists and can be instantiated with *validator_args*.

        :param check_name: The check this suggestion belongs to (used in error messages).
        :param validator_name: Name of the validator factory as registered.
        :param validator_args: Keyword arguments to pass to the factory.
        :returns: ``(True, "")`` on success; ``(False, error_message)`` on failure.
        """
        if not validator_name or validator_name.lower() == "none":
            return False, (
                f"Check {check_name!r}: LLM did not suggest a validator (validator_name is null or 'none')."
            )

        try:
            entry = self._registry.get(validator_name)
        except KeyError:
            available = self._registry.list_validators()
            return False, (
                f"Check {check_name!r}: validator {validator_name!r} is not registered. "
                f"Available validators: {available}"
            )

        # Validate argument names against the factory signature.
        sig = inspect.signature(entry.factory)
        invalid_args = [k for k in validator_args if k not in sig.parameters]
        if invalid_args:
            valid_params = [p for p in sig.parameters if p != "self"]
            return False, (
                f"Check {check_name!r}: validator {validator_name!r} received unknown "
                f"argument(s) {invalid_args}. Valid parameters: {valid_params}"
            )

        # Try to instantiate the validator to catch type errors early.
        try:
            entry.factory(**validator_args)
        except Exception as exc:
            return False, (
                f"Check {check_name!r}: failed to instantiate validator {validator_name!r} "
                f"with args {validator_args!r}: {exc}"
            )

        return True, ""

    def instantiate(self, validator_name: str, validator_args: dict[str, Any]) -> Callable[[Any], bool]:
        """
        Instantiate and return the validator callable for *validator_name*.

        :param validator_name: Registered validator factory name.
        :param validator_args: Keyword arguments for the factory.
        :raises KeyError: If *validator_name* is not registered.
        :raises Exception: If the factory raises during instantiation.
        """
        entry = self._registry.get(validator_name)
        return entry.factory(**validator_args)

    @staticmethod
    def _format_signature(factory: Callable[..., Any]) -> str:
        """Return a compact human-readable parameter list for a validator factory."""
        try:
            sig = inspect.signature(factory)
        except (ValueError, TypeError):
            return "(unknown)"
        parts: list[str] = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            annotation = (
                param.annotation.__name__
                if isinstance(param.annotation, type)
                else (str(param.annotation) if param.annotation is not inspect.Parameter.empty else "Any")
            )
            if param.default is inspect.Parameter.empty:
                parts.append(f"{name}: {annotation}")
            else:
                parts.append(f"{name}: {annotation} = {param.default!r}")
        return "(" + ", ".join(parts) + ")" if parts else "(no parameters)"
