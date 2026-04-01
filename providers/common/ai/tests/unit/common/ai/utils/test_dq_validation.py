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

import pytest

from airflow.providers.common.ai.utils.dq_validation import (
    ValidatorRegistry,
    between_check,
    default_registry,
    duplicate_pct_check,
    exact_check,
    null_pct_check,
    register_validator,
    row_count_check,
)


class TestNullPctCheck:
    def test_passes_when_value_at_threshold(self):
        assert null_pct_check(max_pct=0.05)(0.05) is True

    def test_passes_when_value_below_threshold(self):
        assert null_pct_check(max_pct=0.05)(0.01) is True

    def test_fails_when_value_above_threshold(self):
        assert null_pct_check(max_pct=0.05)(0.06) is False

    def test_accepts_integer_value(self):
        assert null_pct_check(max_pct=1)(0) is True

    def test_accepts_string_numeric_value(self):
        assert null_pct_check(max_pct=0.1)("0.05") is True

    def test_raises_type_error_for_non_numeric(self):
        with pytest.raises(TypeError, match="null_pct_check"):
            null_pct_check(max_pct=0.05)("not-a-number")

    def test_repr_contains_threshold(self):
        fn = null_pct_check(max_pct=0.05)
        assert "0.05" in repr(fn)
        assert "null_pct_check" in repr(fn)


class TestRowCountCheck:
    def test_passes_when_value_equals_min(self):
        assert row_count_check(min_count=1000)(1000) is True

    def test_passes_when_value_above_min(self):
        assert row_count_check(min_count=1000)(9999) is True

    def test_fails_when_value_below_min(self):
        assert row_count_check(min_count=1000)(999) is False

    def test_accepts_string_numeric(self):
        assert row_count_check(min_count=5)("10") is True

    def test_raises_type_error_for_non_numeric(self):
        with pytest.raises(TypeError, match="row_count_check"):
            row_count_check(min_count=100)(None)

    def test_repr_contains_threshold(self):
        fn = row_count_check(min_count=500)
        assert "500" in repr(fn)
        assert "row_count_check" in repr(fn)


class TestDuplicatePctCheck:
    def test_passes_when_value_at_threshold(self):
        assert duplicate_pct_check(max_pct=0.01)(0.01) is True

    def test_passes_when_value_below_threshold(self):
        assert duplicate_pct_check(max_pct=0.01)(0.005) is True

    def test_fails_when_value_above_threshold(self):
        assert duplicate_pct_check(max_pct=0.01)(0.02) is False

    def test_raises_type_error_for_non_numeric(self):
        with pytest.raises(TypeError, match="duplicate_pct_check"):
            duplicate_pct_check(max_pct=0.01)(object())

    def test_repr_contains_threshold(self):
        fn = duplicate_pct_check(max_pct=0.02)
        assert "0.02" in repr(fn)


class TestBetweenCheck:
    def test_passes_on_lower_boundary(self):
        assert between_check(min_val=0.0, max_val=100.0)(0.0) is True

    def test_passes_on_upper_boundary(self):
        assert between_check(min_val=0.0, max_val=100.0)(100.0) is True

    def test_passes_in_range(self):
        assert between_check(min_val=0.0, max_val=100.0)(50.0) is True

    def test_fails_below_range(self):
        assert between_check(min_val=10.0, max_val=100.0)(9.9) is False

    def test_fails_above_range(self):
        assert between_check(min_val=0.0, max_val=100.0)(100.1) is False

    def test_raises_value_error_when_min_greater_than_max(self):
        with pytest.raises(ValueError, match="min_val"):
            between_check(min_val=100.0, max_val=0.0)

    def test_raises_type_error_for_non_numeric(self):
        with pytest.raises(TypeError, match="between_check"):
            between_check(min_val=0.0, max_val=10.0)("abc")

    def test_repr_contains_bounds(self):
        fn = between_check(min_val=1.0, max_val=9.0)
        assert "1.0" in repr(fn)
        assert "9.0" in repr(fn)


class TestExactCheck:
    def test_passes_for_equal_value(self):
        assert exact_check(expected=0)(0) is True

    def test_fails_for_unequal_value(self):
        assert exact_check(expected=0)(1) is False

    def test_works_with_string(self):
        assert exact_check(expected="active")("active") is True

    def test_works_with_none(self):
        assert exact_check(expected=None)(None) is True

    def test_repr_contains_expected(self):
        fn = exact_check(expected=42)
        assert "42" in repr(fn)
        assert "exact_check" in repr(fn)


class TestValidatorRegistry:
    def test_register_and_get(self):
        registry = ValidatorRegistry()
        llm_ctx = "Compute hours since MAX(ts). Returns DOUBLE."

        @registry.register("my_check", llm_context=llm_ctx, check_category="freshness")
        def my_check(*, threshold: float):
            return lambda v: float(v) <= threshold

        entry = registry.get("my_check")
        assert entry.factory is my_check
        assert entry.llm_context == llm_ctx
        assert entry.check_category == "freshness"

    def test_duplicate_name_raises(self):
        registry = ValidatorRegistry()

        @registry.register("dup_check")
        def first(*, x: int):
            return lambda v: v == x

        with pytest.raises(ValueError, match="already registered"):

            @registry.register("dup_check")
            def second(*, x: int):
                return lambda v: v != x

    def test_get_unknown_raises(self):
        registry = ValidatorRegistry()
        with pytest.raises(KeyError, match="not_registered"):
            registry.get("not_registered")

    def test_list_validators(self):
        registry = ValidatorRegistry()

        @registry.register("b_check")
        def b_factory(*, x: int):
            return lambda v: v == x

        @registry.register("a_check")
        def a_factory(*, x: int):
            return lambda v: v == x

        assert registry.list_validators() == ["a_check", "b_check"]

    def test_unregister(self):
        registry = ValidatorRegistry()

        @registry.register("temp_check")
        def temp(*, x: int):
            return lambda v: v == x

        registry.unregister("temp_check")
        assert "temp_check" not in registry.list_validators()

    def test_unregister_unknown_raises(self):
        registry = ValidatorRegistry()
        with pytest.raises(KeyError, match="not_registered"):
            registry.unregister("not_registered")

    def test_factory_gets_metadata_attributes(self):
        registry = ValidatorRegistry()

        @registry.register("attr_check", llm_context="some context", check_category="custom")
        def attr_factory(*, x: int):
            return lambda v: v == x

        assert attr_factory._validator_name == "attr_check"
        assert attr_factory._llm_context == "some context"
        assert attr_factory._check_category == "custom"


class TestBuildLlmContext:
    def test_returns_empty_for_plain_callables(self):
        registry = ValidatorRegistry()
        validators = {"check1": lambda v: v > 0}
        assert registry.build_llm_context(validators) == ""

    def test_collects_context_from_registry(self):
        registry = ValidatorRegistry()

        @registry.register("ctx_check", llm_context="Count rows. Returns INTEGER.")
        def ctx_factory(*, min_count: int):
            def _check(value):
                return int(value) >= min_count

            return _check  # decorator auto-stamps _validator_name

        validators = {"my_metric": ctx_factory(min_count=100)}
        result = registry.build_llm_context(validators)
        assert "my_metric" in result
        assert "Count rows. Returns INTEGER." in result
        assert "CUSTOM VALIDATOR CONTEXT" in result

    def test_collects_context_from_attribute(self):
        registry = ValidatorRegistry()

        def custom(value):
            return float(value) >= 0.95

        custom.llm_context = "Completeness ratio. Returns float 0.0-1.0."

        validators = {"completeness": custom}
        result = registry.build_llm_context(validators)
        assert "completeness" in result
        assert "Completeness ratio" in result

    def test_mixed_validators(self):
        registry = ValidatorRegistry()

        @registry.register("reg_check", llm_context="Registered context.")
        def reg_factory(*, x: int):
            def _check(v):
                return int(v) >= x

            return _check  # decorator auto-stamps _validator_name

        plain = lambda v: v > 0

        validators = {
            "with_context": reg_factory(x=10),
            "without_context": plain,
        }
        result = registry.build_llm_context(validators)
        assert "with_context" in result
        assert "without_context" not in result


class TestBuiltinValidatorsRegistered:
    """Verify all built-in validators are in the default registry."""

    @pytest.mark.parametrize(
        "name",
        ["null_pct_check", "row_count_check", "duplicate_pct_check", "between_check", "exact_check"],
    )
    def test_builtin_in_default_registry(self, name):
        entry = default_registry.get(name)
        assert entry.llm_context
        assert entry.check_category

    def test_builtin_closures_carry_validator_name(self):
        fn = null_pct_check(max_pct=0.05)
        assert fn._validator_name == "null_pct_check"

        fn2 = row_count_check(min_count=100)
        assert fn2._validator_name == "row_count_check"

    def test_default_registry_build_llm_context_with_builtins(self):
        validators = {
            "nulls": null_pct_check(max_pct=0.05),
            "rows": row_count_check(min_count=100),
        }
        result = default_registry.build_llm_context(validators)
        assert "nulls" in result
        assert "rows" in result


class TestRegisterValidatorConvenience:
    def test_register_validator_uses_default_registry(self):
        name = "_test_convenience_check"
        try:

            @register_validator(name, llm_context="Test context.")
            def _test_convenience_check(*, x: int):
                return lambda v: v == x

            entry = default_registry.get(name)
            assert entry.llm_context == "Test context."
        finally:
            # Clean up to avoid polluting other tests.
            default_registry.unregister(name)


# ---------------------------------------------------------------------------
# Row-level registry support
# ---------------------------------------------------------------------------


class TestRowLevelRegistry:
    def test_register_with_row_level_true(self):
        name = "_test_row_level_check"
        try:

            @register_validator(name, llm_context="ROW-LEVEL check.", row_level=True)
            def _test_row_level_check():
                return lambda v: bool(v)

            entry = default_registry.get(name)
            assert entry.row_level is True
        finally:
            default_registry.unregister(name)

    def test_is_row_level_via_registry_entry(self):
        name = "_test_rl_is_row_level"
        try:

            @register_validator(name, llm_context="ROW-LEVEL.", row_level=True)
            def _rl():
                return lambda v: bool(v)  # decorator auto-stamps _row_level

            validator_fn = _rl()
            assert default_registry.is_row_level(validator_fn) is True
        finally:
            default_registry.unregister(name)

    def test_is_row_level_false_for_aggregate(self):
        fn = null_pct_check(max_pct=0.05)
        assert default_registry.is_row_level(fn) is False

    def test_build_llm_context_separates_row_level(self):
        name = "_test_rl_ctx"
        try:

            @register_validator(name, llm_context="ROW-LEVEL ctx.", row_level=True)
            def _rl_ctx():
                return lambda v: bool(v)  # decorator auto-stamps _row_level

            validators = {
                name: _rl_ctx(),
                "nulls": null_pct_check(max_pct=0.0),
            }
            ctx = default_registry.build_llm_context(validators)
            assert "ROW-LEVEL" in ctx
            assert "nulls" in ctx
        finally:
            default_registry.unregister(name)

    def test_kwargs_stamped_as_underscore_attrs(self):
        name = "_test_as_kwargs"
        try:

            @register_validator(name)
            def _factory(*, max_pct: float, min_count: int):
                return lambda v: True

            fn = _factory(max_pct=0.05, min_count=100)
            assert fn._max_pct == 0.05
            assert fn._min_count == 100
        finally:
            default_registry.unregister(name)

    def test_qualname_set_to_human_readable(self):
        name = "_test_as_qualname"
        try:

            @register_validator(name)
            def _factory(*, threshold: float):
                return lambda v: float(v) <= threshold

            fn = _factory(threshold=0.1)
            assert fn.__qualname__ == f"{name}(threshold=0.1)"
        finally:
            default_registry.unregister(name)

    def test_repr_returns_human_readable_string(self):
        name = "_test_as_repr"
        try:

            @register_validator(name)
            def _factory(*, max_pct: float):
                return lambda v: float(v) <= max_pct

            fn = _factory(max_pct=0.03)
            # repr() embeds __qualname__ → human-readable substring
            assert name in repr(fn)
            assert "max_pct=0.03" in repr(fn)
            # Direct __repr__() call returns the full call string
            assert fn.__repr__() == f"{name}(max_pct=0.03)"
        finally:
            default_registry.unregister(name)

    def test_explicit_override_is_preserved(self):
        """Factory author can still override auto-stamped attrs explicitly."""
        name = "_test_as_override"
        try:

            @register_validator(name, row_level=True)
            def _factory():
                fn = lambda v: bool(v)
                fn._row_level = False  # type: ignore[attr-defined]  # explicit override
                return fn

            fn = _factory()
            assert fn._row_level is False  # explicit wins over decorator default
        finally:
            default_registry.unregister(name)

    def test_no_stamps_needed_for_builtin_null_pct_check(self):
        fn = null_pct_check(max_pct=0.05)
        assert fn._validator_name == "null_pct_check"
        assert fn._row_level is False
        assert fn._max_pct == 0.05
        assert fn.__qualname__ == "null_pct_check(max_pct=0.05)"
        assert "null_pct_check" in repr(fn)
        assert "max_pct=0.05" in repr(fn)
        assert fn.__repr__() == "null_pct_check(max_pct=0.05)"
