"""Explicit platform contract adapter for native single-product scalar batches.

The platform validates its versioned financial DSL. All numerical lowering,
CSE, interval computation and scheduling remain in CalMetricsEngine C++.
This opt-in API does not select a backend for existing service requests.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping, Sequence

from compute_policy import validate_execution_audit
from .typed_dsl import compose_typed_expression
from .typed_operators import TYPED_DSL_VERSION
from custom_indicators.series_definitions import (
    normalize_parameter_schema,
    normalize_series_parameters,
    parameter_variable_types,
)
from custom_indicators.series_parameters import validate_parameter_definition
from custom_indicators.variable_registry import (
    normalize_variable_latex, resolve_source_contract_versions, variable_types,
)


class _ValidatedPreparedBatch:
    """Validate each native result without copying or changing its lifetime."""

    def __init__(self, prepared):
        self._prepared = prepared

    def run(self):
        # The audited call executes once and returns the same borrowed buffer.
        return self.run_audit().values

    def run_audit(self):
        result = self._prepared.run_audit()
        validate_execution_audit(result.audit)
        return result

    def run_snapshot(self):
        result = self._prepared.run_snapshot()
        validate_execution_audit(result.audit)
        return result


class CppIndicatorBatchPlan:
    """Prepared immutable definitions; parameter values never change the graph id."""

    def __init__(self, definitions: Sequence[Mapping[str, Any]]):
        from calmetrics_engine import GraphCompiler

        if not definitions:
            raise ValueError("Expected at least one independent indicator")
        self._definitions = deepcopy(list(definitions))
        expressions, bindings, contracts = [], [], []
        physical: set[str] = {"adjusted_nav"}
        native_types: dict[str, str] = {}
        self._parameter_keys: list[dict[str, str]] = []
        for index, definition in enumerate(self._definitions):
            if definition.get("context_kind", "single_product") != "single_product":
                raise ValueError("C++ batch adapter requires single_product context")
            if (
                definition.get("result_kind", "scalar") != "scalar"
                or definition.get("output_contract", "scalar") != "scalar"
            ):
                raise ValueError(
                    "C++ batch adapter requires independent scalar outputs"
                )
            schema = normalize_parameter_schema(
                definition.get("parameter_schema") or []
            )
            definition["parameter_schema"] = schema
            if schema:
                validate_parameter_definition(definition)
            dsl_version = str(definition.get("dsl_version") or TYPED_DSL_VERSION)
            types = {
                **variable_types("single_product", dsl_version),
                **parameter_variable_types(definition),
            }
            plan = compose_typed_expression(
                normalize_variable_latex(str(definition.get("expression") or "")),
                variable_types=types,
                dsl_version=dsl_version,
                operator_registry_version=definition.get("operator_registry_version"),
                parameter_names=frozenset(item["id"] for item in schema),
            )
            source_versions = resolve_source_contract_versions(plan.dsl_version, definition)
            # Financial/source versions are an explicit CSE namespace, not inferred from names.
            contracts.append(
                json.dumps(
                    {
                        "adapter": "firp-scalar-cpp-1",
                        "dsl": plan.dsl_version,
                        "registry": plan.operator_registry_version,
                        "data": source_versions["data_contract_version"],
                        "variables": source_versions["variable_registry_version"],
                        "context": source_versions["context_schema_version"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            expressions.append(plan.python_expression)
            prefix = f"metric_{index}__"
            rate = prefix + "system_annual_percent"
            native_types[rate] = "scalar"
            annual = f"divide({rate},100)"
            per_observation = (
                f"subtract(power(maximum(0,add(1,{annual})),divide(1,252)),1)"
            )
            elapsed = "subtract(last(observation_dates),first(observation_dates))"
            aliases = {
                "returns": "subtract(divide(interval_tail(adjusted_nav),lag(adjusted_nav)),1)",
                "log_returns": "log(divide(interval_tail(adjusted_nav),lag(adjusted_nav)))",
                "observation_count": "length(lag(adjusted_nav))",
                "window_elapsed_days": elapsed,
                "periods_per_year": "252",
                "annual_risk_free_rate_decimal": annual,
                "risk_free_rate_per_observation": per_observation,
                "risk_free_rate_per_period": per_observation,
                "risk_free_return_window": f"subtract(power(maximum(0,add(1,{annual})),divide({elapsed},365)),1)",
            }
            keys = {}
            for item in schema:
                name = item["id"]
                if definition.get("parameter_contract_version") == "1.0":
                    keys[name] = prefix + "parameter_" + name
                    aliases[name] = keys[name]
                    native_types[keys[name]] = "scalar"
                else:
                    aliases[name] = repr(float(item["default"]))
            self._parameter_keys.append(keys)
            used = {node.label for node in plan.nodes if node.kind == "variable"}
            physical.update(used - aliases.keys())
            if used & {"window_elapsed_days", "risk_free_return_window"}:
                physical.add("observation_dates")
            bindings.append(aliases)
        # Nominal/financial types were checked above. Native input geometry is an
        # aligned float64 time axis; returns are derived inside each interval.
        native_types.update({name: "series" for name in sorted(physical)})
        self.graph = GraphCompiler(native_types).compile(
            expressions,
            error_policy="isolate",
            root_bindings=bindings,
            source_contracts=contracts,
            minimum_observations=2,
        )

    @property
    def fingerprint(self) -> str:
        return self.graph.fingerprint

    def parameters(
        self, supplied: Sequence[Mapping[str, Any] | None] | None = None
    ) -> dict[str, float]:
        if supplied is None:
            supplied = [None] * len(self._definitions)
        if len(supplied) != len(self._definitions):
            raise ValueError("Expected one parameter mapping per indicator")
        result: dict[str, float] = {}
        for index, (definition, overrides, keys) in enumerate(
            zip(self._definitions, supplied, self._parameter_keys)
        ):
            resolved = normalize_series_parameters(definition, overrides)
            result[f"metric_{index}__system_annual_percent"] = float(
                definition.get("annual_risk_free_rate_percent", 0.0)
            )
            result.update({keys[name]: float(resolved[name]) for name in keys})
        return result

    def execute(self, scheduler, inputs, starts, ends, *, parameters=None, **options):
        """Use the caller-owned native scheduler; never create a pool per request."""
        result = scheduler.execute(
            self.graph,
            inputs,
            starts,
            ends,
            parameters=self.parameters(parameters),
            **options,
        )
        validate_execution_audit(result.audit)
        return result

    def prepare(self, scheduler, inputs, starts, ends, *, parameters=None):
        """Use run_snapshot() for retained results; run() explicitly borrows output."""
        prepared = scheduler.prepare_execution(
            self.graph, inputs, starts, ends, parameters=self.parameters(parameters)
        )
        return _ValidatedPreparedBatch(prepared)
