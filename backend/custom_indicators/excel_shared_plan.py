"""Group native multi-output evidence without duplicating its calculation sheet."""
from __future__ import annotations

import copy
from dataclasses import replace
import json

from cal_indicators.typed_dsl import compose_typed_scalar_bundle
from .excel_formula import SingleProductExcelFormulaCompiler


def shared_excel_plans(entries, sheet_name_for):
    """Return per-result views and unique worksheet jobs using identical windows.

    Ordinary scalar/legacy bundles retain their original layout. Native
    multi-output graphs share one compiler and worksheet per target/window.
    The views change only the selected root, never generate another scan.
    """
    groups = {}
    for index, (definition, plan, evidence) in enumerate(entries):
        native = any(node.inferred_type.kind == "record" for node in plan.nodes)
        key = (evidence.target["kind"], evidence.target["product_id"],
               tuple(sorted(plan.context_requirements)), json.dumps(evidence.result.get("window"), sort_keys=True),
               plan.dsl_version, plan.operator_registry_version) if native else ("single", index)
        groups.setdefault(key, []).append(index)
    views = [None] * len(entries)
    jobs = []
    for indexes in groups.values():
        representative = next((index for index in indexes if entries[index][2].context), indexes[0])
        definition, plan, evidence = entries[representative]
        sheet_name = sheet_name_for(representative + 1, evidence.target["product_id"])
        if not evidence.context:
            for index in indexes:
                output, output_plan, item = entries[index]
                jobs.append((output, output_plan, item, sheet_name_for(index+1, item.target["product_id"]), None))
            continue
        expressions = {str(index): entries[index][1].python_expression for index in indexes}
        roots = {str(representative): plan.root_id}
        if len(indexes) > 1:
            bundle = compose_typed_scalar_bundle(
                {"result_" + key: value for key, value in expressions.items()},
                variable_types=plan.context_requirements, dsl_version=plan.dsl_version,
                operator_registry_version=plan.operator_registry_version,
            )
            roots = {str(index): bundle.roots["result_" + str(index)] for index in indexes}
            plan = replace(plan, nodes=bundle.nodes, root_id=roots[str(representative)],
                           context_requirements=bundle.context_requirements)
        else:
            roots = {str(indexes[0]): plan.root_id}
        compiler = SingleProductExcelFormulaCompiler(
            plan=plan, context=evidence.context, dates_by_variable=evidence.dates_by_variable,
            sheet_name=sheet_name, prefix=f"P{representative+1:02d}",
        )
        for index in indexes:
            view = copy.copy(compiler)
            view.root_id = roots[str(index)]
            views[index] = view
        jobs.append((definition, plan, evidence, sheet_name, compiler))
    return views, jobs
