"""Tool declarations are one authority for dispatch, permissions and progress."""
from dataclasses import replace

import pytest

from agent.contracts import AgentError, PageContext
from agent.scopes import allowed_tools, require_tool
from agent.tools import TOOL_REGISTRY, build_tool_registry, parse_arguments, tool_specs


def test_registered_tools_preserve_scope_and_domain_boundaries():
    shared = {'task.read', 'task.plan', 'memory.propose'}
    indicator = shared | {'context.read','page.read','page.recompute','metrics.lookup','metrics.infer','metrics.validate','metrics.draft_save',
                 'metrics.rolling_draft','metrics.availability','metrics.preview','products.search'}
    product = (indicator - {'page.recompute'}) | {'page.analyze'} | {'products.eval','products.series','products.plans'}
    portfolio = shared | {'page.read','page.analyze','context.read','metrics.lookup','products.plans','portfolios.context','portfolios.eval'}
    assert set(allowed_tools('indicator_center','single_product')) == indicator
    assert set(allowed_tools('product_research','single_product')) == product
    assert set(allowed_tools('product_research','portfolio')) == portfolio
    for scope,kind,expected in [('indicator_center','single_product',indicator), ('product_research','single_product',product), ('product_research','portfolio',portfolio)]:
        assert {spec['name'] for spec in tool_specs(allowed_tools(scope,kind))} == expected
        for name in TOOL_REGISTRY:
            if name in expected:
                require_tool(scope,kind,name)
            else:
                with pytest.raises(AgentError): require_tool(scope,kind,name)
    with pytest.raises(AgentError): tool_specs(('not.registered',))
    with pytest.raises(AgentError): parse_arguments('metrics.lookup',{'injected':'value'})


def test_registry_fails_closed_for_incomplete_or_duplicate_tools():
    tool = TOOL_REGISTRY['metrics.lookup']
    with pytest.raises(ValueError,match='duplicate'): build_tool_registry(tool,tool)
    for bad in [replace(tool,handler=None), replace(tool,scopes=()), replace(tool,domains=('unknown',)),
                replace(tool,arguments=object), replace(tool,equivalent_to='missing'), replace(tool,view=None),
                replace(tool,current_data=True), replace(tool,current_data=lambda page, args: True)]:
        with pytest.raises(ValueError): build_tool_registry(bad)
    with pytest.raises(TypeError): TOOL_REGISTRY['injected'] = tool


def test_tool_data_policy_distinguishes_immutable_runs_from_current_scenarios():
    portfolio = PageContext.model_validate({'page': 'holding-diagnosis', 'page_instance_id': 'run',
        'calculation': {'context_kind': 'portfolio', 'run_id': 'run-1'}})
    product = PageContext.model_validate({'page': 'product-compare', 'page_instance_id': 'compare',
        'calculation': {'context_kind': 'single_product'}})
    assert not TOOL_REGISTRY['portfolios.eval'].uses_current_data(portfolio, {})
    for operation in ('diagnosis', 'metrics'):
        assert not TOOL_REGISTRY['page.analyze'].uses_current_data(portfolio, {'operation': operation})
    assert TOOL_REGISTRY['page.analyze'].uses_current_data(portfolio, {'operation': 'scenario'})
    assert TOOL_REGISTRY['page.analyze'].uses_current_data(product, {'operation': 'comparison'})
    assert TOOL_REGISTRY['metrics.preview'].uses_current_data(product, {})
    assert not TOOL_REGISTRY['task.read'].uses_current_data(product, {})
