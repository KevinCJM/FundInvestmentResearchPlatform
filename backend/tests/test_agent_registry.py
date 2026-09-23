"""Tool declarations are one authority for dispatch, permissions and progress."""
from dataclasses import replace

import pytest

from agent.contracts import AgentError
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
                replace(tool,arguments=object), replace(tool,equivalent_to='missing'), replace(tool,view=None)]:
        with pytest.raises(ValueError): build_tool_registry(bad)
    with pytest.raises(TypeError): TOOL_REGISTRY['injected'] = tool
