"""Read-only mathematical presentation of a regime graph, never its execution."""
from __future__ import annotations

from cal_indicators.typed_dsl import ValueType, compose_typed_expression
from cal_indicators.typed_latex import escape_latex_text, render_python_expression_latex
from .indicator_nodes import is_typed_formula_node, typed_node_expression
from .v2_registry import NODE_REGISTRY, PARAMETER_LABELS


def _text(value):
    # Escape every TeX text special, including the two not used in DSL identifiers.
    escaped = escape_latex_text(str(value)).replace('^', r'\textasciicircum{}').replace('~', r'\textasciitilde{}')
    return r'\text{' + escaped + '}'


def _number(value):
    return render_python_expression_latex(repr(value))


def graph_math_presentation(definition):
    """Render each named root with only its dependencies and actual parameters.

    Typed numeric expressions reuse Indicator Center's canonical renderer.
    Domain nodes without a closed-form expansion retain a named operator and
    their parameter list; they are never replaced by an approximate formula.
    """
    nodes = {node.id: node for node in definition.graph.nodes}
    indices = {node.id: index + 1 for index, node in enumerate(definition.graph.nodes)}

    def symbol(node_id, port):
        ports = NODE_REGISTRY[nodes[node_id].type]['outputs']
        index = next(i + 1 for i, item in enumerate(ports) if item['name'] == port)
        return rf'\mathbf{{z}}^{{({indices[node_id]},{index})}}'

    def parameters(node):
        properties = NODE_REGISTRY[node.type]['parameter_schema'].get('properties', {})
        defaults = {name: spec['default'] for name, spec in properties.items() if 'default' in spec}
        return {**defaults, **node.parameters}

    def rhs(node, port):
        metadata = NODE_REGISTRY[node.type]
        params = parameters(node)
        inputs = {name: symbol(ref.node_id, ref.port) for name, ref in node.inputs.items()}
        value = inputs.get('value', '')
        at = lambda expression, index='t': rf'\left({expression}\right)_{{{index}}}'
        if node.type == 'source.constant':
            return _number(params['value']) + r'\,\mathbf{1}'
        if node.type.startswith('source.'):
            label = node.label or metadata['label']
            return _text(label)
        if is_typed_formula_node(node, NODE_REGISTRY):
            aliases = params.get('variables', {})
            if not isinstance(aliases, dict):
                raise ValueError('Formula variable aliases must be a mapping')
            variables = {**inputs, **{name: inputs[target] for name, target in aliases.items()}}
            plan = compose_typed_expression(typed_node_expression(node, NODE_REGISTRY, port),
                variable_types={name: ValueType.series('T') for name in variables}, output_contract='series')
            return render_python_expression_latex(plan.python_expression, variables)
        if node.type == 'transform.identity':
            return value
        if node.type == 'transform.log':
            return rf'\ln\left({value}\right)'
        if node.type in {'transform.return', 'transform.yoy'}:
            window = _number(params.get('periods', params.get('window', 1)))
            return rf'\left(\frac{{{at(value)}}}{{{at(value, "t-" + window)}}}-1\right)_t'
        if node.type == 'segment.change':
            start, end = at(inputs['start']), at(inputs['end'])
            return rf'\left(\frac{{{at(value, end)}}}{{{at(value, start)}}}-1\right)_{{t:\,{start}\le t<{end}}}'
        if node.type == 'segment.duration':
            return rf'{inputs["end"]}-{inputs["start"]}'
        if node.type == 'model.range_threshold' and port == 'state':
            def bound(name):
                ref = node.inputs.get(name + '_bound')
                if ref and nodes[ref.node_id].type == 'source.constant':
                    return _number(parameters(nodes[ref.node_id])['value'])
                return at(inputs[name + '_bound']) if ref else _number(params[name])

            upper, lower = bound('upper'), bound('lower')
            labels = [_text(state.label) for state in definition.states[:3]]
            if len(labels) == 3:
                x = at(value)
                # Missing values and invalid boundaries remain unclassified.
                return (r'\begin{gathered}\left(\begin{cases}' + labels[0] + rf',&\mathcal{{V}}_t\land {x}>{upper}\\'
                    + labels[2] + rf',&\mathcal{{V}}_t\land {x}<{lower}\\'
                    + labels[1] + rf',&\mathcal{{V}}_t\land {lower}\le {x}\le {upper}\\'
                    + _text('未识别') + r',&\neg\mathcal{V}_t\end{cases}\right)_t'
                    + rf'\\ \mathcal{{V}}_t=\left[\operatorname{{finite}}({x},{lower},{upper})\land {lower}<{upper}\right]\end{{gathered}}')
        # A named domain operator is exact at the graph level. The accompanying
        # description and parameters explain its semantics without inventing math.
        arguments = ','.join(inputs.values())
        return rf'\operatorname{{{_text(metadata["label"])}}}\!\left({arguments};\,\theta_{{{indices[node.id]}}}\right)_{{{_text(port)}}}'

    display, all_steps = {}, {}
    for output_id, root in definition.graph.outputs.items():
        steps, seen = [], set()

        def visit(node_id, port):
            key = (node_id, port)
            if key in seen:
                return
            seen.add(key)
            node = nodes[node_id]
            for ref in node.inputs.values():
                visit(ref.node_id, ref.port)
            metadata = NODE_REGISTRY[node.type]
            properties = metadata['parameter_schema'].get('properties', {})
            visible_params = []
            for name, value in parameters(node).items():
                if name in {'rows', 'inline_rows', 'expression', 'variables'} or isinstance(value, (list, dict)):
                    continue
                # Connected bounds take precedence over unused fixed defaults.
                if node.type == 'model.range_threshold' and name + '_bound' in node.inputs:
                    continue
                title = properties.get(name, {}).get('title') or PARAMETER_LABELS.get(name, name)
                visible_params.append({'label': title, 'value': str(value)})
            steps.append({'node_id': node_id, 'port': port, 'label': node.label or metadata['label'],
                'latex': symbol(node_id, port) + '=' + rhs(node, port),
                'description': metadata.get('description', ''), 'parameters': visible_params})

        visit(root.node_id, root.port)
        label = definition.graph.channel_metadata.get(output_id)
        title = label.label if label else ('市场状态' if output_id == 'state' else output_id)
        display[output_id] = _text(title) + '=' + rhs(nodes[root.node_id], root.port)
        all_steps[output_id] = steps
    return {'display_latex': display, 'formula_steps': all_steps, 'math_notation_version': 'regime-math/1.0'}
