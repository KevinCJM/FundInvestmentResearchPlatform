"""Declared host navigation and business capabilities, never a natural-language classifier."""
import json
from pathlib import Path

PAGE_PATHS = {
    'indicator-studio': '/settings/indicators-models',
    'product-research': '/product-research/products',
    'product-compare': '/product-research/compare',
    'holding-diagnosis': '/post-investment/research-diagnosis',
    'platform-agent': '/settings/ai-agent',
    'scenario-algorithms': '/settings/scenario-algorithms',
    'historical-regimes': '/settings/scenario-algorithms/workbench',
    'published-scenarios': '/settings/scenario-algorithms',
    'global-events': '/settings/scenario-algorithms?center=events',
    'product-detail': '/product-research/products',
    'evaluation-plan': '/product-research/evaluation',
}
PARAMETERS = {'type': 'object', 'properties': {
    'action': {'type': 'string', 'maxLength': 100}, 'period': {'type': 'string', 'maxLength': 12},
    'as_of': {'type': ['string', 'null'], 'pattern': r'^\d{4}-\d{2}-\d{2}$'},
    'target': {'type': 'object', 'properties': {'kind': {'enum': ['etf', 'fund']}, 'product_id': {'type': 'string', 'maxLength': 100}},
               'required': ['kind', 'product_id'], 'additionalProperties': False},
    'targets': {'type': 'array', 'maxItems': 10, 'items': {'type': 'object', 'properties': {
        'kind': {'enum': ['etf', 'fund']}, 'product_id': {'type': 'string', 'maxLength': 100}},
        'required': ['kind', 'product_id'], 'additionalProperties': False}},
}, 'additionalProperties': False}


def capabilities():
    navigation = json.loads((Path(__file__).resolve().parents[3]/'shared/platform-navigation.json').read_text())
    entries = {}
    for stage in navigation:
        for item in [stage, *stage.get('nodes', [])]:
            entries[item['path']] = {'id': stage['id']+'.'+item['id'], 'title': item['label'], 'description': item['description'],
                'path': item['path'], 'status': item.get('status', 'available'), 'actions': ['navigate'], 'handoff': False,
                'parameters': {'type': 'object', 'additionalProperties': False}}
    for page in ('indicator-studio', 'product-research', 'product-compare', 'holding-diagnosis'):
        path = PAGE_PATHS[page]
        old = entries.get(path, {'title': {'product-compare': '产品比较', 'holding-diagnosis': '持仓诊断'}[page] if page not in {'indicator-studio', 'product-research'} else page,
                                 'description': '读取当前工作区的真实研究条件与结果。', 'path': path, 'status': 'available'})
        entries[path] = {**old, 'id': page, 'actions': ['navigate', 'read', 'execute'], 'parameters': PARAMETERS, 'handoff': page == 'indicator-studio'}
    for item in entries.values():
        if item['status'] == 'prototype':
            item['description'] += ' 当前为未实现原型，仅支持导航，不提供业务执行。'
    for page, title in [('historical-regimes','历史情景工作台'), ('published-scenarios','已发布模型情景'), ('scenario-algorithms','高级情景实验'), ('global-events','全球历史事件')]:
        entries['capability:'+page] = {'id': page, 'title': title, 'description': '仅在此工作区查询目录、校验或试算；保存和发布由原业务页面人工操作。',
            'path': PAGE_PATHS[page], 'status': 'available', 'actions': ['navigate','read','execute'], 'handoff': False,
            'parameters': {'type': 'object', 'additionalProperties': False}}
    return list(entries.values())


def framework_capabilities():
    return [{k: value for k, value in item.items() if k not in {'path', 'status'}} for item in capabilities()]
