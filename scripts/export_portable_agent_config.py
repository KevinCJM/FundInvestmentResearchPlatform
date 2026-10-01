"""Generate host-owned declarative configuration; no agent package is imported."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'backend'))
sys.path.insert(0, str(ROOT))

from integrations.portable_agent.catalog import framework_capabilities
from research_access.tools import TOOL_REGISTRY


def configuration(platform_url, origin, model_endpoints, app_id='fund-research'):
    tools = []
    for name, item in TOOL_REGISTRY.items():
        wire = name.replace('.', '_')
        tools.append({'name': wire, 'description': item.description, 'parameters': item.arguments.model_json_schema(),
            'result_schema': {'type': 'object'}, 'kind': 'http', 'effect': 'draft' if item.progress == 'draft' else 'read',
            'requires_execution_intent': item.progress != 'read', 'scope': 'tool:'+wire,
            'endpoint': platform_url.rstrip('/')+'/internal/portable-agent/tools/'+wire,
            'operations_endpoint': platform_url.rstrip('/')+'/internal/portable-agent/operations',
            'headers_env': {'Authorization': 'PORTABLE_AGENT_HOST_AUTHORIZATION'}, 'idempotent': True, 'timeout': 30})
    properties = {name: {'type': 'string'} for name in ['ref','hash','grant_id','workspace','scope_key','memory_scope','page']}
    properties.update(grant_revision={'type': 'integer'}, capability_id={'type': ['string','null']},
                      authoring_id={'type': ['string','null']}, derived_from={'type': ['string','null']},
                      intent_parameters={'type': 'object'}, summary={'type': 'object'})
    properties['memory_objects'] = {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 10}
    return {'applications': [{'id': app_id, 'title': 'AI 助手', 'instructions': (ROOT/'config/portable-agent/instructions.txt').read_text(),
        'origins': [origin], 'model_endpoints': model_endpoints, 'context_schema': {
            'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False},
        'tools': tools, 'features': ['tasks','memory','intent'], 'capabilities': framework_capabilities(), 'auto_compact': True,
        'require_context_grant': True, 'authorization_url': platform_url.rstrip('/')+'/internal/portable-agent/authorize',
        'input_policy_url': platform_url.rstrip('/')+'/internal/portable-agent/admission',
        'policy_headers_env': {'Authorization': 'PORTABLE_AGENT_HOST_AUTHORIZATION'}, 'max_model_calls': None,
        'max_context_chars': 250000}]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform-url', required=True)
    parser.add_argument('--origin', required=True)
    parser.add_argument('--model-endpoint', action='append', required=True)
    parser.add_argument('--app', default='fund-research')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(configuration(args.platform_url, args.origin, args.model_endpoint, args.app), ensure_ascii=False, indent=2)+'\n')
