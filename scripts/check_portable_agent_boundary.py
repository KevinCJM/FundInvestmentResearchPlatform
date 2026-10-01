"""Fail if a production platform candidate contains an embedded agent implementation."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ['backend/agent', 'frontend/src/components/agent', 'copilotkit', 'standalone-agent', 'backend/services/llm_settings_routes.py',
             'frontend/src/services/agent.ts', 'frontend/src/services/agentClient.ts', 'frontend/src/services/indicatorAgent.ts']


def check(root=ROOT):
    errors = [f'Retired implementation remains: {path}' for path in FORBIDDEN if (root/path).exists()]
    for path in (root/'backend').rglob('*.py'):
        if 'tests' in path.parts or '__pycache__' in path.parts:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = [node.module or ''] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names] if isinstance(node, ast.Import) else []
            for name in names:
                if name.split('.')[0] in {'agent','portable_agent','langchain','langchain_core','langgraph','openai'} or name.startswith('backend.agent'):
                    errors.append(f'{path.relative_to(root)}:{node.lineno}: forbidden runtime import {name}')
    for path in (root/'frontend/src').rglob('*'):
        if path.suffix not in {'.ts','.tsx','.js'} or '.test.' in path.name:
            continue
        text = path.read_text()
        for token in ['/api/agent/', '/api/settings/llm', 'useAgentConversation', '@copilotkit/', 'components/agent/']:
            if token in text:
                errors.append(f'{path.relative_to(root)} references retired agent contract {token}')
    for name in ['backend/requirements.txt','frontend/package.json','Dockerfile','docker-compose.yml','.gitmodules']:
        path = root/name
        if not path.exists():
            continue
        value = path.read_text().lower()
        for token in ['langgraph','langchain','copilotkit','standalone-agent','../portable-web-agent']:
            if token in value:
                errors.append(f'{name}: embeds framework source/dependency {token}')
    return errors


if __name__ == '__main__':
    errors = check()
    print(json.dumps({'status':'failed' if errors else 'passed','errors':errors}, ensure_ascii=False, indent=2))
    raise SystemExit(bool(errors))
