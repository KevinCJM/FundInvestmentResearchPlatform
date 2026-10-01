import importlib.util
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]/'check_portable_agent_boundary.py'


def test_boundary_rejects_renamed_runtime_import_and_retired_frontend_api(tmp_path):
    spec = importlib.util.spec_from_file_location('portable_boundary', MODULE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    (tmp_path/'backend/integrations').mkdir(parents=True)
    (tmp_path/'frontend/src').mkdir(parents=True)
    (tmp_path/'backend/integrations/renamed.py').write_text('from langgraph.graph import StateGraph\n')
    (tmp_path/'frontend/src/renamed.ts').write_text("fetch('/api/agent/sessions')\n")
    errors = module.check(tmp_path)
    assert any('forbidden runtime import' in e for e in errors)
    assert any('retired agent contract' in e for e in errors)
