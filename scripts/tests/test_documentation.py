"""Offline regressions for document integrity and exact Git candidate boundaries."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'check_documentation.py'
HERMES_PATH = 'skills/ai-hermes-self-evolve/scripts/evolve_ai_routing.py'
spec = importlib.util.spec_from_file_location('documentation_check_tested', SOURCE)
check = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = check
spec.loader.exec_module(check)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.STDOUT).decode().strip()


def write(root, path, text):
    p = root / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'config', 'user.email', 'docs@example.invalid')
    git(tmp_path, 'config', 'user.name', 'Docs Test')
    docs = []
    for path in ['README.md', 'AGENTS.md', 'docs/README.md', 'docs/topic.md']:
        docs.append({'path': path, 'title': path, 'topic': '测试', 'role': 'topic',
                     'status': 'active', 'modules': ['test'], 'read_when': 'verify'})
        write(tmp_path, path, '# Title\n\n## 中文标题\n\n## 中文标题\n')
    mapping = {'modules': [{'id': 'test', 'path': 'src', 'entry_files': ['src/service.py']}],
               'documentation': {'documents': docs}}
    write(tmp_path, 'docs/repo_map.json', json.dumps(mapping))
    write(tmp_path, 'docs/README.md', '# Index\n\n' + check.document_index(docs) + '\n')
    write(tmp_path, 'src/service.py', 'VALUE = 1\n')
    write(tmp_path, 'scripts/check_documentation.py', SOURCE.read_text())
    write(tmp_path, HERMES_PATH, (SOURCE.parents[1] / HERMES_PATH).read_text())
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-qm', 'baseline')
    return tmp_path


def inspect(root, changed=None, **kwargs):
    return check.inspect(check.Candidate(root), changed or [], **kwargs)


def test_reference_links_code_blocks_chinese_and_duplicate_anchors(repo):
    write(repo, 'README.md', '# Test\n[ref][target]\n[target]: docs/topic.md#中文标题-1\n\n````md\n[example](missing.md)\n````\n')
    assert inspect(repo)['errors'] == []


@pytest.mark.parametrize('target', ['missing.md', 'topic.md#absent', '/tmp/external.md', '../../escape.md'])
def test_broken_or_nonportable_links_fail(repo, target):
    write(repo, 'docs/topic.md', f'# Topic\n[link]({target})\n')
    assert inspect(repo)['errors']


def test_new_uncatalogued_and_extra_root_documents_fail(repo):
    write(repo, 'extra.md', '# extra\n')
    errors = inspect(repo)['errors']
    assert any('uncatalogued document: extra.md' in e for e in errors)
    assert any('root Markdown' in e for e in errors)


def test_catalog_role_and_module_are_validated(repo):
    p = repo / 'docs/repo_map.json'
    data = json.loads(p.read_text())
    data['documentation']['documents'][0].update(role='mystery', modules=['missing'])
    p.write_text(json.dumps(data))
    errors = inspect(repo)['errors']
    assert any('role/status' in e for e in errors)
    assert any('module ownership' in e for e in errors)


def test_generated_index_drift_is_reported(repo):
    p = repo / 'docs/README.md'
    p.write_text(p.read_text().replace('verify', 'edited', 1))
    assert any('index differs' in e for e in inspect(repo)['errors'])


def test_deleted_target_and_old_inbound_reference_fail(repo):
    write(repo, 'README.md', '[Old](docs/topic.md)\n')
    (repo / 'docs/topic.md').rename(repo / 'docs/renamed.md')
    errors = inspect(repo)['errors']
    assert any('missing catalog document' in e for e in errors)
    assert any('broken link' in e for e in errors)


def test_symlink_outside_repository_is_rejected(repo, tmp_path):
    outside = tmp_path.parent / (tmp_path.name + '-outside.md')
    outside.write_text('outside')
    p = repo / 'docs/topic.md'
    p.unlink()
    p.symlink_to(outside)
    with pytest.raises(ValueError, match='symlink escapes'):
        inspect(repo)


@pytest.mark.parametrize('mode', ['staged', 'commit'])
def test_git_candidate_does_not_treat_symlink_blob_as_document(repo, mode):
    p = repo / 'docs/topic.md'
    p.unlink()
    p.symlink_to('missing.md')
    git(repo, 'add', 'docs/topic.md')
    git(repo, 'commit', '-qm', 'symlink')
    with pytest.raises(ValueError, match='candidate symlink'):
        check.inspect(check.Candidate(repo, mode), [])


HEADER = '| ID | 状态 | 工作项 | 完成判据 | 证据/剩余事项 |\n| --- | --- | --- | --- | --- |\n'


@pytest.mark.parametrize('row', [
    '| TASK-1 | done | Build | Tested | no proof |',
    '| TASK-1 | verified | Build | Tested | no link |',
    '| TASK-1 | planned | Build | | Still needed |',
    '| bad-id | planned | Build | Tested | Still needed |',
])
def test_plan_status_completion_evidence_and_acceptance(row):
    assert check.check_plans('plan.md', HEADER + row + '\n')


def test_scoped_verified_plan_and_free_prose_are_distinct():
    assert not check.check_plans('plan.md', HEADER + '| TASK-1 | verified | Build | Test passes | [Evidence](evidence.md) |\n')
    assert not check.check_plans('plan.md', '历史计划曾经写过 done；这不是活动计划表。')


def test_duplicate_plan_ids_fail():
    row = '| TASK-1 | planned | Build | Tested | Still needed |\n'
    assert any('duplicate' in e for e in check.check_plans('plan.md', HEADER + row + row))


@pytest.mark.parametrize('work', [r'A \| B', r'`A \| B`', r'路径 \\| 选项'])
def test_plan_cells_allow_escaped_pipes(work):
    row = f'| TASK-1 | planned | {work} | Tested | Still needed |\n'
    assert not check.check_plans('plan.md', HEADER + row)


def test_plan_example_in_code_fence_is_not_a_live_plan():
    assert not check.check_plans('plan.md', '```markdown\n' + HEADER + '| EXAMPLE | done | | | |\n```\n')


def test_staged_candidate_ignores_unstaged_document_repairs(repo):
    write(repo, 'README.md', '[bad](docs/missing.md)\n')
    git(repo, 'add', 'README.md')
    write(repo, 'README.md', '# Fixed only in worktree\n')
    assert not inspect(repo)['errors']
    result = check.inspect(check.Candidate(repo, 'staged'), ['README.md'])
    assert any('broken link' in e for e in result['errors'])


def test_untracked_document_is_not_in_staged_candidate(repo):
    write(repo, 'README.md', '[new](docs/untracked.md)\n')
    write(repo, 'docs/untracked.md', '# new\n')
    git(repo, 'add', 'README.md')
    result = check.inspect(check.Candidate(repo, 'staged'), ['README.md'])
    assert any('broken link' in e for e in result['errors'])


def test_commit_candidate_ignores_worktree_changes(repo):
    write(repo, 'README.md', '[bad](missing.md)\n')
    assert inspect(repo)['errors']
    assert not check.inspect(check.Candidate(repo, 'commit'), [])['errors']


@pytest.mark.parametrize('mode', ['staged', 'commit'])
def test_candidate_helper_ignores_code_outside_candidate(repo, mode):
    base = git(repo, 'rev-parse', 'HEAD')
    write(repo, 'src/service.py', 'VALUE = 2\n')
    git(repo, 'add', 'src/service.py')
    if mode == 'commit':
        git(repo, 'commit', '-qm', 'candidate')
        head = git(repo, 'rev-parse', 'HEAD')
    write(repo, HERMES_PATH, "raise RuntimeError('outside candidate')\n")
    if mode == 'commit':
        git(repo, 'add', HERMES_PATH)
        git(repo, 'commit', '-qm', 'unrelated later helper')
    args = ['--staged'] if mode == 'staged' else ['--base-ref', base, '--head-ref', head]
    result = subprocess.run([sys.executable, str(repo / 'scripts/check_documentation.py'),
                             *args, '--json'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout
    report = json.loads(result.stdout)
    assert report['changed_files'] == ['src/service.py']
    assert {d['path'] for d in report['affected_documents']} == {
        'README.md', 'AGENTS.md', 'docs/README.md', 'docs/topic.md'}


@pytest.mark.parametrize('mode', ['staged', 'commit'])
def test_unstaged_helper_repair_cannot_hide_candidate_failure(repo, mode):
    base = git(repo, 'rev-parse', 'HEAD')
    original = (repo / HERMES_PATH).read_text()
    write(repo, HERMES_PATH, "raise RuntimeError('broken candidate helper')\n")
    git(repo, 'add', HERMES_PATH)
    if mode == 'commit':
        git(repo, 'commit', '-qm', 'broken helper candidate')
    write(repo, HERMES_PATH, original)
    args = ['--staged'] if mode == 'staged' else ['--base-ref', base, '--head-ref', 'HEAD']
    result = subprocess.run([sys.executable, str(repo / 'scripts/check_documentation.py'),
                             *args, '--json'], capture_output=True, text=True)
    assert result.returncode == 1, result.stdout
    assert 'broken candidate helper' in json.loads(result.stdout)['errors'][0]


def test_review_fingerprint_includes_helper_code(repo):
    before = inspect(repo, ['src/service.py'])
    with (repo / HERMES_PATH).open('a') as stream:
        stream.write('\n# Routing helper revision\n')
    after = inspect(repo, ['src/service.py'])
    assert before['fingerprint'] != after['fingerprint']


def test_code_impact_is_separate_from_structural_validation(repo):
    report = inspect(repo, ['src/service.py'])
    assert not report['errors']
    assert {x['path'] for x in report['affected_documents']} == {'README.md', 'AGENTS.md', 'docs/README.md', 'docs/topic.md'}
    assert all(x['review']['status'] == 'needs_review' for x in report['affected_documents'])
    assert inspect(repo, ['src/service.py'], require_review=True)['errors']


def test_reviewed_no_change_requires_reason_and_candidate_fingerprint(repo):
    report = inspect(repo, ['src/service.py'])
    review = {'fingerprint': report['fingerprint'], 'documents': [
        {'path': x['path'], 'status': 'reviewed_no_change', 'reason': 'Implementation refactor preserves the documented interface.'}
        for x in report['affected_documents']]}
    assert not inspect(repo, ['src/service.py'], review=review, require_review=True)['errors']
    write(repo, 'src/service.py', 'VALUE = 2\n')
    assert any('fingerprint' in e for e in inspect(repo, ['src/service.py'], review=review)['errors'])


def test_false_updated_declaration_is_rejected(repo):
    report = inspect(repo, ['src/service.py'])
    review = {'fingerprint': report['fingerprint'], 'documents': [
        {'path': 'docs/topic.md', 'status': 'updated', 'reason': 'Claim without a document change.'}]}
    assert any('absent from the changed scope' in e for e in inspect(repo, ['src/service.py'], review=review)['errors'])


@pytest.mark.parametrize('review', [[], {'documents': [None]}])
def test_malformed_review_fails_with_actionable_error(repo, review):
    with pytest.raises(ValueError, match='must be an object'):
        inspect(repo, ['src/service.py'], review=review)


def test_invalid_git_reference_fails_closed(repo, capsys):
    assert check.main(['--project-root', str(repo), '--base-ref', 'does-not-exist', '--json']) == 1
    assert json.loads(capsys.readouterr().out)['status'] == 'failed'


def test_pr_uses_merge_base_and_head_tree(repo, capsys):
    base = git(repo, 'rev-parse', 'HEAD')
    git(repo, 'checkout', '-qb', 'target')
    write(repo, 'target-only.txt', 'target\n')
    git(repo, 'add', '.')
    git(repo, 'commit', '-qm', 'target advances')
    git(repo, 'checkout', '-qb', 'feature', base)
    write(repo, 'src/service.py', 'VALUE = 2\n')
    git(repo, 'add', '.')
    git(repo, 'commit', '-qm', 'feature')
    write(repo, 'README.md', '[unstaged](absent.md)\n')
    assert check.main(['--project-root', str(repo), '--base-ref', 'target', '--json']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['changed_files'] == ['src/service.py']
    assert report['candidate'] == 'commit'


def test_rename_scope_preserves_old_and_new_paths(repo, capsys):
    base = git(repo, 'rev-parse', 'HEAD')
    git(repo, 'mv', 'src/service.py', 'src/renamed.py')
    git(repo, 'commit', '-qm', 'rename')
    assert check.main(['--project-root', str(repo), '--base-ref', base, '--json']) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report['changed_files']) == {'src/service.py', 'src/renamed.py'}


@pytest.mark.parametrize('mode', ['worktree', 'staged', 'commit'])
def test_deleted_document_requires_explicit_review(repo, capsys, mode):
    base = git(repo, 'rev-parse', 'HEAD')
    (repo / 'docs/topic.md').unlink()
    mapping = json.loads((repo / 'docs/repo_map.json').read_text())
    mapping['documentation']['documents'] = [
        d for d in mapping['documentation']['documents'] if d['path'] != 'docs/topic.md']
    write(repo, 'docs/repo_map.json', json.dumps(mapping))
    write(repo, 'docs/README.md', '# Index\n\n' + check.document_index(mapping['documentation']['documents']) + '\n')
    if mode != 'worktree':
        git(repo, 'add', 'docs')
    if mode == 'commit':
        git(repo, 'commit', '-qm', 'retire document')
    args = ['--staged'] if mode == 'staged' else ['--base-ref', base] if mode == 'commit' else []
    assert check.main(['--project-root', str(repo), *args, '--json']) == 0
    report = json.loads(capsys.readouterr().out)
    retired = [d for d in report['affected_documents'] if d['path'] == 'docs/topic.md']
    assert retired == [{'path': 'docs/topic.md', 'role': 'deleted',
                        'caused_by': ['docs/topic.md'], 'review': {'status': 'needs_review'}}]
    review = {'fingerprint': report['fingerprint'], 'documents': [
        {'path': d['path'], 'status': 'reviewed_no_change', 'reason': 'Current contract is unchanged.'}
        for d in report['affected_documents'] if d['path'] != 'docs/topic.md']}
    candidate = check.Candidate(repo, mode)
    result = check.inspect(candidate, report['changed_files'], review, require_review=True)
    assert 'docs/topic.md: documentation impact review is incomplete' in result['errors']
    review['documents'].append({'path': 'docs/topic.md', 'status': 'reviewed_no_change',
                                'reason': 'Retired duplicate; surviving contract and references reviewed.'})
    assert not check.inspect(candidate, report['changed_files'], review, require_review=True)['errors']


def test_document_rename_keeps_old_path_in_review_scope(repo, capsys):
    base = git(repo, 'rev-parse', 'HEAD')
    git(repo, 'mv', 'docs/topic.md', 'docs/renamed.md')
    mapping = json.loads((repo / 'docs/repo_map.json').read_text())
    for d in mapping['documentation']['documents']:
        if d['path'] == 'docs/topic.md':
            d['path'] = 'docs/renamed.md'
    write(repo, 'docs/repo_map.json', json.dumps(mapping))
    write(repo, 'docs/README.md', '# Index\n\n' + check.document_index(mapping['documentation']['documents']) + '\n')
    git(repo, 'add', 'docs')
    git(repo, 'commit', '-qm', 'move document and index')
    assert check.main(['--project-root', str(repo), '--base-ref', base, '--json']) == 0
    report = json.loads(capsys.readouterr().out)
    roles = {d['path']: d['role'] for d in report['affected_documents']}
    assert roles['docs/topic.md'] == 'deleted'
    assert roles['docs/renamed.md'] == 'topic'


def test_readonly_check_does_not_rewrite_documents_or_index(repo):
    paths = [p for p in repo.rglob('*') if p.is_file()]
    before = {str(p): p.read_bytes() for p in paths}
    assert check.main(['--project-root', str(repo), '--changed-file', 'src/service.py']) == 0
    assert before == {str(p): p.read_bytes() for p in paths}
