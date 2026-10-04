"""Offline fixtures; no Obsidian process, network, production data or secrets."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

SPEC = importlib.util.spec_from_file_location('knowledge_base', Path(__file__).parents[1] / 'knowledge_base.py')
kb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kb)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'AGENTS.md').write_text('# Rules\n')
    (tmp_path / 'README.md').write_text('# Root\nroot-only marker')
    (tmp_path / 'docs/README.md').write_text('# Docs\ndocs-only marker')
    (tmp_path / 'docs/contract.md').write_text('# NIW\nDaily NIW model, 月频 is unverified.')
    catalog = {'documentation': {'documents': [
        {'path': p, 'title': title, 'role': role, 'status': status}
        for p, title, role, status in [
            ('README.md', 'Root', 'overview', 'active'),
            ('docs/README.md', 'Docs', 'index', 'active'),
            ('docs/contract.md', 'NIW contract', 'contract', 'active')]]}}
    (tmp_path / 'docs/repo_map.json').write_text(json.dumps(catalog))
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    subprocess.run(['git', '-C', str(tmp_path), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture'], check=True)
    return tmp_path


def fingerprint(root, path):
    return path + '::sha256:' + hashlib.sha256((root / path).read_bytes()).hexdigest()


def card(root, slug='niw', deps=None, result='supported', state='partial'):
    path = kb.draft(root, 'claim', slug, 'NIW 月频边界', 'business')
    meta, _ = kb.metadata((root / path).read_text())
    meta.update(review_state=state, result=result, scope='Static fixture only', reviewed_at='2026-10-04',
                reviewed_by='Fixture reviewer', dependencies=deps if deps is not None else [fingerprint(root, 'docs/contract.md')])
    body = '# NIW 月频边界\n\n' + '\n\n'.join(s + '\n\n仅静态证据，尚未运行。' for s in ('## 主张', '## 证据', '## 四轴', '## 限制', '## 复核条件'))
    (root / path).write_text(kb.encode(meta, body))
    return path


def topic(root, slug='project-topic'):
    path = card(root, slug)
    meta, _ = kb.metadata((root / path).read_text())
    meta.update(type='topic', title='平台目标与当前范围', review_state='partial')
    target = root / 'docs/wiki/topics' / (slug + '.md')
    target.parent.mkdir(parents=True, exist_ok=True)
    body = '# 平台目标与当前范围\n\n' + '\n\n'.join(s + '\n\n仅基于fixture契约的有源摘要。' for s in ('## 这页回答什么', '## 核心认识', '## 当前与边界', '## 依据与继续阅读', '## 复核条件'))
    target.write_text(kb.encode(meta, body))
    (root / path).unlink()
    return target.relative_to(root).as_posix()


def reviewed_source(root, path):
    meta, body = kb.metadata((root / path).read_text())
    meta.update(review_state='reviewed', result='supported', scope='Fixture source provenance only', reviewed_at='2026-10-04',
                reviewed_by='Fixture reviewer', dependencies=[fingerprint(root, 'docs/contract.md')])
    body = body.replace('待填写', '原记录').replace('待核', '未确定')
    (root / path).write_text(kb.encode(meta, body))


def ingest(root, slug='source-a', uri='repo:docs/contract.md', version='1', summary='A bounded authorized fixture excerpt.', supersedes=None, distinct=False):
    return kb.intake(root, slug, 'Source fixture', 'business', 'repo_record', uri, version, summary, 'Fixture-owned excerpt', supersedes, distinct)


def test_draft_is_unverified_not_promotion(repo):
    path = kb.draft(repo, 'source', 'vendor-note', 'Vendor material', 'business', 'vendor_claim', 'https://example.invalid')
    meta, body = kb.metadata((repo / path).read_text())
    assert meta['review_state'] == 'pending' and meta['result'] == 'unverified'
    assert meta['dependencies'] == [] and meta['source_kind'] == 'vendor_claim'
    assert kb.inspect(repo)['untracked'] == 1
    assert kb.validate(meta, body, path) == []


def test_exclusive_create(repo):
    path = kb.draft(repo, 'claim', 'same', 'First', 'developer')
    original = (repo / path).read_bytes()
    with pytest.raises(FileExistsError):
        kb.draft(repo, 'claim', 'same', 'Overwrite', 'developer')
    assert (repo / path).read_bytes() == original


@pytest.mark.parametrize('slug', ['../escape', '/tmp/out', 'name.md', 'Upper', 'two words', ''])
def test_invalid_draft_id(repo, slug):
    with pytest.raises(ValueError):
        kb.draft(repo, 'claim', slug, 'Safe', 'developer')


def test_current_is_byte_match_not_test_proof(repo):
    path = card(repo)
    report = kb.inspect(repo)
    assert report['records'][0]['freshness'] == 'current'
    assert report['records'][0]['metadata']['evidence_kind'] == 'static'
    assert not report['invalid'] and not report['stale']
    hits = kb.search(repo, '月频 NIW')
    assert hits[0]['path'] == path and hits[0]['scope'] == 'Static fixture only'


def test_change_does_not_rewrite_card(repo):
    path = card(repo)
    original = (repo / path).read_bytes()
    (repo / 'docs/contract.md').write_text('Changed authority')
    assert kb.inspect(repo)['stale'] == 1
    hit = kb.search(repo, 'NIW')[0]
    assert hit['freshness'] == 'stale' and hit['needs_review']
    assert hit['result'] == 'supported'
    assert (repo / path).read_bytes() == original


def test_missing_dependency(repo):
    card(repo)
    (repo / 'docs/contract.md').unlink()
    report = kb.inspect(repo)
    assert report['stale'] == 1
    assert 'missing' in report['records'][0]['dependency_errors'][0]


def test_source_change_propagates(repo):
    source = card(repo, 'source-card')
    dependent = card(repo, 'dependent', [fingerprint(repo, source)])
    (repo / 'docs/contract.md').write_text('changed upstream')
    report = {r['path']: r for r in kb.inspect(repo)['records']}
    assert report[dependent]['freshness'] == 'stale'
    assert source in report[dependent]['changed_dependencies']


def test_cycle_is_invalid_even_with_mismatching_hashes(repo):
    a = card(repo, 'cycle-a')
    b = card(repo, 'cycle-b', [fingerprint(repo, a)])
    meta, body = kb.metadata((repo / a).read_text())
    meta['dependencies'] = [fingerprint(repo, b)]
    (repo / a).write_text(kb.encode(meta, body))
    report = kb.inspect(repo)
    assert report['invalid'] >= 1
    assert any('cyclic evidence dependency' in r['errors'] for r in report['records'])


def test_supported_requires_review_and_no_placeholder(repo):
    path = kb.draft(repo, 'claim', 'missing-evidence', 'Claim', 'business')
    meta, body = kb.metadata((repo / path).read_text())
    meta.update(result='supported', review_state='reviewed')
    errors = kb.validate(meta, body, path)
    assert any('dependencies' in error for error in errors)
    assert any('placeholder' in error for error in errors)


@pytest.mark.parametrize('path', ['../outside.md', '/tmp/outside.md', '.env', 'docs/.env', 'data/secret.md', 'frontend/node_modules/p/package.json'])
def test_rejects_sensitive_and_out_of_scope_paths(repo, path):
    with pytest.raises(ValueError):
        kb.safe_file(repo, path)


def test_symlink_file_and_directory_rejected(repo):
    (repo / 'docs/link.md').symlink_to(repo / 'docs/contract.md')
    with pytest.raises(ValueError, match='symlink'):
        kb.safe_file(repo, 'docs/link.md')
    (repo / 'docs/wiki').symlink_to(repo / 'docs', target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        kb.draft(repo, 'claim', 'test', 'No write', 'developer')


def test_fingerprint_requires_tracked_file(repo):
    (repo / 'docs/untracked.md').write_text('new')
    with pytest.raises(ValueError):
        kb.safe_file(repo, 'docs/untracked.md', tracked=True)
    assert kb.safe_file(repo, 'docs/contract.md', tracked=True).is_file()


def test_duplicate_readme_exact_paths(repo):
    assert kb.search(repo, 'root-only')[0]['path'] == 'README.md'
    assert kb.search(repo, 'docs-only')[0]['path'] == 'docs/README.md'


def test_search_limits_and_negative(repo):
    card(repo)
    assert len(kb.search(repo, 'NIW', 1)) == 1
    assert kb.search(repo, 'no such term') == []
    with pytest.raises(ValueError):
        kb.search(repo, '', 1)
    with pytest.raises(ValueError):
        kb.search(repo, 'NIW', 31)


def test_malformed_frontmatter_is_not_trusted(repo):
    path = card(repo)
    text = (repo / path).read_text().replace('review_state: "partial"', 'review_state: ["bad"]')
    (repo / path).write_text(text)
    assert kb.inspect(repo)['invalid'] == 1
    assert kb.search(repo, 'NIW')[0]['needs_review']


def test_duplicate_metadata_key_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        kb.metadata('---\nid: "one"\nid: "two"\n---\n')


def test_wrong_vault_stops_before_write(tmp_path):
    assert kb.main(['--root', str(tmp_path), 'draft', 'claim', 'id', '--title', 'x', '--domain', 'business']) == 2
    assert not (tmp_path / 'docs').exists()


def test_check_exit_code_and_readonly(repo):
    card(repo)
    assert kb.main(['--root', str(repo), 'check']) == 0
    (repo / 'docs/contract.md').write_text('changed')
    assert kb.main(['--root', str(repo), 'check']) == 1


def test_tree_watch_detects_new_files_without_reading_them(repo):
    path = card(repo)
    meta, body = kb.metadata((repo / path).read_text())
    meta['watch_globs'] = ['backend/**', 'frontend/src/**']
    (repo / path).write_text(kb.encode(meta, body))
    assert kb.inspect(repo)['stale'] == 0
    (repo / 'backend').mkdir()
    (repo / 'backend/new_runner.py').write_text('unexpected = True')
    report = kb.inspect(repo)
    assert report['stale'] == 1
    assert report['records'][0]['changed_watch_paths'] == ['backend/new_runner.py']
    assert kb.search(repo, 'NIW')[0]['needs_review']


@pytest.mark.parametrize('committed', [False, True], ids=['staged', 'committed'])
@pytest.mark.parametrize('watched_path', ['backend/only.py', 'shared/only.py'])
def test_tree_watch_preserves_both_rename_paths(repo, committed, watched_path):
    (repo / 'backend').mkdir()
    (repo / 'backend/only.py').write_text('value = 1\n')
    kb.git(repo, 'add', 'backend/only.py')
    kb.git(repo, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
           'commit', '-qm', 'source before rename')
    path = card(repo)
    meta, body = kb.metadata((repo / path).read_text())
    meta['watch_globs'] = [str(Path(watched_path).parent) + '/**']
    (repo / path).write_text(kb.encode(meta, body))
    (repo / 'shared').mkdir()
    kb.git(repo, 'mv', 'backend/only.py', 'shared/only.py')
    if committed:
        kb.git(repo, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
               'commit', '-qm', 'move source outside watched tree')
    assert kb.git(repo, 'diff', '--name-status', '-z', meta['source_revision'], '--') == (
        'R100\0backend/only.py\0shared/only.py\0')
    record = kb.inspect(repo)['records'][0]
    assert record['freshness'] == 'stale'
    assert record['changed_watch_paths'] == [watched_path]


def test_tree_watch_parses_copy_status_and_nul_delimited_paths(repo, monkeypatch):
    path = card(repo)
    meta, body = kb.metadata((repo / path).read_text())
    meta['watch_globs'] = ['backend/**', 'shared/**']
    (repo / path).write_text(kb.encode(meta, body))
    run_git = kb.git

    def copy_diff(root, *args):
        if args[0] == 'diff':
            assert '--name-status' in args and '-z' in args
            return 'C100\0backend/source\tname\n.py\0shared/copy name.py\0M\0backend/changed.py\0'
        return run_git(root, *args)

    monkeypatch.setattr(kb, 'git', copy_diff)
    record = kb.inspect(repo)['records'][0]
    assert record['freshness'] == 'stale'
    assert record['changed_watch_paths'] == [
        'backend/changed.py', 'backend/source\tname\n.py', 'shared/copy name.py']


def test_bad_tree_watch_baseline_fails_closed(repo):
    path = card(repo)
    meta, body = kb.metadata((repo / path).read_text())
    meta['watch_globs'], meta['source_revision'] = ['backend/**'], 'f' * 40
    (repo / path).write_text(kb.encode(meta, body))
    report = kb.inspect(repo)
    assert report['invalid'] == 1
    assert 'source_revision is not an available repository commit' in report['records'][0]['errors']


def fake_cli(monkeypatch, repo, responses, seen):
    def run(args, **kwargs):
        seen.append(args)
        output = 'path\t' + str(repo) + '\n' if args[2] == 'vault' else responses.get(args[2], '')
        return subprocess.CompletedProcess(args, 0, stdout=output, stderr='')
    monkeypatch.setattr(kb.subprocess, 'run', run)


def test_obsidian_exact_path_read_and_zero_exit_error(repo, monkeypatch):
    seen = []
    fake_cli(monkeypatch, repo, {'read': (repo / 'docs/README.md').read_text() + '\n'}, seen)
    assert kb.obsidian_read(repo, 'Explicit Vault', 'read', 'docs/README.md')['path'] == 'docs/README.md'
    assert seen[-1] == ['obsidian', 'vault=Explicit Vault', 'read', 'path=docs/README.md']
    fake_cli(monkeypatch, repo, {'read': 'Error: File not found.\n'}, seen)
    with pytest.raises(ValueError, match='read failed'):
        kb.obsidian_read(repo, 'Explicit Vault', 'read', 'docs/README.md')


def test_obsidian_wrong_vault_and_stale_disk_stop(repo, monkeypatch):
    seen = []
    fake_cli(monkeypatch, repo / 'elsewhere', {}, seen)
    with pytest.raises(ValueError, match='does not match'):
        kb.obsidian_read(repo, 'Wrong Vault', 'read', 'README.md')
    assert len(seen) == 1
    fake_cli(monkeypatch, repo, {'read': 'another active note'}, seen)
    with pytest.raises(ValueError, match='differs'):
        kb.obsidian_read(repo, 'Explicit Vault', 'read', 'README.md')


def test_obsidian_search_and_property(repo, monkeypatch):
    seen = []
    fake_cli(monkeypatch, repo, {'search': '["docs/contract.md"]', 'property:read': 'reviewed\n'}, seen)
    assert kb.obsidian_read(repo, 'Vault', 'search', 'NIW')['paths'] == ['docs/contract.md']
    assert kb.obsidian_read(repo, 'Vault', 'property', 'docs/contract.md', property_name='review_state')['output'].strip() == 'reviewed'
    with pytest.raises(ValueError, match='property'):
        kb.obsidian_read(repo, 'Vault', 'property', 'docs/contract.md')


def test_obsidian_native_error_exit_zero_and_no_mutation(repo, monkeypatch):
    def run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout='Vault not found.\n', stderr='')
    monkeypatch.setattr(kb.subprocess, 'run', run)
    with pytest.raises(ValueError, match='failed'):
        kb.obsidian_read(repo, 'Missing', 'read', 'AGENTS.md')
    with pytest.raises(ValueError, match='supported'):
        kb.obsidian_read(repo, 'Vault', 'delete', 'AGENTS.md')


@pytest.mark.parametrize('source_result,source_state', [('unverified', 'pending'), ('conflict', 'reviewed'), ('rejected', 'reviewed'), ('superseded', 'reviewed')])
def test_current_source_does_not_launder_unreviewed_evidence(repo, source_result, source_state):
    source = kb.draft(repo, 'source', 'uncertain-source', 'Source', 'business', uri='repo:docs/contract.md')
    meta, body = kb.metadata((repo / source).read_text())
    meta.update(result=source_result, review_state=source_state, scope='source only', reviewed_at='2026-10-04',
                reviewed_by='Fixture', dependencies=[fingerprint(repo, 'docs/contract.md')])
    body = body.replace('待填写', '原始记录').replace('待核', '尚不确定')
    (repo / source).write_text(kb.encode(meta, body))
    dependent = card(repo, 'dependent-claim', [fingerprint(repo, source)])
    third = card(repo, 'downstream-claim', [fingerprint(repo, dependent)])
    records = {r['path']: r for r in kb.inspect(repo)['records']}
    assert records[source]['freshness'] == records[dependent]['freshness'] == 'current'
    assert records[dependent]['needs_review'] and records[third]['needs_review']
    assert records[dependent]['upstream_needs_review'] == [source]
    assert all(h['needs_review'] for h in kb.search(repo, 'NIW') if h['path'] in (dependent, third))


def test_current_assistant_language_prefers_portable_fixture(repo):
    path = card(repo, 'portable')
    meta, body = kb.metadata((repo / path).read_text())
    meta['title'] = '当前助手：Portable 当前接入'
    (repo / path).write_text(kb.encode(meta, body + '\n当前助手读取现行接入契约，旧Harness是历史。'))
    assert kb.search(repo, '当前助手')[0]['path'] == path


@pytest.mark.parametrize('filename', ['*.md', '[ab].md'])
def test_fingerprint_uses_literal_git_pathspec(repo, filename):
    (repo / 'docs' / filename).write_text('not tracked')
    with pytest.raises(ValueError):
        kb.safe_file(repo, 'docs/' + filename, tracked=True)


def test_non_watch_card_requires_real_commit(repo):
    path = card(repo)
    meta, body = kb.metadata((repo / path).read_text())
    meta['source_revision'] = 'f' * 40
    (repo / path).write_text(kb.encode(meta, body))
    assert kb.inspect(repo)['invalid'] == 1
    assert kb.search(repo, 'NIW')[0]['needs_review']


def test_dot_dependency_is_rejected_without_traceback(repo):
    with pytest.raises(ValueError, match='unsafe path'):
        kb.safe_file(repo, '.')
    assert kb.main(['--root', str(repo), 'fingerprint', '.']) == 2


def test_tree_watch_includes_ignored_source_but_not_bytecode(repo):
    (repo / '.gitignore').write_text('local_test.py\n__pycache__/\n')
    path = card(repo)
    meta, body = kb.metadata((repo / path).read_text())
    meta['watch_globs'] = ['backend/**']
    (repo / path).write_text(kb.encode(meta, body))
    (repo / 'backend/__pycache__').mkdir(parents=True)
    (repo / 'backend/__pycache__/test.pyc').write_bytes(b'bytecode')
    assert kb.inspect(repo)['stale'] == 0
    (repo / 'backend/local_test.py').write_text('import openai')
    report = kb.inspect(repo)
    assert report['stale'] == 1
    assert report['records'][0]['changed_watch_paths'] == ['backend/local_test.py']


def test_obsidian_error_on_stderr_is_not_success(repo, monkeypatch):
    def run(args, **kwargs):
        if args[2] == 'vault':
            return subprocess.CompletedProcess(args, 0, stdout='path\t' + str(repo) + '\n', stderr='')
        return subprocess.CompletedProcess(args, 0, stdout='', stderr='Error: File not found.')
    monkeypatch.setattr(kb.subprocess, 'run', run)
    with pytest.raises(ValueError, match='failed'):
        kb.obsidian_read(repo, 'Vault', 'backlinks', 'docs/README.md')


def test_intake_identity_dedup_and_content_warning(repo):
    first = ingest(repo)
    assert first['created']
    same = ingest(repo, 'another-slug')
    assert same['status'] == 'duplicate' and same['path'] == first['path']
    assert ingest(repo, 'other-source', 'https://example.invalid/other')['status'] == 'possible_duplicate_content'
    assert ingest(repo, 'distinct-source', 'https://example.invalid/other', distinct=True)['created']
    meta, _ = kb.metadata((repo / first['path']).read_text())
    assert meta['result'] == 'unverified' and meta['review_state'] == 'pending'


def test_same_source_version_different_content_never_overwrites(repo):
    first = ingest(repo)
    before = (repo / first['path']).read_bytes()
    with pytest.raises(ValueError, match='different text'):
        ingest(repo, 'another-slug', summary='Changed excerpt')
    assert (repo / first['path']).read_bytes() == before


def test_intake_normalizes_version_in_metadata_and_body(repo):
    first = ingest(repo, version=' \tv1\n ')
    meta, body = kb.metadata((repo / first['path']).read_text())
    assert meta['source_version'] == 'v1'
    assert '来源版本：v1\n\n以下是获准记录的摘要' in body


@pytest.mark.parametrize('existing_version', ['v1', ' \tv1\n '], ids=['canonical', 'legacy-padded'])
@pytest.mark.parametrize('incoming_version', ['v1', ' \tv1\n '], ids=['canonical', 'padded'])
def test_intake_version_whitespace_cannot_bypass_identity(repo, existing_version, incoming_version):
    first = ingest(repo, version='v1')
    target = repo / first['path']
    meta, body = kb.metadata(target.read_text())
    meta['source_version'] = existing_version
    target.write_text(kb.encode(meta, body))
    before = target.read_bytes()
    same = ingest(repo, 'another-slug', version=incoming_version)
    assert same['status'] == 'duplicate' and same['path'] == first['path']
    with pytest.raises(ValueError, match='different text'):
        ingest(repo, 'changed-slug', version=incoming_version, summary='Changed excerpt')
    assert target.read_bytes() == before
    assert kb.note_paths(repo) == [first['path']]


def test_intake_preserves_edit_after_draft_read(repo, monkeypatch):
    parse = kb.metadata
    target = repo / 'docs/wiki/sources/source-a.md'
    edited_bytes = None

    def edit_after_parse(text):
        nonlocal edited_bytes
        result = parse(text)
        if target.exists() and edited_bytes is None:
            edited_bytes = target.read_bytes() + b'\nExternal editor note.\n'
            target.write_bytes(edited_bytes)
        return result

    monkeypatch.setattr(kb, 'metadata', edit_after_parse)
    with pytest.raises(ValueError, match='concurrent change'):
        ingest(repo)
    assert target.read_bytes() == edited_bytes


def test_intake_expected_hash_uses_exact_draft_bytes(repo, monkeypatch):
    create = kb.draft

    def crlf_draft(*args, **kwargs):
        path = create(*args, **kwargs)
        target = repo / path
        target.write_bytes(target.read_bytes().replace(b'\n', b'\r\n'))
        return path

    monkeypatch.setattr(kb, 'draft', crlf_draft)
    assert ingest(repo)['created']


@pytest.mark.parametrize('uri', ['https://name:password@example.invalid/a', 'https://example.invalid/a?token=x', 'https://example.invalid/?X-Amz-Signature=x', 'file:///private/account.pdf', 'private:/path/to/account', 'https://example.invalid/\nsecret'])
def test_intake_rejects_credential_and_private_path_uris(repo, uri):
    with pytest.raises(ValueError):
        ingest(repo, uri=uri)
    assert kb.note_paths(repo) == []


def test_canonical_uri_keeps_versions_and_fragments():
    assert kb.canonical_source('HTTPS://EXAMPLE.invalid/p?b=2&a=1#section') == 'https://example.invalid/p?a=1&b=2#section'
    assert kb.canonical_source('https://example.invalid/p?v=1') != kb.canonical_source('https://example.invalid/p?v=2')


def test_explicit_source_supersession_invalidates_downstream(repo):
    old = ingest(repo)['path']
    reviewed_source(repo, old)
    dependent = card(repo, 'dependent', [fingerprint(repo, old)])
    assert ingest(repo, 'source-v2', version='2', summary='A revised source excerpt', supersedes=old)['created']
    records = {r['path']: r for r in kb.inspect(repo)['records']}
    assert records[old]['freshness'] == records[dependent]['freshness'] == 'stale'
    assert records[old]['metadata']['result'] == 'supported'


def test_complete_lifecycle_with_explicit_human_review(repo):
    source = ingest(repo)['path']
    claim = card(repo, 'qualified-claim', [fingerprint(repo, source)])
    target = topic(repo)
    original_authority = (repo / 'docs/contract.md').read_bytes()
    with pytest.raises(ValueError, match='needs review'):
        kb.integrate(repo, claim, target, 'Scoped integration', 'Reviewer', 'Relevant result', kb.digest(repo / target))
    reviewed_source(repo, source)
    meta, body = kb.metadata((repo / claim).read_text())
    meta['dependencies'] = [fingerprint(repo, source)]
    (repo / claim).write_text(kb.encode(meta, body))
    before = kb.digest(repo / target)
    created = kb.integrate(repo, claim, target, 'Scoped integration', 'Reviewer', 'Relevant result', before)
    assert created['status'] == 'integrated'
    assert kb.integrate(repo, claim, target, 'Scoped integration', 'Reviewer', 'Relevant result', created['after_sha256'])['status'] == 'already_integrated'
    assert not kb.queue(repo)['claims_without_topic_integration']
    event = kb.feedback(repo, source, 'New counterevidence needs review', 'Researcher', kb.digest(repo / source))
    records = {r['path']: r for r in kb.inspect(repo)['records']}
    assert records[claim]['freshness'] == records[target]['freshness'] == 'stale'
    assert records[source]['metadata']['result'] == 'supported'
    assert len(kb.queue(repo)['open_feedback']) == 1
    kb.feedback(repo, source, 'Counterevidence documented; review still pending', 'Reviewer', kb.digest(repo / source), event['feedback_id'], 'resolved')
    assert kb.queue(repo)['open_feedback'] == []
    assert kb.inspect(repo)['stale'] >= 1
    assert (repo / 'docs/contract.md').read_bytes() == original_authority


def test_integration_preserves_concurrent_edit_and_authorities(repo):
    claim, target = card(repo), topic(repo)
    with pytest.raises(ValueError, match='concurrent'):
        kb.integrate(repo, claim, target, 'Bounded', 'Reviewer', 'Reason', '0' * 64)
    with pytest.raises(ValueError, match='derived topic'):
        kb.integrate(repo, claim, 'docs/contract.md', 'Bounded', 'Reviewer', 'Reason', kb.digest(repo / 'docs/contract.md'))
    assert 'KB-INTEGRATION' not in (repo / target).read_text()


def test_integration_retry_rejects_prewrite_hash(repo):
    claim, target = card(repo), topic(repo)
    before = kb.digest(repo / target)
    kb.integrate(repo, claim, target, 'Bounded', 'Reviewer', 'Reason', before)
    recorded = (repo / target).read_bytes()
    with pytest.raises(ValueError, match='concurrent change'):
        kb.integrate(repo, claim, target, 'Bounded', 'Reviewer', 'Reason', before)
    assert (repo / target).read_bytes() == recorded


def test_integration_retry_requires_current_hash_and_preserves_external_edits(repo):
    claim, target = card(repo), topic(repo)
    created = kb.integrate(repo, claim, target, 'Bounded', 'Reviewer', 'Reason', kb.digest(repo / target))
    recorded = (repo / target).read_bytes()
    assert kb.integrate(repo, claim, target, 'Bounded', 'Reviewer', 'Reason', created['after_sha256'])['status'] == 'already_integrated'
    edited = recorded + b'\nExternal note outside the integration block.\n'
    (repo / target).write_bytes(edited)
    with pytest.raises(ValueError, match='concurrent change'):
        kb.integrate(repo, claim, target, 'Bounded', 'Reviewer', 'Reason', created['after_sha256'])
    assert kb.integrate(repo, claim, target, 'Bounded', 'Reviewer', 'Reason', kb.digest(repo / target))['status'] == 'already_integrated'
    assert (repo / target).read_bytes() == edited


def test_integration_retry_rechecks_target_before_return(repo, monkeypatch):
    claim, target = card(repo), topic(repo)
    created = kb.integrate(repo, claim, target, 'Bounded', 'Reviewer', 'Reason', kb.digest(repo / target))
    parse = kb.metadata
    reads = 0
    edited = None

    def edit_after_target_parse(text):
        nonlocal reads, edited
        result = parse(text)
        if '<!-- KB-INTEGRATION:niw:BEGIN -->' in text:
            reads += 1
            if reads == 2:  # After inspect(), edit the bytes used by the retry itself.
                edited = (repo / target).read_bytes() + b'\nConcurrent external note.\n'
                (repo / target).write_bytes(edited)
        return result

    monkeypatch.setattr(kb, 'metadata', edit_after_target_parse)
    with pytest.raises(ValueError, match='concurrent change'):
        kb.integrate(repo, claim, target, 'Bounded', 'Reviewer', 'Reason', created['after_sha256'])
    assert edited is not None and (repo / target).read_bytes() == edited


@pytest.mark.parametrize('field', ['summary', 'reviewer', 'reason', 'title', 'scope'])
@pytest.mark.parametrize('phase', ['BEGIN', 'END'])
def test_integration_rejects_reserved_markers_before_first_write(repo, field, phase):
    claim, target = card(repo), topic(repo)
    marker = f'<!-- KB-INTEGRATION:niw:{phase} -->'
    fields = {'summary': 'Bounded', 'reviewer': 'Reviewer', 'reason': 'Reason'}
    if field in fields:
        fields[field] += ' ' + marker
    else:
        meta, body = kb.metadata((repo / claim).read_text())
        meta[field] += ' ' + marker
        (repo / claim).write_text(kb.encode(meta, body))
    before = (repo / target).read_bytes()
    with pytest.raises(ValueError, match='reserved integration marker'):
        kb.integrate(repo, claim, target, **fields, expected=kb.digest(repo / target))
    assert (repo / target).read_bytes() == before


@pytest.mark.parametrize('field', ['reviewer', 'reason', 'summary'])
def test_integration_rejects_ambiguous_receipt_fields_before_first_write(repo, field):
    claim, target = card(repo), topic(repo)
    fields = {'summary': 'Bounded', 'reviewer': 'Reviewer', 'reason': 'Reason'}
    if field == 'summary':
        fields[field] += '\n\n依据：[NIW 月频边界](../claims/niw.md)；范围：Static fixture only\n\n整合审阅：'
    else:
        fields[field] += '；日期：2002-03-04；理由：Another field'
    before = (repo / target).read_bytes()
    with pytest.raises(ValueError, match='ambiguous receipt delimiters'):
        kb.integrate(repo, claim, target, **fields, expected=kb.digest(repo / target))
    assert (repo / target).read_bytes() == before


@pytest.mark.parametrize('changed', [
    {'summary': 'thirty-two-node audit budget'},
    {'summary': 'Static fixture only'},
    {'reviewer': 'Other reviewer'},
    {'reason': 'Different reason'},
], ids=['summary-substring', 'summary-matches-scope', 'reviewer', 'reason'])
def test_integration_retry_requires_exact_request_fields(repo, changed):
    claim, target = card(repo), topic(repo)
    before = kb.digest(repo / target)
    fields = {'summary': 'Coverage has an explicit thirty-two-node audit budget.',
              'reviewer': 'Reviewer', 'reason': 'Relevant result'}
    kb.integrate(repo, claim, target, **fields, expected=before)
    recorded = (repo / target).read_bytes()
    with pytest.raises(ValueError, match='existing integration differs'):
        kb.integrate(repo, claim, target, **(fields | changed), expected=kb.digest(repo / target))
    assert (repo / target).read_bytes() == recorded


@pytest.mark.parametrize('damage', ['missing-end', 'duplicate-block', 'duplicate-begin', 'duplicate-end'])
def test_integration_retry_rejects_ambiguous_or_incomplete_blocks(repo, damage):
    claim, target = card(repo), topic(repo)
    before = kb.digest(repo / target)
    kb.integrate(repo, claim, target, 'Bounded', 'Reviewer', 'Reason', before)
    text = (repo / target).read_text()
    begin = '<!-- KB-INTEGRATION:niw:BEGIN -->'
    end = '<!-- KB-INTEGRATION:niw:END -->'
    if damage == 'missing-end':
        text = text.replace(end, '')
    elif damage == 'duplicate-block':
        text += '\n' + begin + text.split(begin, 1)[1]
    else:
        text += '\n' + (begin if damage == 'duplicate-begin' else end) + '\n'
    (repo / target).write_text(text)
    recorded = (repo / target).read_bytes()
    with pytest.raises(ValueError, match='existing integration differs'):
        kb.integrate(repo, claim, target, 'Bounded', 'Reviewer', 'Reason', kb.digest(repo / target))
    assert (repo / target).read_bytes() == recorded


def test_integration_exact_retry_preserves_original_date_hash_and_multiline_fields(repo, monkeypatch):
    claim, target = card(repo), topic(repo)
    real_date = kb.date

    class EarlierDate(real_date):
        @classmethod
        def today(cls):
            return cls(2001, 2, 3)

    before = kb.digest(repo / target)
    summary = 'First paragraph.\n\nSecond paragraph, with (parentheses) and [brackets].'
    reviewer, reason = 'Reviewer\nTeam', 'Relevant result\nwith a bounded scope.'
    monkeypatch.setattr(kb, 'date', EarlierDate)
    created = kb.integrate(repo, claim, target, summary, reviewer, reason, before)
    recorded = (repo / target).read_bytes()
    monkeypatch.setattr(kb, 'date', real_date)
    result = kb.integrate(repo, claim, target, ' \n' + summary + '\n ', reviewer, reason, created['after_sha256'])
    assert result['status'] == 'already_integrated'
    assert (repo / target).read_bytes() == recorded


@pytest.mark.parametrize('embedded_in', ['reviewer', 'reason'])
def test_integration_retry_rejects_ambiguous_review_field_separators(repo, embedded_in):
    claim, target = card(repo), topic(repo)
    before = kb.digest(repo / target)
    today = kb.date.today().isoformat()
    if embedded_in == 'reason':
        reviewer, reason = 'Reviewer', 'Reason；日期：2002-03-04；理由：Other reason'
        retry_reviewer, retry_reason = f'Reviewer；日期：{today}；理由：Reason', 'Other reason'
    else:
        reviewer, reason = 'Reviewer；日期：2002-03-04；理由：Other reviewer', 'Reason'
        retry_reviewer, retry_reason = 'Reviewer', f'Other reviewer；日期：{today}；理由：Reason'
    kb.integrate(repo, claim, target, 'Bounded', 'Reviewer', 'Reason', before)
    text = (repo / target).read_text().replace(
        f'整合审阅：Reviewer；日期：{today}；理由：Reason',
        f'整合审阅：{reviewer}；日期：{today}；理由：{reason}')
    (repo / target).write_text(text)
    recorded = (repo / target).read_bytes()
    with pytest.raises(ValueError, match='existing integration differs'):
        kb.integrate(repo, claim, target, 'Bounded', retry_reviewer, retry_reason, kb.digest(repo / target))
    assert (repo / target).read_bytes() == recorded


def test_feedback_is_idempotent_and_hash_guarded(repo):
    path = card(repo)
    before = kb.digest(repo / path)
    event = kb.feedback(repo, path, 'Check new evidence', 'Reviewer', before)
    assert kb.feedback(repo, path, 'Check new evidence', 'Reviewer', kb.digest(repo / path))['status'] == 'already_recorded'
    with pytest.raises(ValueError, match='concurrent'):
        kb.feedback(repo, path, 'Different feedback', 'Reviewer', before)
    assert event['feedback_id'] in (repo / path).read_text()


@pytest.mark.parametrize('edit', [False, True], ids=['prewrite-hash', 'external-edit'])
def test_feedback_duplicate_requires_current_target_hash(repo, edit):
    path = card(repo)
    before = kb.digest(repo / path)
    kb.feedback(repo, path, 'Check evidence', 'Reviewer', before)
    expected = kb.digest(repo / path) if edit else before
    if edit:
        (repo / path).write_text((repo / path).read_text() + '\nExternal note.\n')
    recorded = (repo / path).read_bytes()
    with pytest.raises(ValueError, match='concurrent change'):
        kb.feedback(repo, path, 'Check evidence', 'Reviewer', expected)
    assert (repo / path).read_bytes() == recorded
    assert kb.feedback(repo, path, 'Check evidence', 'Reviewer', kb.digest(repo / path))['status'] == 'already_recorded'


@pytest.mark.parametrize('form', ['bare', 'fenced', 'quoted', 'wrong-hash', 'missing-dependency'])
def test_queue_requires_a_valid_current_integration_receipt(repo, form):
    claim, target = card(repo), topic(repo)
    if form == 'bare':
        text = (repo / target).read_text() + '\n<!-- KB-INTEGRATION:niw:BEGIN -->\n'
    else:
        kb.integrate(repo, claim, target, 'Bounded', 'Reviewer', 'Reason', kb.digest(repo / target))
        text = (repo / target).read_text()
        begin = '<!-- KB-INTEGRATION:niw:BEGIN -->'
        prefix, block = text.split(begin, 1)
        block = begin + block
        if form == 'fenced':
            text = prefix + '```markdown\n' + block + '```\n'
        elif form == 'quoted':
            text = prefix + '\n'.join('> ' + line for line in block.splitlines()) + '\n'
        elif form == 'wrong-hash':
            text = text.replace('claim SHA-256：' + kb.digest(repo / claim), 'claim SHA-256：' + '0' * 64)
        else:
            meta, body = kb.metadata(text)
            meta['dependencies'] = [fingerprint(repo, 'docs/contract.md')]
            text = kb.encode(meta, body)
    (repo / target).write_text(text)
    assert claim in kb.queue(repo)['claims_without_topic_integration']


def test_integration_retry_does_not_trust_a_fenced_receipt(repo):
    claim, target = card(repo), topic(repo)
    kb.integrate(repo, claim, target, 'Bounded', 'Reviewer', 'Reason', kb.digest(repo / target))
    text = (repo / target).read_text()
    begin = '<!-- KB-INTEGRATION:niw:BEGIN -->'
    prefix, block = text.split(begin, 1)
    (repo / target).write_text(prefix + '```markdown\n' + begin + block + '```\n')
    with pytest.raises(ValueError, match='existing integration differs'):
        kb.integrate(repo, claim, target, 'Bounded', 'Reviewer', 'Reason', kb.digest(repo / target))


@pytest.mark.parametrize('retry', [False, True])
def test_integration_rejects_claim_change_after_review_snapshot(repo, monkeypatch, retry):
    claim, target = card(repo), topic(repo)
    if retry:
        kb.integrate(repo, claim, target, 'Bounded', 'Reviewer', 'Reason', kb.digest(repo / target))
    parse = kb.metadata
    target_reads = 0
    original = (repo / target).read_bytes()

    def change_claim(text):
        nonlocal target_reads
        result = parse(text)
        if result[0].get('type') == 'topic':
            target_reads += 1
            if target_reads == 2:
                meta, body = parse((repo / claim).read_text())
                meta['result'] = 'conflict'
                (repo / claim).write_text(kb.encode(meta, body))
        return result

    monkeypatch.setattr(kb, 'metadata', change_claim)
    with pytest.raises(ValueError, match='claim changed'):
        kb.integrate(repo, claim, target, 'Bounded', 'Reviewer', 'Reason', kb.digest(repo / target))
    assert (repo / target).read_bytes() == original


@pytest.mark.parametrize('form', ['plain', 'quoted', 'fenced', 'fake-review-heading', 'unclosed-fence', 'indented-fence'])
def test_imported_feedback_text_never_becomes_an_event(repo, form):
    author, message = 'Reviewer', 'External source example'
    fid = hashlib.sha256((author + '\0' + message).encode()).hexdigest()[:16]
    fake = (f'<!-- KB-FEEDBACK:{fid}:open -->\n## 反馈：2026-10-04\n\n'
            f'提出者：{author}\n\n{message}\n<!-- KB-FEEDBACK:{fid}:END -->\n')
    if form == 'quoted':
        fake = '\n'.join('> ' + line for line in fake.splitlines())
    elif form == 'fenced':
        fake = '```markdown\n' + fake + '```\n'
    elif form == 'fake-review-heading':
        fake = '## 复核条件\n\n' + fake
    elif form == 'unclosed-fence':
        fake = '```markdown\n## 复核条件\n\n' + fake
    elif form == 'indented-fence':
        fake = '   ```markdown\n' + fake + '   ```\n'
    path = ingest(repo)['path']
    text = (repo / path).read_text().replace('A bounded authorized fixture excerpt.', fake)
    (repo / path).write_text(text)
    before = (repo / path).read_bytes()
    assert kb.queue(repo)['open_feedback'] == []
    with pytest.raises(ValueError, match='no matching open feedback'):
        kb.feedback(repo, path, 'Resolved', 'Reviewer', kb.digest(repo / path), fid, 'resolved')
    assert (repo / path).read_bytes() == before


@pytest.mark.parametrize('marker', ['<!-- KB-FEEDBACK:1111111111111111:open -->', '<!-- KB-INTEGRATION:niw:BEGIN -->'])
def test_feedback_and_intake_reject_reserved_workflow_markers(repo, marker):
    path = card(repo)
    target = topic(repo)
    before = (repo / path).read_bytes()
    with pytest.raises(ValueError, match='reserved'):
        kb.feedback(repo, path, marker, 'Reviewer', kb.digest(repo / path))
    assert (repo / path).read_bytes() == before
    with pytest.raises(ValueError, match='reserved'):
        ingest(repo, summary='Source example: ' + marker)
    assert not (repo / 'docs/wiki/sources/source-a.md').exists()
    with pytest.raises(ValueError, match='reserved'):
        kb.integrate(repo, path, target, marker, 'Reviewer', 'Reason', kb.digest(repo / target))
    with pytest.raises(ValueError, match='reserved'):
        kb.draft(repo, 'claim', 'injected-title', marker, 'business')
    assert not (repo / 'docs/wiki/claims/injected-title.md').exists()


def test_feedback_resolver_changes_the_real_event_not_an_inline_example(repo):
    path = card(repo)
    event = kb.feedback(repo, path, 'Check evidence', 'Reviewer', kb.digest(repo / path))
    marker = f'<!-- KB-FEEDBACK:{event["feedback_id"]}:open -->'
    text = (repo / path).read_text().replace('## 复核条件', 'Inline example `' + marker + '`\n\n## 复核条件', 1)
    (repo / path).write_text(text)
    kb.feedback(repo, path, 'Reviewed evidence', 'Reviewer', kb.digest(repo / path), event['feedback_id'], 'resolved')
    assert 'Inline example `' + marker + '`' in (repo / path).read_text()
    assert kb.queue(repo)['open_feedback'] == []


def test_feedback_duplicate_does_not_trust_a_fenced_record(repo):
    path = card(repo)
    event = kb.feedback(repo, path, 'Check evidence', 'Reviewer', kb.digest(repo / path))
    marker = f'<!-- KB-FEEDBACK:{event["feedback_id"]}:open -->'
    text = (repo / path).read_text()
    prefix, block = text.split(marker, 1)
    (repo / path).write_text(prefix + '```markdown\n' + marker + block + '```\n')
    assert kb.queue(repo)['open_feedback'] == []
    result = kb.feedback(repo, path, 'Check evidence', 'Reviewer', kb.digest(repo / path))
    assert result['status'] == 'open'


@pytest.mark.parametrize('operation', ['feedback', 'integrate'])
@pytest.mark.parametrize('content', ['## 复核条件\n\nA heading inside record data.', '```markdown\nUnclosed example.'])
def test_workflow_writers_validate_complete_markdown_candidate(repo, operation, content):
    claim, target = card(repo), topic(repo)
    path = claim if operation == 'feedback' else target
    before = (repo / path).read_bytes()

    def write():
        if operation == 'feedback':
            return kb.feedback(repo, path, content, 'Reviewer', kb.digest(repo / path))
        return kb.integrate(repo, claim, target, content, 'Reviewer', 'Reason', kb.digest(repo / path))

    if content.startswith('```'):
        with pytest.raises(ValueError, match='ambiguous or hidden'):
            write()
        assert (repo / path).read_bytes() == before
    else:
        result = write()
        if operation == 'feedback':
            assert kb.queue(repo)['open_feedback'] == [{'path': path, 'id': result['feedback_id']}]
        else:
            assert claim not in kb.queue(repo)['claims_without_topic_integration']


@pytest.mark.parametrize('damage', ['missing-end', 'duplicate', 'crossed', 'message-changed'])
def test_feedback_reader_rejects_malformed_records_and_reports_review(repo, damage):
    path = card(repo)
    first = kb.feedback(repo, path, 'First evidence', 'Reviewer', kb.digest(repo / path))
    second = kb.feedback(repo, path, 'Second evidence', 'Reviewer', kb.digest(repo / path))
    text = (repo / path).read_text()
    first_start = f'<!-- KB-FEEDBACK:{first["feedback_id"]}:open -->'
    first_end = f'<!-- KB-FEEDBACK:{first["feedback_id"]}:END -->'
    second_start = f'<!-- KB-FEEDBACK:{second["feedback_id"]}:open -->'
    second_end = f'<!-- KB-FEEDBACK:{second["feedback_id"]}:END -->'
    if damage == 'missing-end':
        text = text.replace(first_end, '')
    elif damage == 'duplicate':
        text += '\n' + first_start + text.split(first_start, 1)[1].split(first_end, 1)[0] + first_end + '\n'
    elif damage == 'crossed':
        text = text.replace(first_end, second_end).replace(second_end, first_end, 1)
        # Move the second BEGIN before the first END, then close in crossed order.
        text = text.replace(second_start, '').replace(first_end, second_start + '\n' + first_end, 1)
    else:
        text = text.replace('First evidence', 'Different evidence')
    (repo / path).write_text(text)
    report = kb.queue(repo)
    assert {'path': path, 'id': first['feedback_id']} not in report['open_feedback']
    assert report['workflow_records_needing_review']
    with pytest.raises(ValueError, match='no matching open feedback'):
        kb.feedback(repo, path, 'Reviewed', 'Reviewer', kb.digest(repo / path), first['feedback_id'], 'resolved')


def test_feedback_duplicate_rechecks_target_before_return(repo, monkeypatch):
    path = card(repo)
    kb.feedback(repo, path, 'Evidence', 'Reviewer', kb.digest(repo / path))
    expected = kb.digest(repo / path)
    parse = kb.metadata
    edited = None

    def edit_after_parse(text):
        nonlocal edited
        result = parse(text)
        if edited is None:
            edited = (repo / path).read_bytes() + b'\nExternal note.\n'
            (repo / path).write_bytes(edited)
        return result

    monkeypatch.setattr(kb, 'metadata', edit_after_parse)
    with pytest.raises(ValueError, match='concurrent change'):
        kb.feedback(repo, path, 'Evidence', 'Reviewer', expected)
    assert (repo / path).read_bytes() == edited


def test_queue_never_reads_workflow_markers_from_frontmatter(repo):
    path = card(repo)
    meta, body = kb.metadata((repo / path).read_text())
    meta['aliases'] = ['<!-- KB-FEEDBACK:1111111111111111:open -->']
    (repo / path).write_text(kb.encode(meta, body))
    report = kb.queue(repo)
    assert report['open_feedback'] == report['workflow_records_needing_review'] == []


def test_workflow_parser_dependency_is_lazy_and_reports_clear_error(repo, monkeypatch):
    import builtins
    card(repo)
    load = builtins.__import__

    def without_markdown(name, *args, **kwargs):
        if name == 'markdown_it':
            raise ImportError('fixture: missing parser')
        return load(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_markdown)
    assert kb.inspect(repo)['count'] == 1
    assert kb.search(repo, 'NIW')
    with pytest.raises(ImportError, match='requirements-docs.txt'):
        kb.queue(repo)


def test_writer_lock_and_symlink_runtime_rejected(repo):
    with kb.write_lock(repo):
        with pytest.raises(ValueError, match='another knowledge writer'):
            with kb.write_lock(repo):
                pass
    (repo / '.run/knowledge-base.lock').unlink()
    (repo / '.run').rmdir()
    (repo / '.run').symlink_to(repo / 'docs', target_is_directory=True)
    with pytest.raises(ValueError, match='symlink runtime'):
        ingest(repo)


def test_topic_dependencies_are_rechecked(repo):
    topic(repo)
    assert kb.inspect(repo)['records'][0]['freshness'] == 'current'
    (repo / 'docs/contract.md').write_text('Changed facts')
    assert kb.inspect(repo)['records'][0]['freshness'] == 'stale'
    assert kb.search(repo, '平台')[0]['path'].endswith('project-topic.md')


@pytest.mark.parametrize('intent', ['overview', 'module', 'history', 'compare'])
def test_four_context_intents_are_evidence_packets(repo, intent):
    packet = kb.context_packet(repo, 'NIW', intent)
    assert packet['hits'] and packet['evidence_steps'] and 'answer' not in packet
    assert 'no matching evidence found' in kb.context_packet(repo, 'unmatched_word', intent)['warnings']


def test_domain_and_status_filters(repo):
    path = topic(repo)
    meta, body = kb.metadata((repo / path).read_text())
    meta['domains'] = ['developer']
    (repo / path).write_text(kb.encode(meta, body))
    assert kb.search(repo, '平台', domain='developer')
    assert kb.search(repo, '平台', domain='business') == []
    assert kb.search(repo, '平台', status='draft')
    assert kb.search(repo, '平台', status='historical') == []


def test_catalog_and_coverage_use_original_owner_map(repo):
    r = json.loads((repo / 'docs/repo_map.json').read_text())
    r['documentation']['documents'] += [
        {'path': 'AGENTS.md', 'title': 'Rules', 'role': 'policy', 'status': 'active', 'modules': ['docs']},
        {'path': 'docs/wiki/catalog.md', 'title': 'Catalog', 'role': 'index', 'status': 'active', 'modules': ['docs']}]
    for d in r['documentation']['documents']:
        d['knowledge_domains'], d['modules'] = ['developer'], ['docs']
    r['modules'] = [{'id': 'docs', 'name': 'Documentation', 'path': 'docs/contract.md'}]
    (repo / 'docs/repo_map.json').write_text(json.dumps(r))
    (repo / 'docs/wiki').mkdir()
    (repo / 'docs/wiki/catalog.md').write_text('placeholder')
    (repo / 'docs/unlinked.png').write_bytes(b'fixture binary')
    report = kb.catalog(repo, write=True)
    assert report['errors'] == [] and report['counts']['modules'] == 1
    assert any(a['path'] == 'docs/unlinked.png' and not a['incoming'] for a in report['attachments'])
    assert kb.catalog(repo)['catalog_current']
    (repo / 'docs/extra.md').write_text('# Extra')
    assert kb.inventory(repo)['errors']


def test_registered_root_config_and_assets_can_be_hashed(repo):
    (repo / 'config.py').write_text('fixture = True')
    (repo / 'shared').mkdir()
    (repo / 'shared/navigation.json').write_text('{}')
    r = json.loads((repo / 'docs/repo_map.json').read_text())
    r['modules'] = [{'entry_files': ['config.py', 'shared/navigation.json']}]
    (repo / 'docs/repo_map.json').write_text(json.dumps(r))
    assert kb.safe_file(repo, 'config.py') and kb.safe_file(repo, 'shared/navigation.json')
    (repo / 'shared/unregistered.json').write_text('{}')
    with pytest.raises(ValueError):
        kb.safe_file(repo, 'shared/unregistered.json')
    (repo / 'docs/unlinked.png').write_bytes(b'image fixture')
    assert kb.safe_file(repo, 'docs/unlinked.png')


@pytest.mark.parametrize('uri', ['https://example.invalid/#access_token=DEMO', 'https://example.invalid/#/callback?token=DEMO', 'private:opaque?token=DEMO', 'discussion:opaque?version=1'])
def test_all_source_entrypoints_reject_credential_uri_fragments(repo, uri):
    with pytest.raises(ValueError):
        ingest(repo, uri=uri)
    with pytest.raises(ValueError):
        kb.draft(repo, 'source', 'blocked', 'Unsafe URI', 'business', uri=uri)
    path = kb.draft(repo, 'source', 'manual', 'Manual edit', 'business')
    meta, body = kb.metadata((repo / path).read_text())
    meta['source_uri'] = uri
    (repo / path).write_text(kb.encode(meta, body))
    assert kb.inspect(repo)['invalid'] == 1


def test_inventory_does_not_read_registered_symlink_to_secret(repo):
    (repo / '.env').write_text('[LEAK](unlinked.png)')
    (repo / 'docs/leak.md').symlink_to(repo / '.env')
    (repo / 'docs/unlinked.png').write_bytes(b'fixture')
    r = json.loads((repo / 'docs/repo_map.json').read_text())
    r['documentation']['documents'].append({'path': 'docs/leak.md', 'title': 'Unsafe', 'role': 'topic', 'status': 'active', 'modules': [], 'knowledge_domains': ['developer']})
    (repo / 'docs/repo_map.json').write_text(json.dumps(r))
    report = kb.inventory(repo)
    assert any(e.get('unsafe_document') == 'docs/leak.md' for e in report['errors'])
    assert all('docs/leak.md' not in a['incoming'] for a in report['attachments'])


def test_queue_requires_claim_for_source_usage(repo):
    source, target = ingest(repo)['path'], topic(repo)
    meta, body = kb.metadata((repo / target).read_text())
    meta['dependencies'].append(fingerprint(repo, source))
    (repo / target).write_text(kb.encode(meta, body))
    assert source in kb.queue(repo)['sources_without_claims']
    card(repo, 'source-consumer', [fingerprint(repo, source)])
    assert source not in kb.queue(repo)['sources_without_claims']


def test_exact_identity_duplicate_precedes_other_content_candidate(repo):
    ingest(repo, 'a-source', 'https://example.invalid/a')
    z = ingest(repo, 'z-source', 'https://example.invalid/z', distinct=True)
    repeated = ingest(repo, 'z-repeat', 'https://example.invalid/z')
    assert repeated['status'] == 'duplicate' and repeated['path'] == z['path']


@pytest.mark.parametrize('query,domain,expected', [
    ('服务谁', 'business', 'business-purpose'), ('投研流程', 'business', 'business-process'),
    ('RiskScale Mandate', 'business', 'business-allocation'), ('产品研究', 'business', 'business-products'),
    ('市场状态', 'business', 'business-regimes'), ('金融口径', 'business', 'business-evidence'),
    ('核算', 'business', 'business-investment'), ('业务决定', 'business', 'business-decisions'),
    ('系统分层', 'developer', 'developer-architecture'), ('计算', 'developer', 'developer-data-compute'),
    ('前端', 'developer', 'developer-modules'), ('架构决定', 'developer', 'developer-decisions'),
    ('交付', 'developer', 'developer-delivery'), ('部署', 'developer', 'developer-operations'),
    ('竞品', None, 'research-library'),
])
def test_real_project_high_level_questions(query, domain, expected):
    root = Path(__file__).resolve().parents[2]
    hits = kb.search(root, query, domain=domain)
    assert hits and hits[0]['path'] == f'docs/wiki/topics/{expected}.md'


def test_nonweb_identity_preserves_query_and_rejects_authority():
    assert kb.canonical_source('repo:docs/contract.md?v=1') != kb.canonical_source('repo:docs/contract.md?v=2')
    for uri in ('repo://user:password@host/doc', 'doi://host/10.1234/demo', 'private://host/opaque'):
        with pytest.raises(ValueError, match='authority'):
            kb.canonical_source(uri)


def test_encoded_query_and_fragment_credentials_are_rejected(repo):
    for uri in ('https://example.invalid/?%74oken=DEMO', 'https://example.invalid/#%61ccess_token=DEMO'):
        with pytest.raises(ValueError):
            ingest(repo, uri=uri)
        with pytest.raises(ValueError):
            kb.draft(repo, 'source', 'blocked', 'Blocked', 'business', uri=uri)
    assert not kb.note_paths(repo)


def test_hash_and_summary_cli(repo, capsys):
    path = card(repo)
    assert kb.main(['--root', str(repo), 'hash', path]) == 0
    assert json.loads(capsys.readouterr().out)['sha256'] == kb.digest(repo / path)
    assert kb.main(['--root', str(repo), 'check', '--summary']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['count'] == 1 and report['records'][0]['result'] == 'supported'
    assert 'metadata' not in report['records'][0]


def test_aliases_are_searchable_and_validated(repo):
    path = topic(repo)
    meta, body = kb.metadata((repo / path).read_text())
    meta['aliases'] = ['unique alias phrase']
    (repo / path).write_text(kb.encode(meta, body))
    assert kb.search(repo, 'unique alias')[0]['path'] == path
    meta['aliases'] = ['']
    assert any('aliases' in error for error in kb.validate(meta, body, path))


def test_main_cli_lifecycle(repo, capsys):
    """Exercise public argument routing and JSON results, including explicit review boundaries."""
    def run(*args, expected=0):
        assert kb.main(['--root', str(repo), *args]) == expected
        captured = capsys.readouterr()
        return json.loads(captured.out) if captured.out else captured.err

    source = run('intake', 'cli-source', '--title', 'CLI source', '--domain', 'business', '--source-kind', 'repo_record',
                 '--uri', 'repo:docs/contract.md', '--version', '1', '--summary', 'Bounded CLI evidence.', '--rights', 'Fixture owned')['path']
    assert run('intake', 'cli-source', '--title', 'CLI source', '--domain', 'business', '--source-kind', 'repo_record',
               '--uri', 'repo:docs/contract.md', '--version', '1', '--summary', 'Bounded CLI evidence.', '--rights', 'Fixture owned')['status'] == 'duplicate'
    claim = run('draft', 'claim', 'cli-claim', '--title', 'NIW CLI claim', '--domain', 'business')['created']
    reviewed_source(repo, source)
    meta, _ = kb.metadata((repo / claim).read_text())
    meta.update(scope='Fixture CLI only', reviewed_by='Fixture', reviewed_at='2026-10-04', result='supported',
                review_state='partial', dependencies=[fingerprint(repo, source)])
    body = '# NIW CLI claim\n' + '\n'.join(section + '\nScoped fixture evidence.' for section in ('## 主张', '## 证据', '## 四轴', '## 限制', '## 复核条件'))
    (repo / claim).write_text(kb.encode(meta, body))
    target = topic(repo)
    target_hash = run('hash', target)['sha256']
    assert run('integrate', claim, '--topic', target, '--summary', 'Scoped CLI result', '--reviewer', 'Fixture',
               '--reason', 'Fixture only', '--expected-sha256', target_hash)['status'] == 'integrated'
    assert run('queue')['claims_without_topic_integration'] == []
    assert run('search', 'NIW', '--domain', 'business')[0]['needs_review'] is False
    assert run('context', 'NIW', '--intent', 'module')['hits']
    event = run('feedback', source, '--message', 'New fixture counterevidence', '--by', 'Fixture', '--expected-sha256', run('hash', source)['sha256'])
    assert len(run('queue')['open_feedback']) == 1
    assert run('resolve-feedback', source, '--feedback-id', event['feedback_id'], '--decision', 'deferred', '--message', 'Needs independent evidence',
               '--by', 'Fixture', '--expected-sha256', run('hash', source)['sha256'])['status'] == 'deferred'
    assert run('queue')['open_feedback'] == []
    assert run('check', '--summary', expected=1)['stale'] >= 3
    assert run('fingerprint', 'README.md')[0].startswith('README.md::sha256:')
    assert 'knowledge_base:' in run('fingerprint', source, expected=2)
    compact = run('catalog', '--write')
    assert set(compact) == {'counts', 'errors', 'catalog_current', 'path'}
    # The fixture's new drafts intentionally remain outside its original document catalog.
    assert run('coverage', expected=1)['errors']


def test_catalog_write_checks_safety_before_digest(repo, monkeypatch):
    (repo / 'docs/wiki').mkdir()
    (repo / '.env').write_text('FAKE_SECRET_NOT_REAL')
    (repo / 'docs/wiki/catalog.md').symlink_to(repo / '.env')
    def forbidden_digest(path):
        pytest.fail('catalog symlink must be rejected before hashing or reading its target')
    monkeypatch.setattr(kb, 'digest', forbidden_digest)
    with pytest.raises(ValueError, match='symlink'):
        kb.catalog(repo, write=True)


@pytest.mark.parametrize('uri', [
    'https://example.invalid/?client_secret=DEMO', 'https://example.invalid/?refresh_token=DEMO',
    'https://example.invalid/#id_token=DEMO', 'https://example.invalid/#/access_token=DEMO',
    'https://example.invalid/#/route?token=DEMO', 'https://example.invalid/#/%61ccess_token=DEMO',
])
def test_standard_oauth_and_hash_route_credentials_rejected(repo, uri):
    with pytest.raises(ValueError):
        ingest(repo, uri=uri)
    with pytest.raises(ValueError):
        kb.draft(repo, 'source', 'blocked', 'Blocked', 'business', uri=uri)
    assert not kb.note_paths(repo)
