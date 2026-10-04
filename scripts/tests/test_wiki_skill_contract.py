"""Offline repository-skill contracts and a complete evidence-review loop."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest
from markdown_it import MarkdownIt

from test_documentation import check as documentation
from test_knowledge_base import (
    card, fingerprint, ingest, kb, repo, reviewed_source, topic,
)


SKILL = '.agents/skills/obsidian-wiki/SKILL.md'


def register_fixture_catalog(root):
    """Use the same original owner map as discovery, never a second catalog."""
    mapping = json.loads((root / 'docs/repo_map.json').read_text())
    documents = mapping['documentation']['documents']
    existing = {item['path'] for item in documents}
    for path in ['AGENTS.md', 'docs/wiki/catalog.md', *kb.note_paths(root)]:
        if path not in existing:
            documents.append({'path': path, 'title': path, 'role': 'topic', 'status': 'active'})
    for entry in documents:
        entry.update(knowledge_domains=['business', 'developer'], modules=['documentation'])
    mapping['modules'] = [{'id': 'documentation', 'name': 'Documentation', 'path': 'docs/contract.md',
                           'then_check_files': [SKILL]}]
    (root / 'docs/repo_map.json').write_text(json.dumps(mapping))
    target = root / 'docs/wiki/catalog.md'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch()


def test_exact_repository_skill_is_discoverable_and_fingerprintable(repo):
    skill = repo / SKILL
    skill.parent.mkdir(parents=True)
    skill.write_text('# Project Wiki skill\nFixture instructions only.\n')
    register_fixture_catalog(repo)
    assert kb.safe_file(repo, SKILL) == skill
    with pytest.raises(ValueError):
        kb.safe_file(repo, SKILL, tracked=True)
    subprocess.run(['git', '-C', str(repo), 'add', '--', SKILL], check=True)
    assert kb.safe_file(repo, SKILL, tracked=True) == skill
    report = kb.catalog(repo, write=True)
    assert report['errors'] == []
    assert [entry['path'] for entry in report['supporting_documents']] == [SKILL]
    assert SKILL not in {entry['path'] for entry in report['documents']}
    assert f'../../{SKILL}' in (repo / 'docs/wiki/catalog.md').read_text()
    assert kb.catalog(repo)['catalog_current']


@pytest.mark.parametrize('path', [
    '.agents/skills/other/SKILL.md',
    '.agents/skills/obsidian-wiki/notes.md',
    '.agents/skills/obsidian-wiki/private/context.md',
    '.agents/skills/obsidian-wiki/.env',
    '.agents/skills/obsidian-wiki/SKILL.md/secret.md',
    '.agents/skills/obsidian-wiki/../obsidian-wiki/SKILL.md',
    '.agents/skills/obsidian-wiki/./SKILL.md',
    './.agents/skills/obsidian-wiki/SKILL.md',
    '.agents/skills/obsidian-wiki/skill.md',
    '.codex/config.json', '.obsidian/workspace.json', '.env',
    '.github/workflows/documentation.yml',
])
def test_skill_allowlist_never_opens_other_hidden_paths(repo, path):
    # Even a registered path cannot widen the exact hidden-file exception.
    mapping = json.loads((repo / 'docs/repo_map.json').read_text())
    mapping['modules'] = [{'entry_files': [path]}]
    (repo / 'docs/repo_map.json').write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match='unsafe path'):
        kb.safe_file(repo, path)


@pytest.mark.parametrize('component', ['.agents', '.agents/skills', '.agents/skills/obsidian-wiki', SKILL])
def test_skill_allowlist_rejects_every_symlink_component(repo, component):
    link = repo / component
    link.parent.mkdir(parents=True, exist_ok=True)
    destination = repo / 'docs/contract.md' if component == SKILL else repo / 'docs'
    link.symlink_to(destination, target_is_directory=component != SKILL)
    with pytest.raises(ValueError, match='symlink'):
        kb.safe_file(repo, SKILL)


def test_inventory_does_not_read_neighboring_agent_or_private_files(repo, monkeypatch):
    skill = repo / SKILL
    skill.parent.mkdir(parents=True)
    skill.write_text('# Project skill\n')
    private = repo / '.agents/private/session.md'
    private.parent.mkdir(parents=True)
    private.write_text('FIXTURE PRIVATE BYTES MUST NOT BE READ')
    register_fixture_catalog(repo)
    original_read_text, original_read_bytes = Path.read_text, Path.read_bytes

    def guarded_read_text(path, *args, **kwargs):
        assert path != private, 'inventory read a neighboring private file'
        return original_read_text(path, *args, **kwargs)

    def guarded_read_bytes(path, *args, **kwargs):
        assert path != private, 'inventory read a neighboring private file'
        return original_read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', guarded_read_text)
    monkeypatch.setattr(Path, 'read_bytes', guarded_read_bytes)
    report = kb.inventory(repo)
    assert report['errors'] == []
    assert {entry['path'] for entry in report['supporting_documents']} == {SKILL}


def test_skill_allowlist_preserves_the_evidence_size_limit(repo):
    skill = repo / SKILL
    skill.parent.mkdir(parents=True)
    with skill.open('wb') as handle:
        handle.truncate(32 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match='32 MiB'):
        kb.safe_file(repo, SKILL)


def test_repository_skill_redirect_and_local_references_resolve():
    root = Path(kb.__file__).resolve().parents[1]
    entry = kb.safe_file(root, SKILL)
    canonical = root / 'skills/obsidian-wiki/SKILL.md'
    assert not canonical.is_symlink()
    assert 'name: obsidian-wiki' in canonical.read_text()
    assert 'scripts/knowledge_base.py' not in entry.read_text(), 'discovery entry must stay a thin redirect'
    references = sorted(canonical.parent.glob('references/*.md'))
    assert {path.stem for path in references} == {'query', 'context-pack', 'ingest', 'update', 'lint'}
    documents = [entry, canonical, canonical.with_name('UPSTREAM.md'), *references]
    redirects = []
    for document in documents:
        assert kb.safe_file(root, document.relative_to(root).as_posix()) == document
        for token in MarkdownIt('commonmark').parse(document.read_text()):
            for child in token.children or []:
                if child.type != 'link_open':
                    continue
                target = documentation.local_target(document.relative_to(root).as_posix(), child.attrGet('href'))
                if target is None:
                    continue
                relative, fragment = target
                # Validate before resolve so no symlink component can disappear.
                resolved = kb.safe_file(root, relative).resolve()
                assert resolved.is_relative_to(root) and resolved.is_file(), (document, relative)
                if fragment:
                    assert fragment in documentation.markdown(resolved.read_text())[0]
                if document == entry:
                    redirects.append(resolved)
    assert redirects == [canonical]


def test_documented_commands_use_the_actual_cli_argument_contract(monkeypatch):
    """Parse real examples, then stop before any read or mutation dispatch."""
    class Parsed(Exception):
        def __init__(self, namespace):
            self.namespace = namespace

    parse_args = kb.argparse.ArgumentParser.parse_args

    def parse_and_stop(parser, args=None, namespace=None):
        raise Parsed(parse_args(parser, args, namespace))

    monkeypatch.setattr(kb.argparse.ArgumentParser, 'parse_args', parse_and_stop)
    root = Path(kb.__file__).resolve().parents[1]
    commands, intents = set(), set()
    for document in sorted((root / 'skills/obsidian-wiki/references').glob('*.md')):
        for token in MarkdownIt('commonmark').parse(document.read_text()):
            if token.type != 'fence' or token.info != 'bash':
                continue
            for line in token.content.splitlines():
                argv = shlex.split(line)
                if argv[:2] != ['python3', 'scripts/knowledge_base.py']:
                    continue
                with pytest.raises(Parsed) as captured:
                    kb.main(argv[2:])
                parsed = captured.value.namespace
                commands.add(parsed.command)
                if parsed.command == 'context':
                    intents.add(parsed.intent)
    assert commands == {'search', 'context', 'check', 'intake', 'draft', 'fingerprint',
                        'hash', 'feedback', 'queue', 'resolve-feedback', 'coverage'}
    assert intents == {'overview', 'module', 'history', 'compare'}


def test_documentation_ci_runs_real_repository_checks_and_preserves_reports():
    root = Path(kb.__file__).resolve().parents[1]
    workflow = (root / '.github/workflows/documentation.yml').read_text()
    assert 'ref: ${{ github.event.pull_request.head.sha }}' in workflow
    assert 'scripts/tests/test_wiki_skill_contract.py' in workflow
    assert 'continue-on-error:' not in workflow
    artifacts = workflow.split('uses: actions/upload-artifact@v4', 1)[1]
    for command, report in [
        ('check --summary', 'knowledge-base-check-report.json'),
        ('coverage', 'knowledge-base-coverage-report.json'),
    ]:
        assert f'run: python scripts/knowledge_base.py {command} > {report}' in workflow
        assert report in artifacts


def snapshot(root):
    """Include ignored files, empty directories, Git state, bytes and mtime."""
    return {path.relative_to(root).as_posix():
            (path.is_dir(), path.stat().st_mtime_ns, None if path.is_dir() else path.read_bytes())
            for path in root.rglob('*')}


def test_readonly_commands_leave_a_fresh_checkout_byte_for_byte_unchanged(repo):
    card(repo)
    register_fixture_catalog(repo)
    kb.catalog(repo, write=True)
    scripts = repo / 'scripts'
    scripts.mkdir()
    # Fresh subprocesses must not hide local __pycache__ writes behind a warm import.
    for filename in ('knowledge_base.py', 'check_documentation.py'):
        shutil.copyfile(Path(kb.__file__).with_name(filename), scripts / filename)
    runtime = repo / '.run'
    if runtime.exists():
        shutil.rmtree(runtime)
    before = snapshot(repo)
    for args in [
        ['search', 'NIW'], ['context', 'NIW', '--intent', 'overview'],
        ['context', 'NIW', '--intent', 'module'], ['context', 'NIW', '--intent', 'history'],
        ['context', 'NIW', '--intent', 'compare'], ['context', 'no-match', '--intent', 'overview'],
        ['check', '--summary'], ['queue'], ['hash', 'docs/contract.md'],
        ['fingerprint', 'docs/contract.md'], ['catalog'], ['coverage'],
    ]:
        result = subprocess.run([sys.executable, str(scripts / 'knowledge_base.py'), '--root', str(repo), *args],
                                capture_output=True, text=True, check=True)
        assert json.loads(result.stdout) is not None
        assert snapshot(repo) == before, f'read-only command changed the checkout: {args}'
    assert not (repo / '.run').exists()
    assert not (scripts / '__pycache__').exists()


def test_source_to_query_invalidation_feedback_and_reviewed_reintegration(repo, capsys):
    """Human review is explicit fixture editing, never an automatic hash refresh."""
    def run(*args, expected=0):
        assert kb.main(['--root', str(repo), *args]) == expected
        result = capsys.readouterr()
        return json.loads(result.out) if result.out else result.err

    def integrate(claim_path, target_path, summary, expected=0):
        return run('integrate', claim_path, '--topic', target_path, '--summary', summary,
                   '--reviewer', 'Fixture reviewer', '--reason', 'Review of fixture evidence only',
                   '--expected-sha256', run('hash', target_path)['sha256'], expected=expected)

    source = ingest(repo, summary='NIW fixture evidence version one.')['path']
    assert ingest(repo, slug='duplicate', summary='NIW fixture evidence version one.')['status'] == 'duplicate'
    claim = card(repo, 'review-loop', [fingerprint(repo, source)])
    target = topic(repo, 'review-loop-topic')
    original_authorities = {p: (repo / p).read_bytes() for p in ('AGENTS.md', 'README.md', 'docs/contract.md')}
    assert 'needs review' in integrate(claim, target, 'NIW version-one fixture summary.', expected=2)
    reviewed_source(repo, source)
    claim_meta, claim_body = kb.metadata((repo / claim).read_text())
    claim_meta['dependencies'] = [fingerprint(repo, source)]
    (repo / claim).write_text(kb.encode(claim_meta, claim_body))
    assert integrate(claim, target, 'NIW version-one fixture summary.')['status'] == 'integrated'
    assert run('queue')['sources_without_claims'] == []
    assert run('queue')['claims_without_topic_integration'] == []
    for intent in ('overview', 'module', 'history', 'compare'):
        packet = run('context', 'NIW', '--intent', intent)
        assert any(hit['path'] == target and not hit['needs_review'] for hit in packet['hits'])
    initial_topic = (repo / target).read_bytes()

    # A changed authoritative source invalidates source -> claim -> topic without rewriting them.
    (repo / 'docs/contract.md').write_text('# NIW\nRevised fixture: monthly qualification remains unverified.\n')
    revised_authority = (repo / 'docs/contract.md').read_bytes()
    changed = {item['path']: item for item in run('check', '--summary', expected=1)['records']}
    assert all(changed[p]['freshness'] == 'stale' for p in (source, claim, target))
    assert (repo / target).read_bytes() == initial_topic
    assert all(hit['needs_review'] for hit in run('search', 'NIW') if hit['path'] in (source, claim, target))
    assert 'needs review' in integrate(claim, target, 'Premature reintegration', expected=2)

    event = run('feedback', source, '--message', 'Source changed: narrow the NIW conclusion.', '--by', 'Fixture researcher',
                '--expected-sha256', run('hash', source)['sha256'])
    assert run('queue')['open_feedback'] == [{'path': source, 'id': event['feedback_id']}]
    resolution = run('resolve-feedback', source, '--feedback-id', event['feedback_id'], '--decision', 'resolved',
                     '--message', 'Reviewed changed source; monthly qualification is still unverified.', '--by', 'Fixture reviewer',
                     '--expected-sha256', run('hash', source)['sha256'])
    assert resolution['status'] == 'resolved'
    assert run('queue')['open_feedback'] == []
    assert run('check', '--summary', expected=1)['stale'] == 3

    # Explicit review narrows the prose and the approved/static/tested/deployed axes.
    reviewed_source(repo, source)
    source_meta, source_body = kb.metadata((repo / source).read_text())
    source_body += '\n复核结论：新版本只支持静态范围；月频资格未验证。\n'
    (repo / source).write_text(kb.encode(source_meta, source_body))
    claim_meta['dependencies'] = [fingerprint(repo, source)]
    claim_body = claim_body.replace('仅静态证据，尚未运行。', '批准目标：限fixture；当前实现：静态；已验证范围：本地夹具；实际部署：未核。')
    (repo / claim).write_text(kb.encode(claim_meta, claim_body))
    assert 'needs review' in integrate(claim, target, 'NIW revised bounded summary.', expected=2)

    # Explicitly retire the prior receipt after reviewing the topic. The tool refuses to overwrite it.
    target_meta, target_body = kb.metadata((repo / target).read_text())
    target_meta['dependencies'] = [fingerprint(repo, 'docs/contract.md'), fingerprint(repo, claim)]
    (repo / target).write_text(kb.encode(target_meta, target_body))
    before_reintegration = (repo / target).read_bytes()
    assert 'existing integration differs' in integrate(claim, target, 'NIW revised bounded summary.', expected=2)
    assert (repo / target).read_bytes() == before_reintegration
    records, warnings = kb.workflow_records(target_body)
    assert not warnings
    prior = next(record for record in records if record['kind'] == 'INTEGRATION')
    target_body = target_body[:prior['start']] + target_body[prior['end']:]
    target_body += ('\n历史复核：旧摘要“NIW version-one fixture summary.”只适用于旧来源版本，已撤回；'
                    '原主题sha256=' + hashlib.sha256(initial_topic).hexdigest() + '。\n')
    (repo / target).write_text(kb.encode(target_meta, target_body))
    reintegrated = integrate(claim, target, 'NIW revised bounded summary; no monthly qualification or deployment proof.')
    assert reintegrated['status'] == 'integrated'
    assert run('check', '--summary')['stale'] == 0
    assert all(not hit['needs_review'] for hit in run('search', 'NIW') if hit['path'] in (source, claim, target))
    queue = run('queue')
    assert not any(queue.values())
    assert (repo / 'docs/contract.md').read_bytes() == revised_authority
    assert all((repo / p).read_bytes() == content for p, content in original_authorities.items() if p != 'docs/contract.md')
    assert '月频资格未验证' in (repo / source).read_text()
    assert '实际部署：未核' in (repo / claim).read_text()
    assert '只适用于旧来源版本，已撤回' in (repo / target).read_text()
