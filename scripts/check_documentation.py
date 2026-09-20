#!/usr/bin/env python3
"""Check the documentation candidate; never rewrite docs or execute prose commands."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import posixpath
import re
import subprocess
import sys
import unicodedata
from functools import cached_property
from pathlib import Path
from types import ModuleType
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
CATALOG = 'docs/repo_map.json'
HERMES = 'skills/ai-hermes-self-evolve/scripts/evolve_ai_routing.py'
INDEX = 'docs/README.md'
INDEX_START = '<!-- DOCUMENT-INDEX:BEGIN -->'
INDEX_END = '<!-- DOCUMENT-INDEX:END -->'
ROLES = {'overview', 'index', 'topic', 'contract', 'policy', 'research', 'evidence', 'draft'}
STATES = {'active', 'historical', 'draft'}
PLAN_STATES = {'planned', 'in_progress', 'implemented_unverified', 'verified', 'deferred', 'cancelled'}
IGNORED = ('.tmp*/**', '.codegraph/**', '**/node_modules/**', 'frontend/dist/**', 'data/**', '**/__pycache__/**', '*.pyc')


def git(root: Path, *args: str) -> bytes:
    result = subprocess.run(['git', '-C', str(root), *args], capture_output=True)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors='replace').strip() or 'git failed')
    return result.stdout


def safe_path(path: str) -> str:
    if not isinstance(path, str) or '\\' in path or path.startswith('/'):
        raise ValueError(f'expected a repository-relative path: {path}')
    path = posixpath.normpath(path)
    if path in {'.', '..'} or path.startswith('../'):
        raise ValueError(f'path escapes the repository: {path}')
    return path


class Candidate:
    """Read exclusively from the worktree, index, or one immutable commit."""
    def __init__(self, root: Path, mode: str = 'worktree', ref: str = 'HEAD'):
        self.root, self.mode = root.resolve(), mode
        self.ref = git(root, 'rev-parse', '--verify', f'{ref}^{{commit}}').decode().strip() if mode == 'commit' else ref
        self.symlinks: set[str] = set()
        if mode == 'commit':
            entries = git(root, 'ls-tree', '-r', '-z', self.ref).split(b'\0')
        else:
            entries = git(root, 'ls-files', '--stage', '-z').split(b'\0')
        self.paths: set[str] = set()
        for entry in filter(None, entries):
            metadata, path = entry.split(b'\t', 1)
            self.paths.add(path.decode())
            if metadata.startswith(b'120000 '):
                self.symlinks.add(path.decode())
        if mode == 'worktree':
            raw = git(root, 'ls-files', '--others', '--exclude-standard', '-z')
            self.paths.update(p.decode() for p in raw.split(b'\0') if p)
            self.paths = {p for p in self.paths if (self.root / p).is_file()}
        self.cache: dict[str, bytes] = {}

    def exists(self, path: str) -> bool:
        return path in self.paths or any(p.startswith(path.rstrip('/') + '/') for p in self.paths)

    def read(self, path: str) -> bytes:
        path = safe_path(path)
        if path not in self.paths:
            raise ValueError(f'missing candidate file: {path}')
        if path not in self.cache:
            if self.mode == 'worktree':
                p = self.root / path
                if not p.resolve().is_relative_to(self.root):
                    raise ValueError(f'symlink escapes the repository: {path}')
                self.cache[path] = p.read_bytes()
            else:
                if path in self.symlinks:
                    raise ValueError(f'candidate symlink is not a regular document/input: {path}')
                obj = f':{path}' if self.mode == 'staged' else f'{self.ref}:{path}'
                self.cache[path] = git(self.root, 'show', obj)
        return self.cache[path]

    def text(self, path: str) -> str:
        return self.read(path).decode('utf-8')

    @cached_property
    def hermes(self) -> ModuleType:
        """Load routing logic from the same candidate as its documentation."""
        source = self.read(HERMES)
        filename = str(self.root / HERMES)
        name = '_documentation_hermes_' + hashlib.sha256(filename.encode() + b'\0' + source).hexdigest()
        module = ModuleType(name)
        module.__file__ = filename
        # Dataclasses resolve annotations through the module registry during load.
        sys.modules[name] = module
        try:
            exec(compile(source, filename, 'exec'), module.__dict__)
        except Exception as exc:
            sys.modules.pop(name, None)
            raise ValueError(f'cannot load candidate routing helper {HERMES}: {exc}') from exc
        return module


def managed(path: str) -> bool:
    return path.endswith('.md') and ('/' not in path or path.startswith('docs/') or path == 'deploy/README.md')


def document_index(documents: list[dict]) -> str:
    """Only this generated navigation block mirrors the JSON document catalog."""
    rows = ['| 主题 | 入口／文档 | 用途与读取时机 |', '| --- | --- | --- |']
    for d in documents:
        label = d['title'].replace('|', '\\|')
        relative = posixpath.relpath(d['path'], 'docs')
        detail = d['read_when'].replace('|', '\\|')
        rows.append(f"| {d['topic']} | [{label}]({relative}) | {detail} |")
    return INDEX_START + '\n' + '\n'.join(rows) + '\n' + INDEX_END


def markdown(text: str) -> tuple[set[str], list[str]]:
    tokens = MarkdownIt('commonmark').parse(text)
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    links: list[str] = []
    for i, token in enumerate(tokens):
        if token.type == 'heading_open':
            inline = tokens[i + 1]
            title = ''.join(t.content for t in inline.children or [] if t.type in {'text', 'code_inline'})
            title = title.lower()
            slug = ''.join(c for c in title if c in '_- ' or unicodedata.category(c)[0] in 'LN').replace(' ', '-')
            n = counts.get(slug, 0)
            counts[slug] = n + 1
            anchors.add(f'{slug}-{n}' if n else slug)
        for t in token.children or []:
            if t.type == 'link_open':
                links.append(t.attrGet('href'))
            elif t.type == 'image':
                links.append(t.attrGet('src'))
        if token.type in {'html_block', 'inline'}:
            anchors.update(re.findall(r'\b(?:id|name)=["\']([^"\']+)["\']', token.content))
    return anchors, links


def local_target(source: str, href: str) -> tuple[str, str] | None:
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc:
        return None
    raw = unquote(parsed.path)
    if raw.startswith('/'):
        raise ValueError(f'nonportable absolute link: {href}')
    target = safe_path(posixpath.join(posixpath.dirname(source), raw)) if raw else source
    return target, unquote(parsed.fragment)


def check_plans(path: str, text: str) -> list[str]:
    """Validate the explicit maintenance table, without guessing states from prose."""
    errors, ids = [], set()
    inside = False
    code_lines = set()
    for token in MarkdownIt('commonmark').parse(text):
        if token.type in {'fence', 'code_block'} and token.map:
            code_lines.update(range(*token.map))
    for number, line in enumerate(text.splitlines()):
        if number in code_lines:
            inside = False
            continue
        if line.strip() == '| ID | 状态 | 工作项 | 完成判据 | 证据/剩余事项 |':
            inside = True
            continue
        if not inside:
            continue
        if not line.startswith('|'):
            inside = False
            continue
        cells = [s.strip() for s in line.strip('|').split('|')]
        if all(re.fullmatch(r':?-+:?', c) for c in cells):
            continue
        if len(cells) != 5:
            errors.append(f'{path}: plan row requires five cells')
            continue
        ident, state, work, acceptance, evidence = cells
        if not re.fullmatch(r'[A-Z][A-Z0-9]*-\d+', ident) or ident in ids:
            errors.append(f'{path}: invalid or duplicate plan ID {ident}')
        ids.add(ident)
        if state not in PLAN_STATES or not work or not acceptance or not evidence:
            errors.append(f'{path}: plan {ident} needs legal state, acceptance and evidence/remaining work')
        if state == 'verified' and not markdown(evidence)[1]:
            errors.append(f'{path}: verified plan {ident} needs a link to scoped evidence')
    return errors


def catalog(candidate: Candidate) -> tuple[dict, list[dict], list[str]]:
    mapping = json.loads(candidate.text(CATALOG))
    config = mapping.get('documentation', {})
    documents = config.get('documents', [])
    if not isinstance(documents, list) or not documents:
        raise ValueError('repo_map.documentation.documents must be a nonempty array')
    errors, seen = [], set()
    modules = {m['id'] for m in mapping['modules']}
    for d in documents:
        if not isinstance(d, dict):
            raise ValueError('document catalog row must be an object')
        path = safe_path(d.get('path', ''))
        if path in seen:
            errors.append(f'duplicate documentation owner: {path}')
        seen.add(path)
        if not candidate.exists(path) or path not in candidate.paths:
            errors.append(f'missing catalog document: {path}')
        for key in ('title', 'topic', 'read_when'):
            if not isinstance(d.get(key), str) or not d[key].strip():
                errors.append(f'{path}: missing {key}')
        if d.get('role') not in ROLES or d.get('status') not in STATES:
            errors.append(f'{path}: invalid document role/status')
        if not isinstance(d.get('modules'), list) or not d['modules'] or any(m not in modules for m in d['modules']):
            errors.append(f'{path}: document needs valid module ownership')
        if (d.get('role') == 'draft') != (d.get('status') == 'draft'):
            errors.append(f'{path}: draft role/status disagree')
    expected = {p for p in candidate.paths if managed(p)}
    for path in sorted(expected - seen):
        errors.append(f'uncatalogued document: {path}')
    for path in sorted(seen - expected):
        errors.append(f'document outside the managed Markdown scope: {path}')
    for path in sorted(expected):
        if '/' not in path and path not in {'AGENTS.md', 'README.md'}:
            errors.append(f'root Markdown belongs under docs/: {path}')
    return mapping, documents, errors


def impacts(candidate: Candidate, mapping: dict, documents: list[dict], changed: list[str]) -> list[dict]:
    helpers = candidate.hermes
    causes: dict[str, set[str]] = {}
    for path in changed:
        if path.endswith('.md'):
            causes.setdefault(path, set()).add(path)
            continue
        for m in mapping['modules']:
            refs = [p for _, p in helpers._iter_path_values(m, helpers.MODULE_PATH_FIELDS)]
            if any(path == p.removeprefix('./') or path.startswith(p.removeprefix('./').rstrip('/') + '/') for p in refs):
                for d in documents:
                    if m['id'] in d['modules']:
                        causes.setdefault(d['path'], set()).add(path)
    # Include incoming references for moved/deleted or otherwise changed docs.
    for d in documents:
        if d['path'] not in candidate.paths:
            continue
        for href in markdown(candidate.text(d['path']))[1]:
            try:
                target = local_target(d['path'], href)
            except ValueError:
                continue
            if target and target[0] in changed:
                causes.setdefault(d['path'], set()).add(target[0])
    by_path = {d['path']: d for d in documents}
    return [{'path': p, 'role': by_path[p]['role'], 'caused_by': sorted(c)}
            for p, c in sorted(causes.items()) if p in by_path]


def inspect(candidate: Candidate, changed: list[str], review: dict | None = None,
            require_review: bool = False) -> dict:
    mapping, documents, errors = catalog(candidate)
    parsed: dict[str, tuple[set[str], list[str]]] = {}
    link_count = 0
    for d in documents:
        path = d['path']
        if path not in candidate.paths:
            continue
        text = candidate.text(path)
        parsed[path] = markdown(text)
        errors.extend(check_plans(path, text))
    for path, (_, links) in list(parsed.items()):
        for href in links:
            try:
                local = local_target(path, href)
            except ValueError as exc:
                errors.append(f'{path}: {exc}')
                continue
            if not local:
                continue
            target, anchor = local
            link_count += 1
            if not candidate.exists(target):
                errors.append(f'{path}: broken link {href}')
            elif anchor and target.endswith('.md') and target in candidate.paths:
                if target not in parsed:
                    parsed[target] = markdown(candidate.text(target))
                if anchor not in parsed[target][0]:
                    errors.append(f'{path}: missing anchor {href}')
    if not errors:
        expected = document_index(documents)
        index = candidate.text(INDEX)
        match = re.search(re.escape(INDEX_START) + r'.*?' + re.escape(INDEX_END), index, re.S)
        if not match or match.group() != expected:
            errors.append(f'{INDEX}: index differs from repo_map; regenerate the navigation block')
    affected = impacts(candidate, mapping, documents, changed)
    digest = hashlib.sha256()
    for path in sorted(set(changed) | {d['path'] for d in documents} | {CATALOG, HERMES}):
        digest.update(path.encode() + b'\0')
        digest.update(hashlib.sha256(candidate.read(path)).digest() if path in candidate.paths else b'DELETED')
    fingerprint = digest.hexdigest()
    decisions = {}
    if review is not None:
        if not isinstance(review, dict):
            raise ValueError('review must be an object')
        if review.get('fingerprint') != fingerprint:
            errors.append('review fingerprint does not match the current candidate')
        rows = review.get('documents', [])
        if not isinstance(rows, list):
            raise ValueError('review.documents must be an array')
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('review document row must be an object')
            p = row.get('path')
            if p in decisions or p not in {d['path'] for d in affected}:
                errors.append(f'invalid or duplicate reviewed document: {p}')
            status = row.get('status')
            reason = row.get('reason')
            if status not in {'updated', 'reviewed_no_change', 'needs_review'} or not isinstance(reason, str) or not reason.strip():
                errors.append(f'{p}: review needs a legal status and a concrete reason')
            if status == 'updated' and p not in changed:
                errors.append(f'{p}: marked updated but absent from the changed scope')
            decisions[p] = row
    for d in affected:
        d['review'] = decisions.get(d['path'], {'status': 'needs_review'})
        if require_review and d['review']['status'] == 'needs_review':
            errors.append(f"{d['path']}: documentation impact review is incomplete")
    return {'candidate': candidate.mode, 'fingerprint': fingerprint,
            'documents_checked': len(documents), 'local_links_checked': link_count,
            'changed_files': changed, 'affected_documents': affected,
            'errors': sorted(set(errors)),
            'semantic_review': 'required_by_AGENTS; structural checks do not prove content accuracy',
            'status': 'failed' if errors else 'passed'}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--staged', action='store_true')
    source.add_argument('--base-ref')
    source.add_argument('--changed-file', action='append', default=[])
    parser.add_argument('--head-ref', default='HEAD')
    parser.add_argument('--review-file', type=Path)
    parser.add_argument('--require-review', action='store_true')
    parser.add_argument('--print-index', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.head_ref != 'HEAD' and not args.base_ref:
            raise ValueError('--head-ref requires --base-ref')
        mode = 'staged' if args.staged else 'commit' if args.base_ref else 'worktree'
        candidate = Candidate(args.project_root, mode, args.head_ref)
        helpers = candidate.hermes
        if args.staged:
            entries = helpers._run_git_changed_entries(candidate.root, ['--cached'])
        elif args.base_ref:
            base = git(candidate.root, 'merge-base', args.base_ref, candidate.ref).decode().strip()
            entries = helpers.collect_changed_entries(candidate.root, [], base_ref=base, head_ref=candidate.ref)
        else:
            for path in args.changed_file:
                safe_path(path)
            entries = helpers.collect_changed_entries(candidate.root, args.changed_file)
        changed = sorted({e.path for e in entries if not any(fnmatch.fnmatch(e.path, pat) for pat in IGNORED)})
        if args.print_index:
            _, docs, errors = catalog(candidate)
            if errors:
                raise ValueError('\n'.join(errors))
            print(document_index(docs))
            return 0
        review = json.loads(args.review_file.read_text()) if args.review_file else None
        result = inspect(candidate, changed, review, args.require_review)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {'status': 'failed', 'errors': [str(exc)]}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Documentation: {result['status']}; {result.get('documents_checked', 0)} documents, "
              f"{result.get('local_links_checked', 0)} local links")
        for error in result['errors']:
            print(f'ERROR: {error}')
        for d in result.get('affected_documents', []):
            print(f"REVIEW: {d['path']} ({d['review']['status']}) <- {', '.join(d['caused_by'])}")
    return 1 if result['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
