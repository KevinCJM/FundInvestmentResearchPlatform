#!/usr/bin/env python3
"""Offline discovery and bounded draft workflows for the project Obsidian vault.

Never fetch URLs, execute notes, rewrite authorities, or automatically approve evidence.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date
import fcntl
import fnmatch
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
WIKI = 'docs/wiki'
KINDS = {'claim', 'source', 'topic'}
DRAFT_KINDS = {'claim', 'source'}
STATES = {'pending', 'partial', 'reviewed', 'stale'}
RESULTS = {'supported', 'unverified', 'conflict', 'superseded', 'rejected'}
DOMAINS = {'business', 'developer'}
SOURCE_KINDS = {'repo_record', 'official_source', 'paper', 'vendor_claim', 'user_summary', 'ai_summary'}
FINGERPRINT = re.compile(r'(.+)::sha256:([0-9a-f]{64})\Z')
ID = re.compile(r'[a-z0-9]+(?:-[a-z0-9]+)*\Z')
PLACEHOLDER = re.compile(r'\b(?:TODO|TBD)\b|待填写|待补充', re.I)


def git(root: Path, *args: str) -> str:
    result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
    if result.returncode:
        raise ValueError(result.stderr.strip() or 'git failed')
    return result.stdout.strip()


def declared_paths(root: Path) -> set[str]:
    """Exceptional root/shared/config evidence comes from the existing Hermes owner."""
    path = root / 'docs/repo_map.json'
    if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
        return set()
    result = set()
    mapping = json.loads(path.read_text(encoding='utf-8'))
    for module in mapping.get('modules', []):
        for key in ('entry_files', 'first_read_files', 'then_check_files', 'read_before_edit', 'related_tests', 'related_configs'):
            result.update(p.removeprefix('./') for p in module.get(key, []) if isinstance(p, str))
        if isinstance(module.get('path'), str):
            result.add(module['path'].removeprefix('./'))
        result.update(p.removeprefix('./') for p in module.get('grounding', {}).get('evidence', []) if isinstance(p, str))
    return result


def safe_file(root: Path, relative: str, *, tracked: bool = False) -> Path:
    """Reject absolute/traversing paths, secrets/runtime folders and every symlink component."""
    p = Path(relative)
    if (not relative or not p.parts or p.is_absolute() or '\\' in relative or '..' in p.parts
            or (any(x.startswith('.') for x in p.parts) and relative != '.gitmodules')):
        raise ValueError(f'unsafe path: {relative}')
    if any(x.casefold() in {'secrets', 'credentials', 'tokens'} for x in p.parts):
        raise ValueError(f'sensitive path rejected: {relative}')
    if p.as_posix() != relative or (p.parts[0] not in {'docs', 'backend', 'frontend', 'scripts', 'deploy', 'skills', 'images'}
                                   and relative not in {'README.md', 'AGENTS.md'} and relative not in declared_paths(root)):
        raise ValueError(f'path outside knowledge/code scope: {relative}')
    if (p.suffix not in {'.md', '.py', '.ts', '.tsx', '.js', '.mjs', '.json', '.yml', '.yaml', '.toml', '.txt',
                         '.png', '.jpg', '.jpeg', '.webp', '.svg', '.html', '.pdf'}
            and relative not in declared_paths(root)):
        raise ValueError(f'unsupported dependency type: {relative}')
    if any(x in {'node_modules', 'dist', '__pycache__', 'data', 'output'} for x in p.parts):
        if not (p.parts[:2] == ('docs', 'data') and not any(x in {'node_modules', 'dist', '__pycache__', 'output'} for x in p.parts)):
            raise ValueError(f'runtime path rejected: {relative}')
    target = root
    for component in p.parts:
        target /= component
        if target.is_symlink():
            raise ValueError(f'symlink rejected: {relative}')
    if not target.resolve().is_relative_to(root.resolve()) or not target.is_file():
        raise ValueError(f'missing or out-of-vault file: {relative}')
    if tracked:
        git(root, '--literal-pathspecs', 'ls-files', '--error-unmatch', '--', relative)
    if target.stat().st_size > 32 * 1024 * 1024:
        raise ValueError(f'evidence file exceeds 32 MiB limit: {relative}')
    return target


def metadata(text: str) -> tuple[dict, str]:
    """Read only the documented JSON-valued YAML subset."""
    lines = text.splitlines()
    if not lines or lines[0] != '---':
        raise ValueError('missing frontmatter')
    try:
        end = lines.index('---', 1)
    except ValueError as exc:
        raise ValueError('unterminated frontmatter') from exc
    result = {}
    for line in lines[1:end]:
        key, sep, value = line.partition(':')
        if not sep or not re.fullmatch(r'[a-z_][a-z0-9_]*', key) or key in result:
            raise ValueError(f'invalid or duplicate frontmatter key: {key}')
        try:
            result[key] = json.loads(value.strip())
        except json.JSONDecodeError as exc:
            raise ValueError(f'{key}: expected a JSON value in YAML frontmatter') from exc
    return result, '\n'.join(lines[end + 1:])


def encode(meta: dict, body: str) -> str:
    return '---\n' + '\n'.join(f'{key}: {json.dumps(value, ensure_ascii=False)}' for key, value in meta.items()) + '\n---\n\n' + body.strip() + '\n'


def note_paths(root: Path) -> list[str]:
    result = []
    for kind in sorted(KINDS):
        directory = root / WIKI / (kind + 's')
        if directory.is_symlink():
            raise ValueError(f'symlink rejected: {directory.relative_to(root)}')
        if directory.exists():
            result.extend(p.relative_to(root).as_posix() for p in directory.glob('*.md'))
    return sorted(result)


def canonical_source(uri: str) -> str:
    uri = uri.strip()
    if not uri or any(ord(c) < 32 for c in uri):
        raise ValueError('source URI must be nonempty and contain no control characters')
    parsed = urlsplit(uri)
    scheme = parsed.scheme.lower()
    sensitive = {'token', 'access_token', 'auth', 'authorization', 'signature', 'sig', 'api_key', 'apikey', 'key', 'password', 'credential', 'client_secret', 'refresh_token', 'id_token', 'session_token', 'oauth_token', 'oauth_token_secret', 'access_key', 'secret_key', 'api_secret'}
    query = parse_qsl(parsed.query, keep_blank_values=True)
    credential_keys = [k.lower() for k, _ in query]
    credential_keys += [k.lower() for k in re.findall(r'(?:^|[&;?/])([^=&#?;/]+)=', unquote(parsed.fragment))]
    if any(k in sensitive or k.startswith(('x-amz-', 'x-goog-')) for k in credential_keys):
        raise ValueError('signed or credential-bearing source URI rejected')
    if scheme in {'http', 'https'}:
        if parsed.username or parsed.password or not parsed.hostname:
            raise ValueError('source URI must not contain credentials')
        return urlunsplit((scheme, parsed.netloc.lower(), parsed.path or '/', urlencode(sorted(query)), parsed.fragment))
    if scheme in {'repo', 'doi', 'discussion', 'private'} and parsed.path:
        if parsed.netloc:
            raise ValueError('non-web source references must not contain an authority')
        if scheme in {'discussion', 'private'}:
            if '/' in parsed.path or '\\' in parsed.path:
                raise ValueError('private/discussion sources require opaque reference IDs, not file paths')
            if parsed.query or parsed.fragment:
                raise ValueError('private/discussion sources must be opaque IDs without query or fragment')
        return urlunsplit((scheme, '', parsed.path, urlencode(sorted(query)), parsed.fragment))
    raise ValueError('use an http(s), repo, doi, discussion, or private source reference')


def validate(meta: dict, body: str, path: str) -> list[str]:
    errors = []
    for key in ('id', 'type', 'title', 'scope', 'reviewed_at', 'reviewed_by', 'source_revision'):
        if not isinstance(meta.get(key), str):
            errors.append(f'{key} must be a string')
    if not ID.fullmatch(str(meta.get('id', ''))) or Path(path).stem != meta.get('id'):
        errors.append('id must be a lowercase slug matching the filename')
    kind = meta.get('type')
    if not isinstance(kind, str) or kind not in KINDS or Path(path).parent.name != str(kind) + 's':
        errors.append('type must match its claims/sources/topics directory')
    if not meta.get('title') or '\n' in str(meta.get('title', '')) or '\r' in str(meta.get('title', '')):
        errors.append('title must be one nonempty line')
    if not isinstance(meta.get('review_state'), str) or meta['review_state'] not in STATES:
        errors.append('invalid review_state')
    if not isinstance(meta.get('result'), str) or meta['result'] not in RESULTS:
        errors.append('invalid result')
    domains = meta.get('domains')
    if not isinstance(domains, list) or not domains or any(not isinstance(d, str) or d not in DOMAINS for d in domains) or len(set(domains)) != len(domains):
        errors.append('domains must be a nonempty unique business/developer list')
    aliases = meta.get('aliases', [])
    if not isinstance(aliases, list) or any(not isinstance(a, str) or not a.strip() for a in aliases):
        errors.append('aliases must contain nonempty strings')
    deps = meta.get('dependencies')
    if not isinstance(deps, list) or any(not isinstance(d, str) or not FINGERPRINT.fullmatch(d) for d in deps):
        errors.append('dependencies must contain path::sha256:hash strings')
    elif len({FINGERPRINT.fullmatch(d).group(1) for d in deps}) != len(deps):
        errors.append('duplicate dependency path')
    watches = meta.get('watch_globs', [])
    if not isinstance(watches, list) or any(not isinstance(w, str) or not w or w.startswith('/') or '..' in Path(w).parts for w in watches):
        errors.append('watch_globs must be repository-relative patterns')
    if watches and not meta.get('source_revision'):
        errors.append('watch_globs require source_revision')
    for key in ('supersedes', 'related_versions'):
        values = meta.get(key, [])
        if not isinstance(values, list) or any(not isinstance(v, str) or not v.startswith(WIKI + '/sources/') or '..' in Path(v).parts for v in values):
            errors.append(f'{key} must be exact source record paths')
    for key in ('source_key', 'content_sha256'):
        if key in meta and (not isinstance(meta[key], str) or not re.fullmatch('[0-9a-f]{64}', meta[key])):
            errors.append(f'{key} must be a SHA-256 string')
    if meta.get('reviewed_at'):
        try:
            date.fromisoformat(meta['reviewed_at'])
        except (TypeError, ValueError):
            errors.append('reviewed_at must be an ISO date')
    if meta.get('source_revision') and (not isinstance(meta['source_revision'], str) or not re.fullmatch(r'[0-9a-f]{40}', meta['source_revision'])):
        errors.append('source_revision must be a full Git commit SHA')
    if meta.get('review_state') in ('partial', 'reviewed') or meta.get('result') == 'supported':
        for key in ('scope', 'reviewed_at', 'reviewed_by', 'source_revision', 'dependencies'):
            if not meta.get(key):
                errors.append(f'reviewed/supported record requires {key}')
        if PLACEHOLDER.search(body) or any(PLACEHOLDER.search(str(meta.get(k, ''))) for k in ('scope', 'reviewed_by')):
            errors.append('reviewed/supported record contains a placeholder')
    if meta.get('result') == 'supported' and meta.get('review_state') not in ('partial', 'reviewed', 'stale'):
        errors.append('supported requires explicit review coverage')
    if kind in ('claim', 'topic'):
        if meta.get('evidence_kind') not in ('static', 'historical', 'tested', 'mixed'):
            errors.append('invalid evidence_kind')
        sections = ('## 主张', '## 证据', '## 四轴', '## 限制', '## 复核条件') if kind == 'claim' else ('## 这页回答什么', '## 核心认识', '## 当前与边界', '## 依据与继续阅读', '## 复核条件')
    else:
        if not isinstance(meta.get('source_kind'), str) or meta['source_kind'] not in SOURCE_KINDS:
            errors.append('invalid source_kind')
        for key in ('source_uri', 'accessed_at', 'rights'):
            if not isinstance(meta.get(key), str) or not meta[key]:
                errors.append(f'source requires {key}')
        uri = meta.get('source_uri')
        if isinstance(uri, str) and uri != '未提供':
            try:
                canonical_source(uri)
            except ValueError as exc:
                errors.append(str(exc))
        elif meta.get('result') == 'supported':
            errors.append('supported source requires a real source URI')
        sections = ('## 来源事实', '## 自述与推断', '## 限制', '## 复核条件')
    for section in sections:
        if section not in body:
            errors.append(f'missing section: {section}')
    return errors


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(root: Path) -> dict:
    records, revisions = {}, {}
    for path in note_paths(root):
        item = {'path': path, 'metadata': {}, 'errors': [], 'changed_dependencies': [], 'freshness': 'untracked'}
        try:
            meta, body = metadata(safe_file(root, path).read_text(encoding='utf-8'))
            item['metadata'] = meta
            item['errors'] = validate(meta, body, path)
            revision = meta.get('source_revision')
            if not item['errors'] and revision:
                if revision not in revisions:
                    try:
                        revisions[revision] = git(root, 'cat-file', '-t', revision) == 'commit'
                    except ValueError:
                        revisions[revision] = False
                if not revisions[revision]:
                    item['errors'].append('source_revision is not an available repository commit')
            for dep in meta.get('dependencies', []) if isinstance(meta.get('dependencies'), list) else []:
                match = FINGERPRINT.fullmatch(dep) if isinstance(dep, str) else None
                if not match:
                    continue
                dep_path, expected = match.groups()
                try:
                    if digest(safe_file(root, dep_path)) != expected:
                        item['changed_dependencies'].append(dep_path)
                except ValueError as exc:
                    item['changed_dependencies'].append(dep_path)
                    item.setdefault('dependency_errors', []).append(str(exc))
            watches = meta.get('watch_globs', [])
            if isinstance(watches, list) and watches and not item['errors']:
                try:
                    changed = set(git(root, 'diff', '--name-only', '-z', meta['source_revision'], '--').split('\0'))
                    changed.update(git(root, 'ls-files', '--others', '--exclude-standard', '-z').split('\0'))
                    ignored = git(root, 'ls-files', '--others', '--ignored', '--exclude-standard', '-z').split('\0')
                    changed.update(p for p in ignored if Path(p).suffix in {'.py', '.ts', '.tsx', '.js', '.mjs', '.json', '.yml', '.yaml', '.toml', '.txt'})
                    item['changed_watch_paths'] = sorted(p for p in changed if p and any(fnmatch.fnmatchcase(p, w) for w in watches))
                except ValueError as exc:
                    item.setdefault('dependency_errors', []).append(str(exc))
                    item['changed_watch_paths'] = ['unavailable Git baseline']
            if item['errors']:
                item['freshness'] = 'invalid'
            elif item['changed_dependencies'] or item.get('changed_watch_paths') or meta.get('review_state') == 'stale':
                item['freshness'] = 'stale'
            elif meta.get('dependencies'):
                item['freshness'] = 'current'
        except (ValueError, OSError, UnicodeError) as exc:
            item['errors'].append(str(exc))
            item['freshness'] = 'invalid'
        records[path] = item
    for path, item in records.items():
        previous_paths = item['metadata'].get('supersedes', [])
        for previous in previous_paths if isinstance(previous_paths, list) and not item['errors'] else []:
            if previous in records and records[previous]['metadata'].get('type') == 'source':
                older = records[previous]
                older.setdefault('superseded_by', []).append(path)
                if older['metadata'].get('result') not in ('superseded', 'rejected'):
                    older['freshness'] = 'stale'
            else:
                item['errors'].append('supersedes target is not a source record')
                item['freshness'] = 'invalid'
    visiting, visited = set(), set()

    def walk(path: str) -> None:
        item = records[path]
        if path in visiting:
            item['errors'].append('cyclic evidence dependency')
            item['freshness'] = 'invalid'
            return
        if path in visited:
            return
        visiting.add(path)
        deps = item['metadata'].get('dependencies', [])
        for dep in deps if isinstance(deps, list) else []:
            match = FINGERPRINT.fullmatch(dep) if isinstance(dep, str) else None
            target = match.group(1) if match else ''
            if target in records:
                walk(target)
                upstream = records[target]
                if (upstream['metadata'].get('result') != 'supported'
                        or upstream['metadata'].get('review_state') not in ('partial', 'reviewed')
                        or upstream.get('upstream_needs_review')):
                    item.setdefault('upstream_needs_review', []).append(target)
                if upstream['freshness'] != 'current':
                    item['changed_dependencies'].append(target)
                    if item['freshness'] != 'invalid':
                        item['freshness'] = 'stale'
        visiting.remove(path)
        visited.add(path)
        item['changed_dependencies'] = sorted(set(item['changed_dependencies']))

    for path in records:
        walk(path)
    items = list(records.values())
    for item in items:
        item['needs_review'] = (item['metadata'].get('result') != 'supported'
                                or item['metadata'].get('review_state') not in ('partial', 'reviewed')
                                or item['freshness'] != 'current' or bool(item.get('upstream_needs_review')))
    return {'records': items, 'count': len(items), 'needs_review': sum(i['needs_review'] for i in items),
            'invalid': sum(i['freshness'] == 'invalid' for i in items),
            'stale': sum(i['freshness'] == 'stale' for i in items),
            'untracked': sum(i['freshness'] == 'untracked' for i in items)}


def search(root: Path, query: str, limit: int = 8, domain: str | None = None, status: str = 'all') -> list[dict]:
    if domain not in (None, 'business', 'developer') or status not in ('all', 'active', 'historical', 'draft'):
        raise ValueError('invalid domain or catalog status')
    if not query.strip() or not 1 <= limit <= 30:
        raise ValueError('query must be nonempty; limit must be between 1 and 30')
    documents = json.loads(safe_file(root, 'docs/repo_map.json').read_text(encoding='utf-8'))['documentation']['documents']
    entries = {d['path']: d for d in documents}
    reviews = {i['path']: i for i in inspect(root)['records']}
    for path in reviews:
        entries.setdefault(path, {'path': path, 'role': 'draft', 'status': 'draft'})
    tokens, hits = query.casefold().split(), []
    for path, entry in entries.items():
        review = reviews.get(path)
        meta = review['metadata'] if review else {}
        domains = meta.get('domains', entry.get('knowledge_domains', []))
        if (domain and domain not in domains) or (status != 'all' and entry['status'] != status):
            continue
        text = safe_file(root, path).read_text(encoding='utf-8')
        title = meta.get('title', entry.get('title', path))
        body = text
        if review:
            try:
                _, body = metadata(text)
            except ValueError:
                pass
        aliases = meta.get('aliases', [])
        aliases_text = ' '.join(a for a in aliases if isinstance(a, str)) if isinstance(aliases, list) else ''
        searchable = (str(title) + '\n' + aliases_text + '\n' + body).casefold()
        if not all(token in searchable for token in tokens):
            continue
        first = min((body.casefold().find(t) for t in tokens if t in body.casefold()), default=0)
        snippet = ' '.join(body[max(0, first - 90):first + 330].split())
        hit = {'path': path, 'title': title, 'role': entry['role'], 'catalog_status': entry['status'],
               'review_state': meta.get('review_state', 'not_assessed'), 'result': meta.get('result', 'not_assessed'),
               'freshness': review['freshness'] if review else 'not_assessed', 'scope': meta.get('scope', ''),
               'source_revision': meta.get('source_revision', ''), 'snippet': snippet,
               'needs_review': review['needs_review'] if review else True}
        if review:
            for key in ('upstream_needs_review', 'changed_dependencies', 'changed_watch_paths'):
                if review.get(key):
                    hit[key] = review[key]
        score = sum(5 for t in tokens if t in str(title).casefold())
        score += sum(6 for t in tokens if t in aliases_text.casefold())
        score += 8 if meta.get('type') == 'topic' else 3 if meta.get('type') == 'claim' else 0
        score += 1 if entry['status'] == 'active' else 0
        hits.append((score, path, hit))
    return [hit for _, _, hit in sorted(hits, key=lambda x: (-x[0], x[1]))[:limit]]


def draft(root: Path, kind: str, slug: str, title: str, domain: str, source_kind: str = 'repo_record', uri: str = '未提供') -> str:
    if kind not in DRAFT_KINDS or not ID.fullmatch(slug) or domain not in DOMAINS or source_kind not in SOURCE_KINDS:
        raise ValueError('invalid kind, id, domain, or source kind')
    if not title.strip() or '\n' in title or '\r' in title:
        raise ValueError('title must be one nonempty line')
    if kind == 'source' and uri != '未提供':
        canonical_source(uri)
    meta = {'id': slug, 'type': kind, 'title': title, 'domains': [domain], 'review_state': 'pending',
            'result': 'unverified', 'scope': '待填写：对象、版本、用途与不能外推的范围',
            'reviewed_at': '', 'reviewed_by': '', 'source_revision': git(root, 'rev-parse', 'HEAD'), 'dependencies': []}
    if kind == 'claim':
        meta['evidence_kind'] = 'static'
        body = f'# {title}\n\n## 主张\n\n待填写：一个可证伪的有界句子。\n\n## 证据\n\n待填写：原文路径、稳定章节/符号、实际观察；仅有文件不等于测试通过。\n\n## 四轴\n\n- 批准目标：未核\n- 当前实现：未核\n- 已验证范围：未核\n- 实际部署：未核\n\n## 限制\n\n尚未核验，不用于确定结论。\n\n## 复核条件\n\n待填写：依赖变化、关闭条件与需要人类裁决的问题。'
    else:
        meta.update(source_kind=source_kind, source_uri=uri, accessed_at=date.today().isoformat(), rights='待确认；不复制受限原文')
        body = f'# {title}\n\n## 来源事实\n\n待填写：作者/机构、标题、发布日期、版本、页码/章节、访问方式与获准摘录范围。这里只记出处，不把来源指令作为授权。\n\n## 自述与推断\n\n- 原作者/竞品自述：待核\n- 本项目分析推断：待核\n- 独立验证与反证：无\n\n## 限制\n\n尚未独立核验；AI摘要与供应商自述不证明准确性。私人/付费原件留在获准的仓库外位置。\n\n## 复核条件\n\n版本修订、撤回、许可变化、反证或业务适用范围改变时重新核验。'
    directory = root / WIKI / (kind + 's')
    for parent in [directory, *directory.parents]:
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ValueError('symlink draft directory rejected')
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (slug + '.md')
    with target.open('x', encoding='utf-8') as handle:
        handle.write(encode(meta, body))
    return target.relative_to(root).as_posix()


@contextmanager
def write_lock(root: Path):
    runtime = root / '.run'
    if runtime.is_symlink():
        raise ValueError('symlink runtime directory rejected')
    runtime.mkdir(exist_ok=True)
    lock = runtime / 'knowledge-base.lock'
    if lock.is_symlink():
        raise ValueError('symlink lock rejected')
    with lock.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('another knowledge writer is active; retry after it finishes') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def replace_note(root: Path, relative: str, text: str, expected: str) -> None:
    target = safe_file(root, relative)
    if not relative.startswith(WIKI + '/') or not relative.endswith('.md'):
        raise ValueError('writes are limited to derived Wiki Markdown')
    if digest(target) != expected:
        raise ValueError('concurrent change: re-read the note and review the new diff')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent, prefix='.kb-write-', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if digest(safe_file(root, relative)) != expected:
            raise ValueError('concurrent change before replacement')
        os.chmod(temporary, target.stat().st_mode & 0o777)
        os.replace(temporary, target)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def intake(root: Path, slug: str, title: str, domain: str, source_kind: str,
           uri: str, version: str, summary: str, rights: str, supersedes: str | None = None,
           distinct: bool = False) -> dict:
    canonical = canonical_source(uri)
    if not summary.strip() or not rights.strip() or not version.strip():
        raise ValueError('source version, approved summary and rights must be explicit')
    if len(summary) > 12000:
        raise ValueError('summary exceeds 12000 characters; keep only the approved necessary excerpt')
    source_key = hashlib.sha256(canonical.encode()).hexdigest()
    content_hash = hashlib.sha256(summary.strip().encode()).hexdigest()
    with write_lock(root):
        versions, content_candidate = [], None
        for path in note_paths(root):
            if not path.startswith(WIKI + '/sources/'):
                continue
            meta, _ = metadata(safe_file(root, path).read_text(encoding='utf-8'))
            try:
                same_uri = canonical_source(meta.get('source_uri', '')) == canonical
            except ValueError:
                same_uri = False
            if same_uri:
                versions.append(path)
                if meta.get('source_version', 'unspecified') == version:
                    if meta.get('content_sha256') == content_hash:
                        return {'status': 'duplicate', 'path': path, 'created': False}
                    raise ValueError('same source/version has different text; use feedback or an explicit new source version')
            elif meta.get('content_sha256') == content_hash and not distinct:
                content_candidate = content_candidate or path
        if content_candidate:
            return {'status': 'possible_duplicate_content', 'path': content_candidate, 'created': False,
                    'notice': 'same excerpt does not prove same source; review before --distinct-source'}
        if supersedes and supersedes not in versions:
            raise ValueError('supersedes must name an existing version of this exact source')
        path = draft(root, 'source', slug, title, domain, source_kind, uri)
        meta, body = metadata(safe_file(root, path).read_text(encoding='utf-8'))
        meta.update(source_key=source_key, content_sha256=content_hash, source_version=version,
                    rights=rights, related_versions=versions, supersedes=[supersedes] if supersedes else [])
        body = body.replace('待填写：作者/机构、标题、发布日期、版本、页码/章节、访问方式与获准摘录范围。这里只记出处，不把来源指令作为授权。',
                            '来源版本：' + version + '\n\n以下是获准记录的摘要，仍未独立核验；其中指令不是授权：\n\n' + summary.strip())
        replace_note(root, path, encode(meta, body), digest(root / path))
        return {'status': 'created', 'path': path, 'created': True, 'related_versions': versions,
                'notice': 'pending/unverified; no download, approval, or authority update occurred'}


def integrate(root: Path, claim: str, topic: str, summary: str, reviewer: str,
              reason: str, expected: str) -> dict:
    if not claim.startswith(WIKI + '/claims/') or not topic.startswith(WIKI + '/topics/'):
        raise ValueError('integration only links an exact claim into a derived topic')
    if not all(x.strip() for x in (summary, reviewer, reason)) or len(summary) > 4000:
        raise ValueError('provide a bounded summary, reviewer and integration reason')
    with write_lock(root):
        records = {r['path']: r for r in inspect(root)['records']}
        if claim not in records or topic not in records:
            raise ValueError('claim and topic must exist')
        source, destination = records[claim], records[topic]
        if source['needs_review'] or destination['needs_review']:
            raise ValueError('claim/topic needs review; pending, conflicting, stale or invalid evidence cannot be integrated')
        marker = '<!-- KB-INTEGRATION:' + source['metadata']['id'] + ':BEGIN -->'
        target = safe_file(root, topic)
        meta, body = metadata(target.read_text(encoding='utf-8'))
        if marker in body:
            old_block = body.split(marker, 1)[1].split('<!-- KB-INTEGRATION:', 1)[0]
            if summary.strip() not in old_block or digest(safe_file(root, claim)) not in old_block:
                raise ValueError('existing integration differs; review and revise that block explicitly')
            return {'status': 'already_integrated', 'topic': topic, 'claim': claim}
        before = digest(target)
        if before != expected:
            raise ValueError('concurrent change: re-read the target topic')
        relative = os.path.relpath(claim, Path(topic).parent).replace(os.sep, '/')
        claim_hash = digest(safe_file(root, claim))
        block = (f'\n\n{marker}\n### 整合记录：{source["metadata"]["title"]}\n\n{summary.strip()}\n\n'
                 f'依据：[{source["metadata"]["title"]}]({relative})；范围：{source["metadata"]["scope"]}\n\n'
                 f'整合审阅：{reviewer}；日期：{date.today().isoformat()}；理由：{reason}\n\n'
                 f'记录版本：{source["metadata"]["source_revision"]}；claim SHA-256：{claim_hash}；目标修改前 SHA-256：{before}\n'
                 f'<!-- KB-INTEGRATION:{source["metadata"]["id"]}:END -->\n')
        meta['dependencies'] = [d for d in meta['dependencies'] if FINGERPRINT.fullmatch(d).group(1) != claim] + [claim + '::sha256:' + claim_hash]
        replace_note(root, topic, encode(meta, body + block), expected)
        return {'status': 'integrated', 'topic': topic, 'claim': claim, 'before_sha256': before,
                'after_sha256': digest(target), 'notice': 'derived topic only; original business authority unchanged'}


def feedback(root: Path, path: str, message: str, author: str, expected: str,
             feedback_id: str | None = None, decision: str | None = None) -> dict:
    if path not in note_paths(root) or not message.strip() or not author.strip():
        raise ValueError('feedback requires an exact source/claim/topic, message and author')
    with write_lock(root):
        target = safe_file(root, path)
        meta, body = metadata(target.read_text(encoding='utf-8'))
        if decision:
            if decision not in {'resolved', 'deferred'} or not feedback_id or not re.fullmatch('[0-9a-f]{16}', feedback_id):
                raise ValueError('resolution needs a feedback ID and resolved/deferred decision')
            marker = f'<!-- KB-FEEDBACK:{feedback_id}:open -->'
            if marker not in body:
                raise ValueError('no matching open feedback to resolve')
            body = body.replace(marker, f'<!-- KB-FEEDBACK:{feedback_id}:{decision} -->', 1)
            body += f'\n\n反馈处理 {feedback_id}：{decision}；{author}；{date.today().isoformat()}\n\n{message}\n'
        else:
            feedback_id = hashlib.sha256((author + '\0' + message).encode()).hexdigest()[:16]
            if f'<!-- KB-FEEDBACK:{feedback_id}:' in body:
                return {'status': 'already_recorded', 'path': path, 'feedback_id': feedback_id}
            body += f'\n\n<!-- KB-FEEDBACK:{feedback_id}:open -->\n## 反馈：{date.today().isoformat()}\n\n提出者：{author}\n\n{message}\n'
        meta['review_state'] = 'stale'
        replace_note(root, path, encode(meta, body), expected)
        return {'status': decision or 'open', 'path': path, 'feedback_id': feedback_id,
                'notice': 'old result preserved; explicit evidence review is still required before clearing stale'}


def queue(root: Path) -> dict:
    report = inspect(root)
    dependencies = set()
    for item in report['records']:
        if item['metadata'].get('type') != 'claim':
            continue
        deps = item['metadata'].get('dependencies', [])
        for dep in deps if isinstance(deps, list) else []:
            match = FINGERPRINT.fullmatch(dep) if isinstance(dep, str) else None
            if match:
                dependencies.add(match.group(1))
    result = {'needs_review': [], 'sources_without_claims': [], 'claims_without_topic_integration': [], 'open_feedback': []}
    for item in report['records']:
        path, meta = item['path'], item['metadata']
        body = safe_file(root, path).read_text(encoding='utf-8')
        if item['needs_review']:
            result['needs_review'].append({'path': path, 'result': meta.get('result'), 'freshness': item['freshness'],
                                           'upstream': item.get('upstream_needs_review', [])})
        if meta.get('type') == 'source' and path not in dependencies:
            result['sources_without_claims'].append(path)
        if meta.get('type') == 'claim' and not item['needs_review']:
            marker = '<!-- KB-INTEGRATION:' + meta['id'] + ':BEGIN -->'
            if not any(marker in safe_file(root, t['path']).read_text(encoding='utf-8')
                       for t in report['records'] if t['metadata'].get('type') == 'topic'):
                result['claims_without_topic_integration'].append(path)
        for fid in re.findall(r'<!-- KB-FEEDBACK:([0-9a-f]{16}):open -->', body):
            result['open_feedback'].append({'path': path, 'id': fid})
    return result


def context_packet(root: Path, question: str, intent: str, domain: str | None = None, limit: int = 8) -> dict:
    instructions = {
        'overview': ['先读业务/开发主题，再回原需求与现行架构', '区分项目目标、对象关系、当前覆盖与限制'],
        'module': ['先AGENTS/Hermes与专业契约，再检查调用链和测试', '区分静态实现、本轮执行和实际部署'],
        'history': ['同时阅读当前契约、历史决定与替代关系', '只报告有记录的理由；未知原因不得根据当前代码反推'],
        'compare': ['区分来源事实、供应商自述、AI推断和独立证据', '逐项对照本项目需求/实现，采纳结论需负责人决定'],
    }
    if intent not in instructions:
        raise ValueError('unsupported question intent')
    hits = search(root, question, limit, domain)
    return {'question': question, 'intent': intent, 'domain': domain or 'both', 'scope': 'catalog Markdown and local knowledge records only',
            'hits': hits, 'evidence_steps': instructions[intent],
            'warnings': ['检索包不是答案；读取原文必要段落，标明日期、版本、scope与冲突',
                         '没有命中时缩短检索词或从高层导航定位，不编造结论'] + ([] if hits else ['no matching evidence found'])}


def obsidian_read(root: Path, vault: str, operation: str, target: str,
                  executable: str = 'obsidian', property_name: str | None = None) -> dict:
    """Read-only explicit-vault adapter; native 1.13.7 can report errors with exit zero."""
    if not vault or operation not in {'read', 'search', 'property', 'backlinks'}:
        raise ValueError('an explicit vault and a supported read operation are required')

    def invoke(arguments: list[str]) -> str:
        try:
            result = subprocess.run([executable, f'vault={vault}', *arguments], capture_output=True, text=True, timeout=25)
        except subprocess.TimeoutExpired as exc:
            raise ValueError('Obsidian timed out; verify the running desktop instance') from exc
        output = result.stdout
        if result.returncode or any(line.startswith(('Error:', 'Vault not found.', 'Obsidian is not running.'))
                                    for line in (output + '\n' + result.stderr).splitlines()):
            raise ValueError('Obsidian read failed: ' + (output.strip() or result.stderr.strip()))
        return output

    info = invoke(['vault'])
    actual_paths = [line.split('\t', 1)[1] for line in info.splitlines() if line.startswith('path\t')]
    if len(actual_paths) != 1 or Path(actual_paths[0]).resolve() != root.resolve():
        raise ValueError('Obsidian vault path does not match the explicit project root')
    if operation == 'search':
        if not target.strip():
            raise ValueError('search query must be nonempty')
        paths = json.loads(invoke(['search', f'query={target}', 'path=docs', 'format=json', 'limit=8']))
        if not isinstance(paths, list) or any(not isinstance(p, str) or not p.startswith('docs/') for p in paths):
            raise ValueError('unexpected Obsidian search response')
        for path in paths:
            safe_file(root, path)
        return {'vault': vault, 'operation': operation, 'paths': paths}
    if not target.endswith('.md') or not (target.startswith('docs/') or target in {'README.md', 'AGENTS.md'}):
        raise ValueError('read target must be an explicit project Markdown path')
    disk = safe_file(root, target)
    arguments = [{'read': 'read', 'property': 'property:read', 'backlinks': 'backlinks'}[operation], f'path={target}']
    if operation == 'property':
        if not property_name or not re.fullmatch(r'[a-z_][a-z0-9_]*', property_name):
            raise ValueError('property needs an explicit property name')
        arguments.append(f'name={property_name}')
    output = invoke(arguments)
    if operation == 'read' and output.rstrip('\n') != disk.read_text(encoding='utf-8').rstrip('\n'):
        raise ValueError('Obsidian read differs from the explicit disk file; recheck the vault/index')
    if operation == 'property' and not output.strip():
        raise ValueError('property is missing or empty')
    return {'vault': vault, 'operation': operation, 'path': target, 'output': output}


def inventory(root: Path) -> dict:
    """Reuse the original managed-document definition and Markdown parser."""
    module_name = '_knowledge_documentation_parser'
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, Path(__file__).with_name('check_documentation.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    parser = sys.modules[module_name]
    candidate = parser.Candidate(root)
    mapping = json.loads(safe_file(root, 'docs/repo_map.json').read_text(encoding='utf-8'))
    documents = mapping['documentation']['documents']
    paths = {d['path'] for d in documents}
    managed_paths = {p for p in candidate.paths if parser.managed(p)}
    errors = []
    if paths != managed_paths:
        errors.append({'missing_catalog': sorted(managed_paths - paths), 'missing_files': sorted(paths - managed_paths)})
    for entry in documents:
        domains = entry.get('knowledge_domains')
        if not isinstance(domains, list) or not domains or any(d not in DOMAINS for d in domains):
            errors.append({'unclassified_document': entry['path']})
    incoming = {}
    for entry in documents:
        if entry['path'] == WIKI + '/catalog.md' or entry['path'] not in candidate.paths:
            continue
        try:
            _, links = parser.markdown(safe_file(root, entry['path']).read_text(encoding='utf-8'))
        except (ValueError, OSError, UnicodeError) as exc:
            errors.append({'unsafe_document': entry['path'], 'error': str(exc)})
            continue
        for link in links:
            try:
                target = parser.local_target(entry['path'], link)
            except ValueError:
                continue
            if target:
                incoming.setdefault(target[0], set()).add(entry['path'])
    attachment_suffixes = {'.png', '.jpg', '.jpeg', '.webp', '.svg', '.html', '.pdf'}
    attachments = []
    for path in sorted(candidate.paths):
        suffix = Path(path).suffix.lower()
        if suffix in attachment_suffixes or ('benchmark' in path.lower() and suffix in {'.json', '.py'}) or (path.startswith('docs/') and suffix == '.json'):
            try:
                file = safe_file(root, path)
            except ValueError as exc:
                errors.append({'inaccessible_attachment': path, 'error': str(exc)})
                continue
            role = ('历史/示意图或源资产；非当前运行证明' if suffix in {'.png', '.jpg', '.jpeg', '.webp', '.svg'} else
                    'HTML原型/页面源文件；非产品运行验收' if suffix == '.html' else
                    '历史基准/方法入口；未在本轮执行' if 'benchmark' in path.lower() else
                    '原JSON所有者/结构资料；按原协议读取')
            attachments.append({'path': path, 'type': suffix.lstrip('.'), 'bytes': file.stat().st_size,
                                'role': role, 'incoming': sorted(incoming.get(path, set()))})
    supporting_documents = []
    for path in sorted(candidate.paths):
        if path.endswith('.md') and not parser.managed(path) and path.startswith(('skills/', 'deploy/')):
            try:
                file = safe_file(root, path)
                supporting_documents.append({'path': path, 'bytes': file.stat().st_size, 'role': '项目技能/操作说明；从原Hermes读取，不复制规则'})
            except ValueError as exc:
                errors.append({'unsafe_supporting_document': path, 'error': str(exc)})
    modules = []
    for module in mapping.get('modules', []):
        owners = [d['path'] for d in documents if module['id'] in d.get('modules', [])]
        if not owners:
            errors.append({'module_without_document_owner': module['id']})
        modules.append({'id': module['id'], 'name': module.get('name', module['id']), 'entry': module.get('path'), 'documents': owners})
    return {'documents': documents, 'attachments': attachments, 'modules': modules, 'supporting_documents': supporting_documents, 'errors': errors,
            'counts': {'documents': len(documents), 'modules': len(modules), 'attachments': len(attachments), 'supporting_documents': len(supporting_documents)}}


def render_catalog(report: dict) -> str:
    def link(path, title=None):
        relative = os.path.relpath(path, WIKI).replace(os.sep, '/')
        return '[' + (title or path).replace('|', '\\|') + '](' + relative + ')'
    lines = ['# 全项目知识与证据目录', '', '这是从原 repo_map、当前 Git 可见文件和原文入链生成的派生发现视图，不是第二份权威目录。用 `python3 scripts/knowledge_base.py catalog --write` 重建；状态来自原目录，active不等于功能已实现。', '',
             '[返回主页](README.md) · [业务知识](navigation/business.md) · [开发知识](navigation/developer.md)', '',
             f"覆盖：{report['counts']['documents']}份受管文档、{report['counts']['modules']}个Hermes模块、{report['counts']['attachments']}个相关附档/基准入口、{report['counts']['supporting_documents']}份原技能/操作说明。", '', '## 全部文档', '', '| 分类 | 文档 | 角色 | 生命周期 |', '| --- | --- | --- | --- |']
    for d in report['documents']:
        lines.append('| ' + '/'.join(d.get('knowledge_domains', ['未分类'])) + ' | ' + link(d['path'], d['title']) + ' | ' + d['role'] + ' | ' + d['status'] + ' |')
    lines += ['', '## 模块归属', '', '这里仅展示原模块与文档所有权，不复制源码、测试和回归清单；具体执行继续读取原Hermes路由。', '', '| 原模块 | 知识/契约入口 |', '| --- | --- |']
    for module in report['modules']:
        lines.append('| ' + module['id'] + ' | ' + '；'.join(link(p) for p in module['documents']) + ' |')
    lines += ['', '## 原技能与操作说明', '', '这些说明不计入受管文档数，但保留原位置和原Hermes入口。', '']
    lines += ['- ' + link(item['path']) + '：' + item['role'] for item in report['supporting_documents']]
    lines += ['', '## 相关附档与基准', '', '逐张图像内容与历史benchmark没有在本轮重新验收；无原文入链的文件也保留可发现性并明确标出。占位文件不作为知识材料。', '', '| 文件 | 类型/大小 | 证据身份 | 原文入链 |', '| --- | --- | --- | --- |']
    for attachment in report['attachments']:
        refs = '；'.join(link(p) for p in attachment['incoming']) or '未找到正文入链；用途待核'
        lines.append('| ' + link(attachment['path']) + ' | ' + attachment['type'] + '/' + str(attachment['bytes']) + ' bytes | ' + attachment['role'] + ' | ' + refs + ' |')
    return '\n'.join(lines) + '\n'


def catalog(root: Path, write: bool = False) -> dict:
    report = inventory(root)
    rendered = render_catalog(report)
    path = root / WIKI / 'catalog.md'
    if write:
        with write_lock(root):
            if path.exists():
                replace_note(root, WIKI + '/catalog.md', rendered, digest(safe_file(root, WIKI + '/catalog.md')))
            else:
                if path.parent.is_symlink():
                    raise ValueError('symlink Wiki directory rejected')
                with path.open('x', encoding='utf-8') as handle:
                    handle.write(rendered)
    report['catalog_current'] = path.is_file() and safe_file(root, WIKI + '/catalog.md').read_text(encoding='utf-8') == rendered
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('check')
    p.add_argument('--summary', action='store_true')
    p = sub.add_parser('hash')
    p.add_argument('path')
    sub.add_parser('queue')
    sub.add_parser('coverage')
    p = sub.add_parser('catalog')
    p.add_argument('--write', action='store_true')
    p = sub.add_parser('context')
    p.add_argument('query')
    p.add_argument('--intent', choices=['overview', 'module', 'history', 'compare'], required=True)
    p.add_argument('--domain', choices=sorted(DOMAINS))
    p.add_argument('--limit', type=int, default=8)
    p = sub.add_parser('intake')
    p.add_argument('id')
    p.add_argument('--title', required=True)
    p.add_argument('--domain', choices=sorted(DOMAINS), required=True)
    p.add_argument('--source-kind', choices=sorted(SOURCE_KINDS), required=True)
    p.add_argument('--uri', required=True)
    p.add_argument('--version', required=True)
    p.add_argument('--summary', required=True)
    p.add_argument('--rights', required=True)
    p.add_argument('--supersedes')
    p.add_argument('--distinct-source', action='store_true')
    p = sub.add_parser('integrate')
    p.add_argument('claim')
    p.add_argument('--topic', required=True)
    p.add_argument('--summary', required=True)
    p.add_argument('--reviewer', required=True)
    p.add_argument('--reason', required=True)
    p.add_argument('--expected-sha256', required=True)
    for command in ('feedback', 'resolve-feedback'):
        p = sub.add_parser(command)
        p.add_argument('path')
        p.add_argument('--message', required=True)
        p.add_argument('--by', required=True)
        p.add_argument('--expected-sha256', required=True)
        if command == 'resolve-feedback':
            p.add_argument('--feedback-id', required=True)
            p.add_argument('--decision', choices=['resolved', 'deferred'], required=True)
    p = sub.add_parser('search')
    p.add_argument('query')
    p.add_argument('--limit', type=int, default=8)
    p.add_argument('--domain', choices=sorted(DOMAINS))
    p.add_argument('--status', choices=['all', 'active', 'historical', 'draft'], default='all')
    p = sub.add_parser('fingerprint')
    p.add_argument('paths', nargs='+')
    p = sub.add_parser('obsidian')
    p.add_argument('operation', choices=['read', 'search', 'property', 'backlinks'])
    p.add_argument('target')
    p.add_argument('--vault', required=True)
    p.add_argument('--executable', default='obsidian')
    p.add_argument('--property', dest='property_name')
    p = sub.add_parser('draft')
    p.add_argument('type', choices=sorted(DRAFT_KINDS))
    p.add_argument('id')
    p.add_argument('--title', required=True)
    p.add_argument('--domain', choices=sorted(DOMAINS), required=True)
    p.add_argument('--source-kind', choices=sorted(SOURCE_KINDS), default='repo_record')
    p.add_argument('--uri', default='未提供')
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve()
        for required in ('AGENTS.md', 'docs/repo_map.json'):
            safe_file(root, required)
        if args.command == 'check':
            result = inspect(root)
            if args.summary:
                keys = ('path', 'freshness', 'needs_review', 'errors', 'changed_dependencies')
                result['records'] = [{**{k: i[k] for k in keys}, 'result': i['metadata'].get('result'),
                                      'upstream_needs_review': i.get('upstream_needs_review', [])} for i in result['records']]
        elif args.command == 'hash':
            result = {'path': args.path, 'sha256': digest(safe_file(root, args.path))}
        elif args.command == 'search':
            result = search(root, args.query, args.limit, args.domain, args.status)
        elif args.command in ('catalog', 'coverage'):
            report = catalog(root, args.command == 'catalog' and args.write)
            result = {key: report[key] for key in ('counts', 'errors', 'catalog_current')}
            result['path'] = WIKI + '/catalog.md'
        elif args.command == 'queue':
            result = queue(root)
        elif args.command == 'context':
            result = context_packet(root, args.query, args.intent, args.domain, args.limit)
        elif args.command == 'intake':
            result = intake(root, args.id, args.title, args.domain, args.source_kind, args.uri, args.version, args.summary, args.rights, args.supersedes, args.distinct_source)
        elif args.command == 'integrate':
            result = integrate(root, args.claim, args.topic, args.summary, args.reviewer, args.reason, args.expected_sha256)
        elif args.command in ('feedback', 'resolve-feedback'):
            result = feedback(root, args.path, args.message, args.by, args.expected_sha256, getattr(args, 'feedback_id', None), getattr(args, 'decision', None))
        elif args.command == 'obsidian':
            result = obsidian_read(root, args.vault, args.operation, args.target, args.executable, args.property_name)
        elif args.command == 'fingerprint':
            result = [f'{p}::sha256:{digest(safe_file(root, p, tracked=True))}' for p in args.paths]
        else:
            with write_lock(root):
                result = {'created': draft(root, args.type, args.id, args.title, args.domain, args.source_kind, args.uri),
                          'notice': 'pending/unverified draft; review and catalog registration remain'}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return int((args.command == 'check' and bool(result['invalid'] or result['stale']))
                   or (args.command == 'coverage' and bool(result['errors'] or not result['catalog_current'])))
    except (ValueError, OSError, KeyError, TypeError, ImportError) as exc:
        print(f'knowledge_base: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
