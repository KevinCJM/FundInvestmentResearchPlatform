"""Thin mutable governance boundary over the existing atomic JSON store."""
from __future__ import annotations
import copy
import json
import uuid
from pathlib import Path
from contextlib import contextmanager
from backend.custom_indicators.repository import AtomicJsonStore, utc_now
from backend.custom_indicators.errors import ConflictError, NotFoundError, ValidationError
from backend.data_storage import guard_path
from .risk_scale_contracts import DraftWrite, DraftUpdate, DraftView
from pydantic import ValidationError as SchemaError


class RiskScaleStore:
    def __init__(self, root: Path):
        self.document = AtomicJsonStore(Path(root) / 'risk_scale_state.json')

    def read(self) -> dict:
        guard_path(self.document.path)
        if self.document.path.exists() and (self.document.path.is_symlink() or self.document.path.stat().st_size > 16_000_000):
            raise ValidationError('RISK_SCALE_STATE_CORRUPT', '风险标尺治理记录不可读。')
        payload = self.document.read_unlocked()
        for key, empty in [('drafts', []), ('defaults', {}), ('retired', {}), ('history', []), ('schemes', {}), ('publications', {})]:
            payload.setdefault(key, empty)
            if not isinstance(payload[key], type(empty)):
                raise ValidationError('RISK_SCALE_STATE_CORRUPT', '风险标尺治理记录结构无效。')
        if any(not isinstance(x, dict) or not isinstance(x.get('id'), str) or type(x.get('revision')) is not int
               or not isinstance(x.get('editable_definition'), dict) for x in payload['drafts']):
            raise ValidationError('RISK_SCALE_STATE_CORRUPT', '风险标尺草稿记录结构无效。')
        if any(not isinstance(x, dict) or x.get('key') != key or type(x.get('revision')) is not int
               or x['revision'] < 1 or (x.get('version_id') is not None and not isinstance(x.get('content_hash'), str))
               for key, x in payload['defaults'].items()):
            raise ValidationError('RISK_SCALE_STATE_CORRUPT', '风险标尺默认记录结构无效。')
        if any(not isinstance(x, dict) for field in ('retired', 'schemes', 'publications') for x in payload[field].values()):
            raise ValidationError('RISK_SCALE_STATE_CORRUPT', '风险标尺治理条目结构无效。')
        try:
            for draft in payload['drafts']:
                DraftView.model_validate(draft)
            for scheme in payload['schemes'].values():
                if not isinstance(scheme.get('name'), str) or type(scheme.get('next_version', 1)) is not int or scheme.get('next_version', 1) < 1:
                    raise ValueError('scheme shape')
            for publication in payload['publications'].values():
                if (not isinstance(publication.get('request_hash'), str) or not isinstance(publication.get('scheme_id'), str)
                        or type(publication.get('version_number')) is not int or publication['version_number'] < 1):
                    raise ValueError('publication shape')
        except (SchemaError, ValueError):
            raise ValidationError('RISK_SCALE_STATE_CORRUPT', '风险标尺治理条目内容无效。') from None
        return payload

    @contextmanager
    def transaction(self):
        guard_path(self.document.path, write=True)
        with self.document.locked():
            payload = self.read()
            yield payload
            # Append-only history and pointer/revision are replaced together.
            if len(json.dumps(payload, allow_nan=False).encode()) > 16_000_000:
                raise ValidationError('RISK_SCALE_STATE_CAPACITY', '治理记录达到容量上限，请维护存储后重试。')
            self.document.write_unlocked(payload)

    def get_draft(self, identifier: str) -> dict:
        result = next((d for d in self.read()['drafts'] if d['id'] == identifier), None)
        if result is None:
            raise NotFoundError('DRAFT_NOT_FOUND', '草稿不存在或已删除。')
        return copy.deepcopy(result)

    @staticmethod
    def _validate_draft(body):
        try:
            size = len(json.dumps(body.editable_definition, allow_nan=False).encode())
        except (TypeError, ValueError) as exc:
            raise ValidationError('DRAFT_INVALID', '草稿包含无效或非有限数据。', 'editable_definition') from exc
        if size > 128_000:
            raise ValidationError('DRAFT_CAPACITY', '草稿超过 128 KB，请缩小编辑内容。')

    def create_draft(self, body: DraftWrite) -> dict:
        self._validate_draft(body)
        now = utc_now()
        item = {**body.model_dump(mode='json'), 'id': 'draft-' + uuid.uuid4().hex,
                'revision': 1, 'created_at': now, 'updated_at': now}
        with self.transaction() as state:
            state['drafts'].append(item)
            state['schemes'].setdefault(body.scheme_id, {'name': body.name, 'revision': 1})
        return item

    def update_draft(self, identifier: str, body: DraftUpdate) -> dict:
        self._validate_draft(body)
        with self.transaction() as state:
            item = next((d for d in state['drafts'] if d['id'] == identifier), None)
            if item is None:
                raise NotFoundError('DRAFT_NOT_FOUND', '草稿不存在或已删除。')
            if item['revision'] != body.expected_revision:
                raise ConflictError('REVISION_CONFLICT', '草稿已被修改，请重新加载。')
            if item['scheme_id'] != body.scheme_id:
                raise ValidationError('DRAFT_SCHEME_CHANGED', '修改草稿不能改变所属方案，请另存新草稿。')
            item.update(body.model_dump(mode='json', exclude={'expected_revision'}))
            item.update(revision=item['revision'] + 1, updated_at=utc_now())
            result = copy.deepcopy(item)
        return result

    def delete_draft(self, identifier: str, revision: int) -> dict:
        with self.transaction() as state:
            item = next((d for d in state['drafts'] if d['id'] == identifier), None)
            if item is None:
                raise NotFoundError('DRAFT_NOT_FOUND', '草稿不存在或已删除。')
            if item['revision'] != revision:
                raise ConflictError('REVISION_CONFLICT', '草稿已被修改，请重新加载。')
            state['drafts'].remove(item)
        return {'deleted': True, 'id': identifier}

    def reserve_version(self, scheme_id, name, operation, request_hash, existing_max=0):
        """Durable version reservation survives promotion/index crashes and other publishers."""
        with self.transaction() as state:
            previous = state['publications'].get(operation)
            if previous:
                if previous['request_hash'] != request_hash or previous['scheme_id'] != scheme_id:
                    raise ConflictError('IDEMPOTENCY_CONFLICT', '同一发布操作已保留给不同输入。')
                return previous['version_number']
            scheme = state['schemes'].setdefault(scheme_id, {'name': name, 'revision': 1})
            number = max(existing_max + 1, scheme.get('next_version', 1))
            scheme['next_version'] = number + 1
            state['publications'][operation] = {'request_hash': request_hash, 'scheme_id': scheme_id,
                                               'version_number': number}
            return number

    @staticmethod
    def key(currency: str, basis: str) -> str:
        return currency + ':' + basis

    @staticmethod
    def binding(state: dict, key: str) -> dict:
        return state['defaults'].get(key, {'key': key, 'revision': 0, 'version_id': None, 'content_hash': None})

    @classmethod
    def set_default(cls, state, key, item, expected_revision, action='activate', reason=''):
        previous = cls.binding(state, key)
        if previous['revision'] != expected_revision:
            raise ConflictError('DEFAULT_CHANGED', '默认引用已变化，请重新加载后确认。')
        current = {'key': key, 'revision': previous['revision'] + 1,
                   'version_id': item['id'] if item else None,
                   'content_hash': item['content_hash'] if item else None}
        state['defaults'][key] = current
        state['history'].append({'action': action, 'previous': previous, 'current': current,
                                 'reason': reason, 'at': utc_now()})
        return current
