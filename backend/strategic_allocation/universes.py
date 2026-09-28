"""Pure scope previews and explicitly confirmed immutable versions."""
from pathlib import Path

from backend.product_pools.constants import UNIVERSE_SNAPSHOT_STORE
from backend.product_pools.scope_lifecycle import (
    bind_scope_mandate,
    checked_scope_mandate,
    scope_mandate_fields,
    normalize_scope_name,
    product_scope_names,
    require_artifact_index,
    scope_governance_lock,
)
from backend.custom_indicators.errors import ConflictError, ValidationError
from backend.sensitivity.repository import digest_json
from .sources import product_source
from .scope_facts import scope_fingerprint


def hashed(payload):
    return {**payload, 'preview_hash': digest_json(payload)}


class StrategicScopes:
    def __init__(self, artifacts, data):
        self.artifacts, self.data = artifacts, data
        # One library across both scope stores: the shared lock lives with the
        # strategic artifacts; peer product names come from the configured
        # product workspace (which never has to be the same directory).
        self.product_store = Path(getattr(data, "universe_dir", getattr(data, "data_dir", "."))) / UNIVERSE_SNAPSHOT_STORE
        self.scope_root = Path(self.artifacts.root)
        self._peer_scope_names = lambda: product_scope_names(self.product_store)

    def retired_universe_ids(self) -> set[str]:
        retired = set()
        for summary in self.artifacts.list("retirement"):
            item = self.artifacts.get(summary["id"], "retirement")
            if item.get("artifact_type") == "strategic_universe_retirement" and item.get("universe_id"):
                retired.add(item["universe_id"])
        return retired

    def active_universe_ids(self) -> set[str]:
        require_artifact_index(self.artifacts)
        items = [self.artifacts.get(summary["id"], "series") for summary in self.artifacts.list("series")]
        universes = [item for item in items if item.get("artifact_type") == "strategic_universe"]
        superseded = {item["supersedes_universe_id"] for item in universes if item.get("supersedes_universe_id")}
        return {item["id"] for item in universes} - superseded - self.retired_universe_ids()

    def _require_unique_universe_name(self, name: str, replacing_id: str | None = None) -> None:
        normalized = normalize_scope_name(name)
        for identifier in self.active_universe_ids():
            if identifier != replacing_id and normalize_scope_name(self.get_universe(identifier)["name"]) == normalized:
                raise ConflictError("UNIVERSE_NAME_CONFLICT", "研究范围名称已存在，请修改名称。", "request.name")
        for existing in self._peer_scope_names().values():
            if normalize_scope_name(existing) == normalized:
                raise ConflictError("UNIVERSE_NAME_CONFLICT", "研究范围名称已存在，请修改名称。", "request.name")

    def get(self, identifier, kind):
        item = self.artifacts.get(identifier, 'series')
        if item.get('artifact_type') != kind:
            raise ValidationError('SAA_VERSION_TYPE', '所选不可变版本类型不匹配。')
        return item

    def get_universe(self, identifier):
        item = self.get(identifier, 'strategic_universe')
        return {**item, **scope_mandate_fields(self.scope_root, 'strategic', item)}

    def bind_mandate(self, identifier, mandate_id):
        return bind_scope_mandate(self.scope_root, 'strategic', self.get_universe(identifier), mandate_id)

    def get_mapping(self, identifier):
        return self.get(identifier, 'implementation_mapping')

    def preview_universe(self, request):
        # Preserve old no-proxy/no-limit hashes, but retain explicit cash/proxy null fields.
        definition = request.model_dump(mode='json')
        for asset in definition['assets']:
            for key in ('research_proxy', 'weight_limits'):
                if asset[key] is None:
                    del asset[key]
        return hashed({'definition': definition, 'implementation_status': 'unmapped',
                       'implementation_gaps': [a.id for a in request.assets], 'research_only': True})

    def confirm_universe(self, body):
        preview = self.preview_universe(body.request)
        if preview['preview_hash'] != body.preview_hash:
            raise ConflictError('SAA_UNIVERSE_PREVIEW_CHANGED', '战略范围已变化，请重新预览。')
        replaces = body.replaces_universe_id
        with scope_governance_lock(self.scope_root).locked():
            original = None
            if replaces is not None:
                original = self.get_universe(replaces)
                if replaces not in self.active_universe_ids():
                    raise ConflictError('SAA_UNIVERSE_INACTIVE', '该战略范围已删除或被替代，不能编辑；请刷新列表后操作当前版本。')
            self._require_unique_universe_name(body.request.name, replaces)
            binding = checked_scope_mandate(self.scope_root, body.mandate_id, original, allow_upgrade=True)
            if (original and original['definition'] == preview['definition']
                    and all(original.get(key) == binding.get(key) for key in ('mandate_id', 'mandate_hash'))):
                return original
            fields = {'artifact_type': 'strategic_universe', 'name': body.request.name, **preview,
                      'scope_fingerprint': scope_fingerprint(preview['definition']), **binding}
            if replaces is not None:
                fields['supersedes_universe_id'] = replaces
            return self.artifacts.save('series', fields)

    def retire_universe(self, identifier: str) -> dict:
        """Remove a scope from the active library; the frozen version stays readable."""

        with scope_governance_lock(self.scope_root).locked():
            require_artifact_index(self.artifacts)
            if identifier in self.retired_universe_ids():
                return {'deleted': True, 'id': identifier}
            universe = self.get_universe(identifier)
            self.artifacts.save('retirement', {
                'artifact_type': 'strategic_universe_retirement',
                'name': universe['name'],
                'universe_id': identifier,
                'universe_hash': universe['content_hash'],
                'research_only': True,
            })
        return {'deleted': True, 'id': identifier}

    def preview_mapping(self, request):
        universe = self.get_universe(request.strategic_universe_id)
        if universe['definition']['as_of'] > str(request.as_of):
            raise ValidationError('SAA_UNIVERSE_DATE', '映射研究日不能早于战略范围。')
        source = product_source(self.data, request.alloc_name, str(request.as_of))
        if source['universe_snapshot_id'] != request.universe_snapshot_id:
            raise ValidationError('SAA_MAPPING_DOMAIN', '真实代理大类绑定了不同的产品域快照；须显式选择同一域。')
        self.data.validate_application(source)
        domain = source['lineage']['universe']
        if domain.get('research_date') and domain['research_date'] > str(request.as_of):
            raise ValidationError('SAA_MAPPING_DOMAIN_DATE', '产品域研究日晚于映射研究日。')
        names = {a['id'] for a in universe['definition']['assets']}
        proxies = {a['id']: a for a in source['assets']}
        coverage = {a.strategic_asset_id: a.proxy_asset_id for a in request.assignments}
        if set(coverage) - names or set(coverage.values()) - set(proxies):
            raise ValidationError('SAA_MAPPING_AXIS', '映射引用未知战略资产或真实代理大类；不按名称自动匹配。')
        gaps = [a['id'] for a in universe['definition']['assets'] if a['id'] not in coverage]
        return hashed({'definition': request.model_dump(mode='json'), 'strategic_universe_hash': universe['content_hash'],
            'source_snapshot': source, 'implementation_status': 'incomplete' if gaps else 'complete',
            'implementation_gaps': gaps,
            'coverage': [{'strategic_asset_id': a['id'], 'proxy_asset_id': coverage.get(a['id']),
                          'status': 'mapped' if a['id'] in coverage else 'missing_products'} for a in universe['definition']['assets']],
            'research_only': True})

    def confirm_mapping(self, body):
        preview = self.preview_mapping(body.request)
        if preview['preview_hash'] != body.preview_hash:
            raise ConflictError('SAA_MAPPING_PREVIEW_CHANGED', '映射、产品域或真实源已变化，请重新预览。')
        return self.artifacts.save('series', {'artifact_type': 'implementation_mapping',
                                             'name': body.request.name, **preview})
