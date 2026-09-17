"""Pure scope previews and explicitly confirmed immutable versions."""
from backend.custom_indicators.errors import ConflictError, ValidationError
from backend.sensitivity.repository import digest_json
from .sources import product_source


def hashed(payload):
    return {**payload, 'preview_hash': digest_json(payload)}


class StrategicScopes:
    def __init__(self, artifacts, data):
        self.artifacts, self.data = artifacts, data

    def get(self, identifier, kind):
        item = self.artifacts.get(identifier, 'series')
        if item.get('artifact_type') != kind:
            raise ValidationError('SAA_VERSION_TYPE', '所选不可变版本类型不匹配。')
        return item

    def get_universe(self, identifier):
        return self.get(identifier, 'strategic_universe')

    def get_mapping(self, identifier):
        return self.get(identifier, 'implementation_mapping')

    def preview_universe(self, request):
        return hashed({'definition': request.model_dump(mode='json'), 'implementation_status': 'unmapped',
                       'implementation_gaps': [a.id for a in request.assets], 'research_only': True})

    def confirm_universe(self, body):
        preview = self.preview_universe(body.request)
        if preview['preview_hash'] != body.preview_hash:
            raise ConflictError('SAA_UNIVERSE_PREVIEW_CHANGED', '战略范围已变化，请重新预览。')
        return self.artifacts.save('series', {'artifact_type': 'strategic_universe',
                                             'name': body.request.name, **preview})

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
