"""One source resolver; product-backed requests retain their existing reader."""
import copy
from backend.custom_indicators.errors import ValidationError
from backend.sensitivity.repository import digest_json
from backend.strategy import equal_weights
from . import kernels


def product_source(data, alloc_name: str, as_of: str) -> dict:
    kernels.require_ready()
    frame, _, _ = data._configuration(alloc_name)
    names = frame['asset_name'].unique().tolist()
    weights = equal_weights(len(names))
    source = data.create_baseline({'alloc_name': alloc_name, 'name': '长期假设的数据来源',
        'as_of': as_of, 'weights': dict(zip(names, weights, strict=True))})
    seen = set()
    for asset in source['assets']:
        for product in asset['products']:
            key = (product['kind'], product['product_id'].upper())
            if key in seen:
                raise ValidationError('SAA_DUPLICATE_CLASS_PRODUCT', '同一产品跨大类出现，请先明确唯一预算归属。')
            seen.add(key)
    return source


def strategic_source(universe: dict, mapping: dict | None, as_of: str) -> dict:
    definition = universe['definition']
    if definition['as_of'] > as_of:
        raise ValidationError('SAA_UNIVERSE_DATE', '战略范围晚于假设研究日，请选择当日适用的范围。')
    source = copy.deepcopy(mapping['source_snapshot']) if mapping else {
        'alloc_name': None, 'universe_snapshot_id': None, 'data_release_id': None, 'lineage': {},
        'pit': {'status': 'research_only', 'reasons': ['纯前瞻战略研究，没有历史代理净值或历史PIT认证。']}}
    assignments = {a['strategic_asset_id']: a['proxy_asset_id'] for a in mapping['definition']['assignments']} if mapping else {}
    proxies = {a['id']: a for a in source.get('assets', [])}
    assets = []
    for asset in definition['assets']:
        proxy = proxies.get(assignments.get(asset['id']))
        assets.append({**copy.deepcopy(asset), 'base_weight': 0.0, 'min_weight': 0.0,
                       'max_weight': 1.0, 'max_abs_tilt': 0.1,
                       'products': copy.deepcopy(proxy['products']) if proxy else [],
                       'proxy_asset_id': proxy['id'] if proxy else None})
    gaps = [a['id'] for a in assets if not a['products']]
    reasons = [f"战略资产 {identifier} 缺少真实代理产品。" for identifier in gaps]
    if mapping and not mapping['definition']['as_of'] <= as_of < mapping['definition']['valid_until']:
        raise ValidationError('SAA_MAPPING_EXPIRED', '映射尚未生效或已到复核日，请确认适用的新映射。')
    source.update(name=definition['name'], as_of=as_of, assets=assets, group_limits=[],
                  strategic_universe_id=universe['id'], strategic_universe_snapshot=copy.deepcopy(universe),
                  implementation_mapping_id=mapping['id'] if mapping else None,
                  implementation_mapping_snapshot=copy.deepcopy(mapping),
                  implementation_status='incomplete' if gaps else 'complete',
                  implementation_gaps=gaps, apply_eligible=not gaps, apply_reasons=reasons)
    source['lineage'].update(strategic_universe_hash=universe['content_hash'],
                             implementation_mapping_hash=mapping['content_hash'] if mapping else None)
    return source


def verify_strategic_snapshot(baseline: dict, *, require_complete: bool = True) -> dict:
    """Reconstruct identity/products from frozen artifacts, never display hints."""
    universe = baseline.get('strategic_universe_snapshot') or {}
    mapping = baseline.get('implementation_mapping_snapshot')
    for item, expected_id in [(universe, baseline.get('strategic_universe_id')),
                              (mapping, baseline.get('implementation_mapping_id'))]:
        if item is None and expected_id is None:
            continue
        if (not item or item.get('id') != expected_id or not item.get('content_hash') or
                digest_json({k: v for k, v in item.items() if k != 'content_hash'}) != item['content_hash']):
            raise ValidationError('SAA_MAPPING_INTEGRITY', '战略范围或映射的不可变身份与哈希不一致。')
    if mapping and (mapping['definition']['strategic_universe_id'] != universe['id'] or
                    mapping.get('strategic_universe_hash') != universe['content_hash']):
        raise ValidationError('SAA_MAPPING_UNIVERSE', '实施映射属于不同的战略范围。')
    expected = strategic_source(universe, mapping, baseline['as_of'])
    if [a['id'] for a in expected['assets']] != [a['id'] for a in baseline['assets']]:
        raise ValidationError('SAA_MAPPING_AXIS', '战略资产轴与冻结范围不一致。')
    for original, actual in zip(expected['assets'], baseline['assets'], strict=True):
        if any(original.get(k) != actual.get(k) for k in ('products', 'proxy_asset_id', 'role', 'liquidity', 'currency')):
            raise ValidationError('SAA_MAPPING_PRODUCTS', '实际代理或经济角色与冻结映射不一致。')
    if expected['lineage'] != baseline['lineage'] or expected['universe_snapshot_id'] != baseline.get('universe_snapshot_id'):
        raise ValidationError('SAA_MAPPING_LINEAGE', '实施映射的产品域或源血缘不一致。')
    if require_complete and expected['implementation_status'] != 'complete':
        raise ValidationError('SAA_IMPLEMENTATION_INCOMPLETE', '战略范围尚有产品映射缺口；可继续前瞻SAA研究，补齐映射后才能进入TAA或产品应用。')
    return expected
