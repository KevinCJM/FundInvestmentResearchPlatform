"""Institutional diagnostics and shared manual-review application boundary."""
from datetime import date
import numpy as np
from . import institution_kernels as numeric
from .institution_contracts import InstitutionalContext

TOPIC_LABELS = {'tax': '税务', 'regulation': '监管', 'currency_hedging': '币种与对冲',
                'leverage': '杠杆', 'special_liquidity': '特殊流动性'}


def review_blockers(mandate: dict, as_of: str) -> list[str]:
    raw = mandate.get('institutional_context')
    if raw is None:
        return []
    try:
        context = InstitutionalContext.model_validate(raw)
    except ValueError:
        return ['冻结的机构核验记录不完整；请复制目标重新核验，不能默认通过。']
    day = date.fromisoformat(as_of)
    reasons = []
    for item in context.review_items:
        if item.status not in {'researcher_checked', 'not_applicable'}:
            reasons.append(f'{TOPIC_LABELS[item.topic]}尚未完成人工核验。')
        elif not item.reviewed_on <= day < item.valid_until:
            reasons.append(f'{TOPIC_LABELS[item.topic]}的人工核验证据尚未生效或已过期。')
    return reasons


def diagnose_institution(mandate: dict) -> dict | None:
    raw = mandate.get('institutional_context')
    if raw is None:
        return None
    numeric.require_ready()
    context = InstitutionalContext.model_validate(raw)
    balance = None
    if context.balance_sheet:
        sheet = context.balance_sheet
        values = np.asarray([getattr(sheet, k) if getattr(sheet, k) is not None else np.nan for k in
            ('investable_assets', 'outside_assets', 'confirmed_liabilities', 'uncalled_commitments')], dtype=np.float64)
        values.flags.writeable = False
        result = numeric.balance_sheet_kernel(values)
        balance = {'as_of': str(sheet.as_of), 'currency': sheet.currency, 'source': sheet.source,
            'total_assets': float(result[0]) if np.isfinite(result[0]) else None,
            'net_assets_after_confirmed_liabilities': float(result[1]) if np.isfinite(result[1]) else None,
            'uncalled_commitments': sheet.uncalled_commitments,
            'status': 'research_snapshot', 'cashflow_model': 'existing_funding_plan_only'}
    return {'balance_sheet': balance, 'cash_reserve_weight': context.cash_reserve_weight,
            'review_blockers': review_blockers(mandate, mandate['as_of']),
            'current_review_blockers': review_blockers(mandate, str(date.today())),
            'automated_compliance': 'not_modelled', 'independent_approval': False,
            'execution': numeric.execution_audit()}
