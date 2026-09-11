"""Explicit fixed-basket input binding; alignment is an I/O allocation boundary."""
import hashlib
import numpy as np
from custom_indicators.errors import ValidationError
from .numeric import availability_status_kernel
from backend.research_series.numba_kernels import align_values_kernel


def required_baskets(prepared):
    return {node.parameters.get("group", "market") for node in prepared.definition.nodes
            if node.id in prepared.order and node.op == "basket_source"}


def validate_baskets(request, prepared):
    for group in required_baskets(prepared):
        if len(request.context_baskets.get(group, [])) < 2:
            raise ValidationError("TIMING_BASKET_REQUIRED", f"请为{ '市场环境' if group == 'market' else '资产类别'}篮子明确选择至少 2 只 ETF；不会用单只产品代替市场广度。")


def bind_baskets(request, prepared, bars, load):
    panels, lineage = {}, {}
    for group in sorted(required_baskets(prepared)):
        members = request.context_baskets[group]
        panel = np.empty((len(members), bars.dates.size), dtype=np.float64)
        facts = []
        for index, code in enumerate(members):
            source = load(code)
            if source.lineage.get("snapshot") != bars.lineage.get("snapshot") or source.lineage.get("price_basis") != bars.lineage.get("price_basis"):
                raise ValidationError("TIMING_BASKET_SNAPSHOT_CHANGED", "环境篮子与研究产品的行情快照或价格口径不一致，请重新运行。")
            if availability_status_kernel(source.dates, source.available_days, 0, len(source.dates)):
                raise ValidationError("TIMING_BASKET_NOT_CAUSAL", f"篮子成员 {code} 存在晚于观察日期可得的数据。")
            if np.array_equal(source.dates, bars.dates):
                panel[index] = source.close
            else:
                # This existing I/O aligner accepts writable contiguous buffers.
                # Copies are confined to this once-per-member alignment boundary;
                # graph nodes reuse the resulting readonly panel and row views.
                panel[index] = align_values_kernel(np.array(bars.dates), np.array(source.dates), np.array(source.close))
            facts.append({"product_id": code, "source_hash": source.lineage.get("source_hash")})
        panel.setflags(write=False)
        panels[group] = panel
        digest = hashlib.sha256(memoryview(panel).cast("B")).hexdigest()
        lineage[group] = {"members": facts, "panel_hash": digest, "axis": "fixed_member_by_target_trading_day",
            "missing_policy": "any_member_unknown_keeps_aggregate_unknown", "fixed_universe_survivorship_warning": True}
    return panels, lineage
