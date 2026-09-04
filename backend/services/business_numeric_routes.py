"""API boundary for fixed-signature browser-facing business arithmetic."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from business_numeric_numba import (
    business_numeric_execution_audit,
    grouped_numeric_controls_kernel,
    ledger_summary_kernel,
    trade_allocation_summary_kernel,
)


router = APIRouter(prefix="/api/business-numeric", tags=["business-numeric"])


class NumericControlGroup(BaseModel):
    key: str = Field(min_length=1, max_length=120)
    values: list[float] = Field(default_factory=list, max_length=10_000)
    target: float = 0.0
    tolerance: float = Field(default=1.0e-8, ge=0.0)


class NumericControlsRequest(BaseModel):
    groups: list[NumericControlGroup] = Field(min_length=1, max_length=500)


class AllocationItem(BaseModel):
    key: str = Field(min_length=1, max_length=120)
    quantity: float = Field(ge=0.0)


class TradeAllocationRequest(BaseModel):
    source_quantity: float = Field(ge=0.0)
    unit_price: float = Field(ge=0.0)
    allocations: list[AllocationItem] = Field(max_length=10_000)
    tolerance: float = Field(default=1.0e-8, ge=0.0)


class JournalLineInput(BaseModel):
    debit: float = Field(default=0.0, ge=0.0)
    credit: float = Field(default=0.0, ge=0.0)


class JournalVoucherInput(BaseModel):
    id: str = Field(min_length=1, max_length=120)
    entity: str = Field(min_length=1, max_length=120)
    lines: list[JournalLineInput] = Field(default_factory=list, max_length=100_000)


class LedgerSummaryRequest(BaseModel):
    entities: list[str] = Field(min_length=1, max_length=100)
    vouchers: list[JournalVoucherInput] = Field(default_factory=list, max_length=10_000)
    pending_flags: list[bool] = Field(default_factory=list, max_length=100_000)
    trial_debit: float = Field(default=0.0, ge=0.0)
    trial_credit: float = Field(default=0.0, ge=0.0)
    tolerance: float = Field(default=0.005, ge=0.0)


def _execution() -> dict:
    try:
        return business_numeric_execution_audit()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail="业务数值 NJIT 内核尚未完成启动预热。",
        ) from exc


@router.post("/controls")
def evaluate_numeric_controls(request: NumericControlsRequest):
    execution = _execution()
    seen: set[str] = set()
    flat_values: list[float] = []
    offsets = [0]
    for group in request.groups:
        if group.key in seen:
            raise HTTPException(status_code=422, detail=f"重复的数值控制键：{group.key}")
        seen.add(group.key)
        flat_values.extend(group.values)
        offsets.append(len(flat_values))

    result = grouped_numeric_controls_kernel(
        np.ascontiguousarray(flat_values, dtype=np.float64),
        np.ascontiguousarray(offsets, dtype=np.int64),
        np.ascontiguousarray([group.target for group in request.groups], dtype=np.float64),
        np.ascontiguousarray([group.tolerance for group in request.groups], dtype=np.float64),
    )
    totals, differences, within, positive, shares, status = result
    if status != 0:
        raise HTTPException(status_code=422, detail="数值控制输入必须是有限非负数且分组边界有效。")

    items = []
    for index, group in enumerate(request.groups):
        start = offsets[index]
        end = offsets[index + 1]
        items.append(
            {
                "key": group.key,
                "total": float(totals[index]),
                "difference": float(differences[index]),
                "within_tolerance": bool(within[index]),
                "positive": bool(positive[index]),
                "normalized_shares": [float(value) for value in shares[start:end]],
            }
        )
    return {"items": items, "execution": execution}


@router.post("/trade-allocation")
def evaluate_trade_allocation(request: TradeAllocationRequest):
    execution = _execution()
    keys = [item.key for item in request.allocations]
    if len(keys) != len(set(keys)):
        raise HTTPException(status_code=422, detail="成交分配键必须唯一。")
    source_amount, amounts, allocated_total, residual, balanced, status = (
        trade_allocation_summary_kernel(
            np.float64(request.source_quantity),
            np.float64(request.unit_price),
            np.ascontiguousarray(
                [item.quantity for item in request.allocations], dtype=np.float64
            ),
            np.float64(request.tolerance),
        )
    )
    if status != 0:
        raise HTTPException(status_code=422, detail="成交数量、价格和分配数量必须为有限非负数。")
    return {
        "source_quantity": request.source_quantity,
        "unit_price": request.unit_price,
        "source_amount": float(source_amount),
        "allocated_total": float(allocated_total),
        "residual": float(residual),
        "balanced": bool(balanced),
        "allocations": [
            {"key": key, "quantity": request.allocations[index].quantity, "amount": float(amounts[index])}
            for index, key in enumerate(keys)
        ],
        "execution": execution,
    }


@router.post("/ledger-summary")
def evaluate_ledger_summary(request: LedgerSummaryRequest):
    execution = _execution()
    if len(request.entities) != len(set(request.entities)):
        raise HTTPException(status_code=422, detail="核算主体类型必须唯一。")
    entity_lookup = {entity: index for index, entity in enumerate(request.entities)}
    voucher_ids = [voucher.id for voucher in request.vouchers]
    if len(voucher_ids) != len(set(voucher_ids)):
        raise HTTPException(status_code=422, detail="凭证编号必须唯一。")
    unknown = sorted({voucher.entity for voucher in request.vouchers} - set(entity_lookup))
    if unknown:
        raise HTTPException(status_code=422, detail=f"凭证包含未声明核算主体：{', '.join(unknown)}")

    debits: list[float] = []
    credits: list[float] = []
    offsets = [0]
    for voucher in request.vouchers:
        debits.extend(line.debit for line in voucher.lines)
        credits.extend(line.credit for line in voucher.lines)
        offsets.append(len(debits))

    result = ledger_summary_kernel(
        np.ascontiguousarray(debits, dtype=np.float64),
        np.ascontiguousarray(credits, dtype=np.float64),
        np.ascontiguousarray(offsets, dtype=np.int64),
        np.ascontiguousarray(
            [entity_lookup[voucher.entity] for voucher in request.vouchers],
            dtype=np.int64,
        ),
        np.int64(len(request.entities)),
        np.ascontiguousarray(request.pending_flags, dtype=np.uint8),
        np.float64(request.trial_debit),
        np.float64(request.trial_credit),
        np.float64(request.tolerance),
    )
    (
        voucher_debits,
        voucher_credits,
        voucher_differences,
        voucher_balanced,
        entity_debits,
        entity_credits,
        entity_differences,
        entity_voucher_counts,
        balanced_count,
        pending_count,
        trial_difference,
        trial_balanced,
        source_event_count,
        voucher_count,
        status,
    ) = result
    if status != 0:
        raise HTTPException(status_code=422, detail="凭证借贷输入必须是有限非负数且分组边界有效。")

    return {
        "metrics": {
            "source_event_count": int(source_event_count),
            "voucher_count": int(voucher_count),
            "balanced_count": int(balanced_count),
            "pending_count": int(pending_count),
        },
        "vouchers": [
            {
                "id": voucher.id,
                "debit": float(voucher_debits[index]),
                "credit": float(voucher_credits[index]),
                "difference": float(voucher_differences[index]),
                "balanced": bool(voucher_balanced[index]),
            }
            for index, voucher in enumerate(request.vouchers)
        ],
        "entities": [
            {
                "entity": entity,
                "voucher_count": int(entity_voucher_counts[index]),
                "debit": float(entity_debits[index]),
                "credit": float(entity_credits[index]),
                "difference": float(entity_differences[index]),
            }
            for index, entity in enumerate(request.entities)
        ],
        "trial": {
            "debit": request.trial_debit,
            "credit": request.trial_credit,
            "difference": float(trial_difference),
            "balanced": bool(trial_balanced),
        },
        "execution": execution,
    }


__all__ = ["router"]
