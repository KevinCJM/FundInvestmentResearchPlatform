"""Read saved SAA classes and real NAV without inventing historical PIT proof.

Pandas is confined to Parquet decoding, validation and axis alignment. The one
NAV-to-return computation delegates to the existing fixed-signature NJIT core.
No preview writes, downloads, zero-filled returns or implicit normalization.
"""
from __future__ import annotations

import hashlib
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from backend.custom_indicators.errors import NotFoundError, ValidationError
from backend.data_storage import guard_path
from backend.product_pools.constants import UNIVERSE_SNAPSHOT_STORE
from backend.product_pools.errors import ProductPoolError
from backend.product_pools.membership import InvestableUniverseMembership
from backend.product_pools.repository import AtomicProductPoolStore
from backend.sensitivity.repository import digest_json, file_hash
from backend.strategy import nav_to_returns_kernel, strategy_execution_audit, warm_strategy_numba_kernels

MAX_OBSERVATIONS = 10_000
MAX_ASSETS = 30
MAX_PARQUET_BYTES = 256_000_000


def warm_tactical_data() -> dict[str, Any]:
    return warm_strategy_numba_kernels()


def _date(value: Any, field: str, *, optional: bool = False) -> str | None:
    if optional and (value is None or value == ""):
        return None
    if not isinstance(value, str) or len(value) != 10:
        raise ValidationError("TAA_DATE_INVALID", f"{field} 须为 YYYY-MM-DD 日期。")
    try:
        parsed = pd.Timestamp(value)
        if pd.isna(parsed) or parsed.strftime("%Y-%m-%d") != value:
            raise ValueError("invalid date")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError("TAA_DATE_INVALID", f"{field} 须为有效日期。") from exc
    return value


def _number(value: Any, label: str, *, maximum: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValidationError("TAA_WEIGHT_INVALID", f"{label} 须为数值。")
    result = float(value)
    if not np.isfinite(result) or not 0 <= result <= maximum:
        raise ValidationError("TAA_WEIGHT_INVALID", f"{label} 须在 0 至 {maximum:g} 之间。")
    return result


def _one(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame.columns:
        return None
    values = frame[column].dropna().astype(str).unique().tolist()
    values = [value for value in values if value]
    if len(values) > 1:
        raise ValidationError("TAA_SOURCE_AMBIGUOUS", f"SAA 配置的 {column} 存在多个版本，无法确定来源。")
    return values[0] if values else None


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    # Python's float JSON encoding preserves the exact finite float roundtrip.
    return [{key: None if pd.isna(value) else value.isoformat()
             if isinstance(value, (date, datetime, pd.Timestamp)) else value
             for key, value in row.items()} for row in frame.to_dict(orient="records")]


class TacticalAllocationData:
    def __init__(self, data_dir: Path, universe_dir: Path | None = None):
        self.data_dir = Path(data_dir)
        self.universe_dir = Path(universe_dir) if universe_dir is not None else self.data_dir

    def _read(self, filename: str, *, alloc_name: str | None = None, columns: list[str] | None = None) -> tuple[pd.DataFrame, str]:
        path = self.data_dir / filename
        guard_path(path)
        if not path.is_file():
            raise NotFoundError("TAA_SAA_DATA_MISSING", f"缺少 {filename}，请先在 SAA 保存真实资产分类方案。")
        if path.stat().st_size > MAX_PARQUET_BYTES:
            raise ValidationError("TAA_SOURCE_BUDGET", "SAA 数据文件超过读取预算，请先缩小数据快照。")
        before = file_hash(path)
        try:
            schema = pq.read_schema(path)
            if "asset_alloc_name" not in schema.names:
                raise ValueError("missing allocation name")
            if pq.read_metadata(path).num_rows > 3_000_000:
                raise ValidationError("TAA_SOURCE_BUDGET", "SAA 数据文件超过解码行数预算。")
            filters = [("asset_alloc_name", "=", alloc_name)] if alloc_name is not None else None
            selected_columns = [column for column in columns if column in schema.names] if columns is not None else None
            frame = pd.read_parquet(path, filters=filters, columns=selected_columns)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise ValidationError("TAA_SAA_DATA_INVALID", f"{filename} 无法按资产配置读取。") from exc
        if file_hash(path) != before:
            raise ValidationError("TAA_SOURCE_CHANGED", "读取期间数据发生变化，请重试。")
        return frame, before

    def _configuration(self, alloc_name: str) -> tuple[pd.DataFrame, str, dict[str, Any]]:
        frame, fingerprint = self._read("asset_alloc_info.parquet", alloc_name=alloc_name)
        return self._describe_configuration(frame, fingerprint)

    @staticmethod
    def _describe_configuration(frame: pd.DataFrame, fingerprint: str) -> tuple[pd.DataFrame, str, dict[str, Any]]:
        required = {"asset_name", "etf_code", "etf_name", "etf_weight"}
        if frame.empty:
            raise NotFoundError("TAA_SAA_NOT_FOUND", "未找到所选 SAA 方案，请重新选择。")
        if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
            raise ValidationError("TAA_SAA_SCHEMA", "SAA 分类缺少资产、产品或类内权重，请先完善 SAA。")
        if frame.duplicated(["asset_name", "etf_code"]).any():
            raise ValidationError("TAA_SAA_DUPLICATE", "同一资产类别内存在重复产品，不能重复计权。")
        names = frame["asset_name"].unique().tolist()
        if not 1 <= len(names) <= MAX_ASSETS or any(not isinstance(x, str) or not x.strip() for x in names):
            raise ValidationError("TAA_ASSET_BUDGET", "SAA 须包含 1 至 30 个有效资产类别。")
        lineage = {"source_file": "asset_alloc_info.parquet", "file_hash": fingerprint,
                   "config_hash": digest_json(_records(frame.sort_values(["asset_name", "etf_code"]))),
                   "created_at": _one(frame, "creat_time"), "as_of": _one(frame, "as_of"),
                   "universe_snapshot_id": _one(frame, "universe_snapshot_id"),
                   "data_release_id": _one(frame, "data_release_id"), "run_mode": _one(frame, "run_mode")}
        return frame, fingerprint, lineage

    def catalog(self) -> dict[str, Any]:
        path = self.data_dir / "asset_alloc_info.parquet"
        guard_path(path)
        if not path.exists():
            return {"allocations": [], "warnings": ["请先在 SAA 保存资产分类方案，再选择一组战略权重。"]}
        frame, fingerprint = self._read("asset_alloc_info.parquet")
        nav = None
        if (self.data_dir / "asset_nv.parquet").exists():
            nav, _ = self._read("asset_nv.parquet", columns=["asset_alloc_name", "asset_name", "as_of", "date"])
        allocations = []
        for raw_name in sorted(frame["asset_alloc_name"].dropna().astype(str).unique()):
            try:
                selected, _, lineage = self._describe_configuration(frame.loc[frame["asset_alloc_name"] == raw_name], fingerprint)
                coverage = None
                if nav is not None and "date" in nav.columns:
                    dates = nav.loc[nav["asset_alloc_name"] == raw_name]
                    if "as_of" in dates.columns:
                        dates = dates.loc[dates["as_of"].fillna("").astype(str) == (lineage["as_of"] or "")]
                    axis = pd.to_datetime(dates["date"], errors="coerce").dropna()
                    if not axis.empty:
                        coverage = {"start_date": axis.min().strftime("%Y-%m-%d"),
                                    "end_date": axis.max().strftime("%Y-%m-%d")}
                allocations.append({"alloc_name": raw_name,
                                    "assets": [{"id": str(x), "name": str(x)} for x in selected["asset_name"].unique()],
                                    "coverage": coverage,
                                    **{key: lineage[key] for key in ("as_of", "universe_snapshot_id", "data_release_id")}})
            except ValidationError as exc:
                allocations.append({"alloc_name": raw_name, "assets": [], "unavailable_reason": exc.message})
        return {"allocations": allocations}

    def _nav_version(self, alloc_name: str, variant: str | None) -> tuple[pd.DataFrame, str, str]:
        frame, file_fingerprint = self._read("asset_nv.parquet", alloc_name=alloc_name)
        if frame.empty or not {"asset_name", "date", "nv"}.issubset(frame.columns):
            raise ValidationError("TAA_NAV_MISSING", "所选 SAA 没有可读取的真实类别净值。")
        if "as_of" in frame:
            frame = frame.loc[frame["as_of"].fillna("").astype(str) == (variant or "")]
        elif variant:
            raise ValidationError("TAA_NAV_VERSION", "类别净值缺少 SAA 声明的研究日版本。")
        if frame.empty:
            raise ValidationError("TAA_NAV_VERSION", "未找到与基线相同研究日版本的类别净值。")
        if len(frame) > MAX_OBSERVATIONS * MAX_ASSETS:
            raise ValidationError("TAA_DATA_BUDGET", "SAA 源净值超过单次读取预算。")
        fingerprint = digest_json(_records(frame.sort_values(["asset_name", "date"])))
        return frame, file_fingerprint, fingerprint

    def _universe(self, identifier: str | None, products: list[dict[str, Any]]) -> tuple[dict | None, list[str]]:
        if not identifier:
            return None, ["SAA 未绑定可投资域快照；可研究，应用到产品配置前须在 SAA 补齐。"]
        path = self.universe_dir / UNIVERSE_SNAPSHOT_STORE
        guard_path(path)
        if not path.is_file():
            return None, ["SAA 引用的可投资域快照文件不存在，暂不能应用到产品配置。"]
        if path.stat().st_size > 32_000_000:
            raise ValidationError("TAA_UNIVERSE_BUDGET", "可投资域文件过大，无法安全读取。")
        # Atomic store reads require no directory creation or lock-file mutation.
        items = AtomicProductPoolStore(path).read_unlocked()["universe_snapshots"]
        snapshot = next((item for item in items if item.get("id") == identifier), None)
        if snapshot is None:
            return None, ["SAA 引用的可投资域版本已不可读，暂不能应用到产品配置。"]
        try:
            membership = InvestableUniverseMembership(SimpleNamespace(get=lambda key: snapshot)).validate(identifier, products)
        except ProductPoolError as exc:
            return {"id": identifier, "snapshot_hash": digest_json(snapshot)}, [exc.message]
        return {**membership.reference, "snapshot_hash": digest_json(snapshot),
                "_resolved_products": [{"kind": member["kind"], "product_id": member["product_id"]}
                                       for member in membership.members]}, []

    def validate_application(self, baseline: dict[str, Any]) -> dict[str, Any]:
        """Recheck the exact immutable universe and current membership before use.

        A stored eligibility flag is a display hint, never application authority.
        Reading and membership validation share one atomic snapshot, without
        creating directories or lock files during this check.
        """
        expected = baseline.get("lineage", {}).get("universe") or {}
        identifier = baseline.get("universe_snapshot_id")
        if not identifier or not expected.get("snapshot_hash") or expected.get("id") != identifier:
            raise ValidationError("TAA_APPLICATION_UNIVERSE", "基线缺少可核验的可投资域版本，请在 SAA 补齐后重新保存基线。")
        products = [product for asset in baseline.get("assets", []) for product in asset.get("products", [])]
        if not products:
            raise ValidationError("TAA_APPLICATION_PRODUCTS", "基线没有可应用的类内产品映射。")
        current, reasons = self._universe(identifier, products)
        if current is None:
            raise ValidationError("TAA_APPLICATION_UNIVERSE", "；".join(reasons))
        if current.get("snapshot_hash") != expected["snapshot_hash"]:
            raise ValidationError("TAA_UNIVERSE_CHANGED", "可投资域内容已改变，与基线冻结版本不一致；请重新检查 SAA 并保存新基线。")
        if reasons:
            raise ValidationError("TAA_APPLICATION_MEMBERSHIP", "；".join(reasons))
        current.pop("_resolved_products", None)
        return current

    def create_baseline(self, payload: dict[str, Any]) -> dict[str, Any]:
        alloc_name = str(payload.get("alloc_name") or "").strip()
        name = str(payload.get("name") or "").strip()
        if not alloc_name or not name or len(name) > 120:
            raise ValidationError("TAA_BASELINE_NAME", "请选择 SAA 方案并填写 1 至 120 字的基线名称。")
        as_of = _date(payload.get("as_of"), "基线研究日")
        frame, _, lineage = self._configuration(alloc_name)
        names = frame["asset_name"].unique().tolist()
        weights, constraints = payload.get("weights"), payload.get("constraints") or {}
        if not isinstance(weights, dict) or set(weights) != set(names):
            raise ValidationError("TAA_BASELINE_ASSETS", "战略权重必须逐一对应所选 SAA 的全部资产类别。")
        if not isinstance(constraints, dict) or set(constraints) - set(names):
            raise ValidationError("TAA_CONSTRAINT_ASSETS", "偏离约束含有未知资产类别。")
        if abs(sum(_number(weights[x], x) for x in names) - 1.0) > 1e-8:
            raise ValidationError("TAA_BASELINE_TOTAL", "战略权重合计须为 100%，不会自动归一化。")
        groups = payload.get("group_limits") or []
        if not isinstance(groups, list) or len(groups) > 30:
            raise ValidationError("TAA_GROUP_BUDGET", "最多设置 30 个分组约束。")
        group_limits, group_ids = [], set()
        for group in groups:
            if not isinstance(group, dict):
                raise ValidationError("TAA_GROUP_INVALID", "分组约束格式无效。")
            identifier = group.get("id")
            members = group.get("assets")
            if (not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 120
                    or identifier in group_ids or not isinstance(members, list) or not members
                    or any(not isinstance(member, str) for member in members)
                    or len(set(members)) != len(members) or set(members) - set(names)):
                raise ValidationError("TAA_GROUP_INVALID", "分组名称和成员须唯一，且只能包含当前 SAA 资产。")
            lo, hi = _number(group.get("lo", 0.0), "分组下限"), _number(group.get("hi", 1.0), "分组上限")
            total = sum(float(weights[member]) for member in members)
            if lo > hi or total < lo - 1e-8 or total > hi + 1e-8:
                raise ValidationError("TAA_GROUP_BASELINE", f"战略权重不满足分组「{identifier}」的上下限。")
            group_ids.add(identifier)
            group_limits.append({"id": identifier, "assets": members, "lo": lo, "hi": hi})
        assets, products = [], []
        for asset in names:
            bound = constraints.get(asset, {})
            if not isinstance(bound, dict):
                raise ValidationError("TAA_CONSTRAINT_INVALID", "每类偏离约束须为对象。")
            lo = _number(bound.get("min_weight", 0.0), "权重下限")
            hi = _number(bound.get("max_weight", 1.0), "权重上限")
            tilt = _number(bound.get("max_abs_tilt", 0.1), "最大偏离")
            base = float(weights[asset])
            if not lo <= base <= hi:
                raise ValidationError("TAA_BASELINE_CONSTRAINT", f"{asset} 的战略权重不在上下限内。")
            constituents = []
            for row in frame.loc[frame["asset_name"] == asset].to_dict(orient="records"):
                # Percent-to-decimal conversion is input unit normalization.
                weight = _number(row["etf_weight"], "类内产品权重", maximum=100.0) / 100.0
                item = {"product_id": str(row["etf_code"]), "name": str(row["etf_name"]), "kind": "etf", "weight": weight}
                constituents.append(item)
                products.append(item)
            if abs(sum(item["weight"] for item in constituents) - 1.0) > 1e-6:
                raise ValidationError("TAA_PRODUCT_TOTAL", f"{asset} 的类内产品权重须合计 100%；杠杆配置暂不支持。")
            assets.append({"id": asset, "name": asset, "base_weight": base,
                           "min_weight": lo, "max_weight": hi, "max_abs_tilt": tilt, "products": constituents})
        universe, apply_reasons = self._universe(lineage["universe_snapshot_id"], products)
        if universe is not None:
            resolved_products = universe.pop("_resolved_products", [])
            if not apply_reasons:
                # Only the shared membership resolver may resolve an exact ID
                # or unambiguous code alias; names never infer product identity.
                for product, member in zip(products, resolved_products, strict=True):
                    product["source_product_id"] = product["product_id"]
                    product["kind"] = member["kind"]
                    product["product_id"] = member["product_id"]
        lineage["universe"] = universe
        _, nav_file_hash, nav_hash = self._nav_version(alloc_name, lineage["as_of"])
        lineage.update(nav_file_hash=nav_file_hash, nav_hash=nav_hash)
        reasons = ["SAA 分类净值不保留完整历史修订版本；此基线用于研究，不构成历史 PIT 认证。"]
        if not lineage["as_of"]:
            reasons.append("资产类别由全历史口径构建，历史回放含事后产品选择风险。")
        if lineage["created_at"] and lineage["created_at"][:10] > as_of:
            reasons.append("原 SAA 分类在研究日之后创建，当时并不存在。")
        if lineage["as_of"] and lineage["as_of"][:10] > as_of:
            reasons.append("SAA 分类使用了基线研究日之后的数据，不能还原当时的资产选择。")
        if universe and universe.get("research_date") and str(universe["research_date"])[:10] > as_of:
            reasons.append("可投资域的研究日在基线研究日之后，含事后产品筛选。")
        return {"name": name, "alloc_name": alloc_name, "as_of": as_of, "assets": assets, "group_limits": group_limits,
                "universe_snapshot_id": lineage["universe_snapshot_id"], "data_release_id": lineage["data_release_id"],
                "lineage": lineage, "pit": {"status": "research_only", "reasons": reasons},
                "apply_eligible": not apply_reasons, "apply_reasons": apply_reasons}

    def load_data(self, baseline: dict[str, Any], start_date: str, end_date: str, as_of: str) -> dict[str, Any]:
        start, end, cutoff = _date(start_date, "开始日期"), _date(end_date, "结束日期"), _date(as_of, "研究日")
        if start >= end or end > cutoff:
            raise ValidationError("TAA_DATA_DATES", "开始日期须早于结束日期，结束日期不得晚于研究日。")
        alloc_name = baseline["alloc_name"]
        _, _, config = self._configuration(alloc_name)
        if config["config_hash"] != baseline.get("lineage", {}).get("config_hash"):
            raise ValidationError("TAA_SAA_CHANGED", "SAA 分类来源已变更，请重新保存基线；历史决策仍保持原版本。")
        # Pick and verify the exact NAV version frozen with the SAA baseline.
        variant = config.get("as_of") or ""
        frame, source_file_hash, nav_hash = self._nav_version(alloc_name, variant)
        if nav_hash != baseline.get("lineage", {}).get("nav_hash"):
            raise ValidationError("TAA_NAV_CHANGED", "SAA 类别净值已变更，请重新保存基线；旧决策可读取冻结结果。")
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if frame["date"].isna().any():
            raise ValidationError("TAA_NAV_DATE", "类别净值含无效日期，无法对齐。")
        if not pd.api.types.is_datetime64_any_dtype(frame["date"]) or frame["date"].dt.tz is not None:
            raise ValidationError("TAA_NAV_DATE", "类别净值须使用无时区的日频观察日期。")
        if not (frame["date"] == frame["date"].dt.normalize()).all():
            raise ValidationError("TAA_NAV_DATE", "类别净值含盘中时间，不能将同日多条数据解释为多个日频收益期。")
        names = [item["id"] for item in baseline["assets"]]
        frame = frame.loc[frame["asset_name"].isin(names) & (frame["date"] >= start) & (frame["date"] <= end)]
        if frame.duplicated(["asset_name", "date"]).any():
            raise ValidationError("TAA_NAV_DUPLICATE", "同一资产同一日期存在多条净值；请先解决重复数据。")
        if len(frame) > MAX_OBSERVATIONS * MAX_ASSETS:
            raise ValidationError("TAA_DATA_BUDGET", "单次最多读取 10000 个观察日和 30 类资产。")
        has_availability = "available_at" in frame
        frame["_available"] = pd.to_datetime(frame["available_at"], errors="coerce") if has_availability else pd.NaT
        if has_availability:
            provided = frame["available_at"].notna() & (frame["available_at"].astype(str).str.strip() != "")
            if (provided & frame["_available"].isna()).any():
                raise ValidationError("TAA_KNOWLEDGE_INVALID", "净值可得时间含无效日期，请修正后再研究。")
        if (frame["_available"].notna() & (frame["_available"] < frame["date"])).any():
            raise ValidationError("TAA_KNOWLEDGE_DATE", "净值的可得时间早于观察日，请先修正数据时点。")
        missing_availability = int(frame["_available"].isna().sum())
        future = frame["_available"].notna() & (frame["_available"] > pd.Timestamp(cutoff) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1))
        excluded_future = int(future.sum())
        frame = frame.loc[~future]
        if frame.empty:
            raise ValidationError("TAA_DATA_NOT_AVAILABLE", "研究日之前没有已可得的类别净值。")
        wide = frame.pivot(index="date", columns="asset_name", values="nv").sort_index().reindex(columns=names)
        # Strict intersection is explicit: no ffill, zero return, or hidden reweight.
        common = wide.notna().all(axis=1)
        excluded_incomplete = int((~common).sum())
        wide = wide.loc[common]
        if len(wide) < 3:
            raise ValidationError("TAA_DATA_SHORT", "至少需要 3 个所有资产都有净值的共同观察日。")
        if len(wide) > MAX_OBSERVATIONS:
            raise ValidationError("TAA_DATA_BUDGET", "单次最多读取 10000 个观察日，请缩短回测区间。")
        if not all(pd.api.types.is_numeric_dtype(wide[name]) for name in names):
            raise ValidationError("TAA_NAV_TYPE", "类别净值必须为数值，不能隐式解释文本或类别编码。")
        # Decode and align once; the existing kernel requires one writable C input.
        values = np.ascontiguousarray(wide.to_numpy(dtype=np.float64))
        execution = strategy_execution_audit()
        returns, status = nav_to_returns_kernel(values)
        if status != 0:
            raise ValidationError("TAA_NAV_INVALID", "类别净值存在缺失、非有限或非正数，不能计算收益。")
        available_frame = frame.pivot(index="date", columns="asset_name", values="_available").reindex(index=wide.index, columns=names)
        # Datetime parsing/epoch encoding belongs to the I/O boundary. -1 is unknown.
        dates_raw = available_frame.to_numpy(dtype="datetime64[ns]")
        available_nav = np.ascontiguousarray(dates_raw.astype("datetime64[D]").astype(np.int64))
        available_nav[np.isnat(dates_raw)] = -1
        available_nav.flags.writeable = False
        from .numeric import returns_availability_kernel
        available_at = returns_availability_kernel(available_nav)
        returns.flags.writeable = False
        available_at.flags.writeable = False
        dates = wide.index.strftime("%Y-%m-%d").tolist()
        reasons = list(baseline.get("pit", {}).get("reasons", []))
        if missing_availability:
            reasons.append(f"{missing_availability} 条净值缺少知识可得时间；依赖这些样本的信号不能按 PIT 交易。")
        if config.get("created_at") and config["created_at"][:10] > start:
            reasons.append("当前 SAA 分类在回测开始之后创建，历史结果包含事后基线选择。")
        if config.get("as_of") and config["as_of"][:10] > start:
            reasons.append("SAA 分类使用了回测开始之后的数据，分类选择尚未通过历史 PIT 验证。")
        lineage = {"alloc_name": alloc_name, "config_hash": config["config_hash"],
                   "source_file": "asset_nv.parquet", "file_hash": source_file_hash,
                   "series_as_of": variant or None, "requested_as_of": cutoff,
                   "alignment": "strict_intersection", "excluded_incomplete_dates": excluded_incomplete,
                   "knowledge_time_granularity": "day", "price_basis": "saved_class_nav",
                   "intraday_execution_verified": False,
                   "excluded_not_yet_available_rows": excluded_future, "missing_availability_rows": missing_availability,
                   "observation_count": len(dates), "start_date": dates[0], "end_date": dates[-1]}
        digest = hashlib.sha256()
        digest.update(digest_json({"lineage": lineage, "assets": names, "dates": dates}).encode())
        digest.update(memoryview(returns).cast("B"))
        digest.update(memoryview(available_at).cast("B"))
        return {"dates": dates[1:], "period_starts": dates[:-1], "returns": returns,
                "available_at": available_at, "lineage": lineage, "reasons": reasons,
                "pit": {"status": "research_only", "reasons": reasons},
                "source_hash": digest.hexdigest(), "execution": execution}
