"""Orchestrates versioned research, immutable evidence and existing workflow adapters."""
from __future__ import annotations

import copy
import hashlib
import json
import threading
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from backend.custom_indicators.errors import IndicatorDomainError, ValidationError, ConflictError
from backend.custom_indicators.repository import AtomicJsonStore, utc_now
from . import numba_kernels as nk
from . import return_kernels as rk
from . import attribution_kernels as ak
from .attribution import build_contribution_analysis, validate_attribution_size
from .return_service import FactorReturnService
from .catalog import BUILTINS, CONTEXTS, DEFAULT_TARGETS, ENGINE_VERSION, MODELS, OPERATORS
from .contracts import (AttributionFields, BindingFields, DatasetFields, FactorFields,
                        PortfolioProfile, ReleaseFields, StudyFields)
from .data import (dataset_last_date, load_panel, product_catalog, records, snapshot_directory)
from .repository import ArtifactStore, VersionStore, clean

ROOT_DATA = Path(__file__).resolve().parents[2] / "data"
PERFORMANCE_KEYS = ("days", "total_return", "annualized_return", "annualized_volatility",
                    "max_drawdown", "benchmark_return", "excess_return", "turnover", "fee_sum")
STAT_KEYS = ("observations", "mean", "std", "icir", "positive_rate")


def mapping(keys, values):
    return clean(dict(zip(keys, values)))


class FactorResearchService:
    def __init__(self, workspace_data_dir: Path = ROOT_DATA, market_data_dir: Path = ROOT_DATA):
        self.workspace_data_dir = Path(workspace_data_dir)
        self.market_data_dir = Path(market_data_dir)
        root = self.workspace_data_dir / "factor_research"
        self.factors = VersionStore(root / "factors.json", "factor-definition-", BUILTINS)
        self.studies = VersionStore(root / "studies.json", "factor-study-")
        self.artifacts = ArtifactStore(root / "artifacts")
        self.events = AtomicJsonStore(root / "events.json")
        self.bindings = AtomicJsonStore(root / "bindings.json")
        self._ready = False
        self._admission = threading.BoundedSemaphore(2)
        self.return_research = FactorReturnService(root, self.artifacts, self.computing, self.execution)

    def execution(self):
        audit = nk.execution_audit()
        if any(len(kernel.nopython_signatures) != 1 or len(kernel.signatures) != 1 for kernel in rk.KERNELS + ak.KERNELS):
            raise RuntimeError("收益率内核存在未批准的计算签名")
        audit["kernel_signatures"].update({kernel.py_func.__name__: [str(s) for s in kernel.nopython_signatures]
                                            for kernel in rk.KERNELS + ak.KERNELS})
        return audit

    def warm(self):
        self._ready = False
        features = nk.warm_factor_kernels()
        returns = rk.warm_return_kernels()
        attribution = ak.warm_attribution_kernels()
        self._ready = features["complete"] and returns["complete"] and attribution["complete"]
        return {"complete": self._ready, **self.execution()}

    @contextmanager
    def computing(self):
        if not self._ready:
            raise IndicatorDomainError("FACTOR_NOT_READY", "因子计算尚未完成启动预热。", status_code=503)
        if not self._admission.acquire(blocking=False):
            raise IndicatorDomainError("FACTOR_BUSY", "已有两个研究任务正在计算，请稍后重试。", status_code=429)
        kernels = nk.KERNELS + rk.KERNELS + ak.KERNELS
        signatures = {kernel.py_func.__name__: tuple(kernel.signatures) for kernel in kernels}
        try:
            yield
            if any(tuple(kernel.signatures) != signatures[kernel.py_func.__name__] for kernel in kernels):
                raise RuntimeError("因子研究请求出现未批准的新数值签名")
        finally:
            self._admission.release()

    def catalog(self):
        root = snapshot_directory(self.market_data_dir)
        last = dataset_last_date(root)
        return {
            "factors": self.factors.list(), "models": MODELS, "contexts": CONTEXTS,
            "snapshot": {"id": root.name, "latest_date": last},
            "capabilities": [
                {"kind": "etf", "available": (root / "etf_daily_df.parquet").exists(), "basis": "复权净值"},
                {"kind": "fund", "available": (root / "fund_nav_df.parquet").exists(), "basis": "复权净值与公告日"},
                {"kind": "stock", "available": False, "reason": "待补齐股票行情、复权与财务时点数据"},
            ],
            "default_study": {
                "name": "ETF 趋势与风险三因子", "product_kind": "etf", "asset_class": "equity",
                "market": "CN", "currency": "CNY", "targets": DEFAULT_TARGETS,
                "universe_source": "manual_fixed", "start_date": "2020-01-01",
                "end_date": last or date.today().isoformat(), "oos_date": "2024-01-01",
                "benchmark": {"kind": "etf", "code": "510300.SH",
                              "label": "沪深300ETF（复权净值代理）", "return_basis": "adjusted_nav"},
                "factors": [{"factor_id": BUILTINS[i]["id"], "revision": 1, "weight": weight}
                            for i, weight in enumerate((0.5, 0.3, 0.2))],
                "normalization": "rank", "horizon": 21, "quantiles": 3, "top_n": 4, "cost_bps": 5,
                "signal_frequency": "monthly", "ic_window": 12, "ic_min_periods": 6,
                "model": "characteristic_composite", "dataset": "active_adjusted_nav",
            },
            "modules": ["characteristics", "returns"],
            "execution": self.execution(), "ready": self._ready,
        }

    def products(self, kind="etf", query="", limit=50):
        return {"items": clean(product_catalog(snapshot_directory(self.market_data_dir), kind, query, limit))}

    def save_factor(self, fields, object_id=None, revision=None):
        value = FactorFields.model_validate(fields).model_dump(mode="json")
        return self.factors.save({**value, "engine_version": ENGINE_VERSION}, object_id, revision)

    def save_study(self, fields, object_id=None, revision=None):
        value = StudyFields.model_validate(fields).model_dump(mode="json")
        if value["product_kind"] == "stock":
            raise ValidationError("STOCK_DATA_NOT_READY", "股票数据尚未具备研究条件。")
        for reference in value["factors"]:
            factor = self.factors.get(reference["factor_id"], reference["revision"])
            if value["product_kind"] not in factor["product_kinds"]:
                raise ValidationError("FACTOR_KIND_MISMATCH", f"因子 {factor['name']} 不适用于当前产品类型。")
        if value["universe_source"] != "manual_fixed":
            from backend.product_pools.repository import ProductPoolRepository
            prefix = "pool_version:"
            if not value["universe_source"].startswith(prefix):
                raise ValidationError("FACTOR_UNIVERSE_SOURCE", "请选择固定研究池或明确的已发布产品池版本。")
            version = ProductPoolRepository(self.workspace_data_dir / "product_pools.json").get_version(
                value["universe_source"][len(prefix):])
            allowed = {member["product_id"] for member in version.get("members", [])
                       if member.get("kind") == value["product_kind"] and member.get("research_status") == "approved"}
            if not set(value["targets"]).issubset(allowed):
                raise ValidationError("FACTOR_UNIVERSE_MEMBERS", "研究产品必须来自所选池版本的已批准成员。")
            value["universe_snapshot"] = version
        return self.studies.save(value, object_id, revision)

    def _study_inputs(self, study):
        definitions = [self.factors.get(ref["factor_id"], ref["revision"]) for ref in study["factors"]]
        parameters = np.ascontiguousarray([[OPERATORS[f["operator"]], f["window"], f["skip"], f["direction"]]
                                         for f in definitions], dtype=np.int64)
        weights = np.ascontiguousarray([ref["weight"] for ref in study["factors"]], dtype=np.float64)
        return definitions, parameters, weights

    def run_study(self, object_id, revision):
        with self.computing():
            study = self.studies.get(object_id, revision)
            definitions, parameters, weights = self._study_inputs(study)
            data = load_panel(self.market_data_dir, study["product_kind"], study["targets"],
                              study["start_date"], study["end_date"], benchmark=study["benchmark"],
                              signal_frequency=study.get("signal_frequency", "monthly"))
            if study["asset_class"] == "equity" and any("债券" in item["fund_type"] or "货币" in item["fund_type"] for item in data["identities"]):
                raise ValidationError("FACTOR_ASSET_CLASS_MISMATCH", "权益研究池含债券或货币产品，请拆分研究或明确选择多资产。")
            decisions = data["decisions"]
            if len(decisions) < 3:
                raise ValidationError("FACTOR_INSUFFICIENT_MONTHS", "区间至少需要三个完整研究信号截面。")
            if (len(decisions) + 1) * len(study["targets"]) * len(definitions) > 2_000_000:
                raise ValidationError("FACTOR_PANEL_LIMIT", "特征面板超过200万单元，请缩短区间、减少产品或改为周/月频。")
            dates = data["dates"]
            split = int(dates.searchsorted(pd.Timestamp(study["oos_date"])))
            start = int(dates.searchsorted(pd.Timestamp(study["start_date"])))
            all_decisions = np.ascontiguousarray(np.append(decisions, len(dates) - 1), dtype=np.int64)
            raw = nk.features_kernel(data["prices"], data["available"], data["days"], all_decisions, parameters)
            method = 0 if study["normalization"] == "rank" else 1
            normalized, scores = nk.normalize_kernel(raw, parameters[:, 3].copy(), weights, method)
            labels = nk.labels_kernel(data["prices"], decisions, study["horizon"])
            ic, rank_ic, groups, pair_counts = nk.diagnostics_kernel(
                normalized[:-1].copy(), scores[:-1].copy(), labels, study["quantiles"])
            if nk.mean_stats_kernel(rank_ic[:, -1].copy())[0] < 3:
                raise ValidationError("FACTOR_NO_VALID_LABELS", "有效截面不足三个；请扩大区间、补齐数据或调整窗口。")
            path, target_weights = nk.backtest_kernel(data["prices"], data["benchmark"],
                                                     decisions, scores[:-1].copy(), study["top_n"], float(study["cost_bps"]))
            masks = {
                "in_sample": np.ascontiguousarray(decisions + 1 + study["horizon"] < split, dtype=np.int64),
                "out_of_sample": np.ascontiguousarray(decisions >= split, dtype=np.int64),
            }
            summaries = {}
            for part, mask in masks.items():
                selection = mask.astype(bool)
                stats = []
                for f, name in enumerate([x["name"] for x in definitions] + ["组合因子"]):
                    stats.append({"name": name, "factor_id": definitions[f]["id"] if f < len(definitions) else "composite",
                                  "ic": mapping(STAT_KEYS, nk.mean_stats_kernel(ic[selection, f].copy())),
                                  "rank_ic": mapping(STAT_KEYS, nk.mean_stats_kernel(rank_ic[selection, f].copy()))})
                lo, hi = (start, split) if part == "in_sample" else (split, len(dates))
                summaries[part] = {
                    "factors": stats,
                    "performance": mapping(PERFORMANCE_KEYS, nk.performance_kernel(path, lo, hi)),
                    "group_returns": [mapping(STAT_KEYS, nk.mean_stats_kernel(groups[selection, q].copy()))
                                      for q in range(study["quantiles"])],
                    "factor_correlation": clean(nk.factor_correlation_kernel(normalized[:-1].copy(), mask)),
                }
            table = nk.score_table_kernel(raw[-1].copy(), normalized[-1].copy(), scores[-1].copy(), weights, method)
            latest = []
            for a, identity in enumerate(data["identities"]):
                latest.append({
                    **identity, "score": table[a, 0], "rank": table[a, 1], "percentile": table[a, 2],
                    "status": "ranked" if np.isfinite(table[a, 0]) else "excluded",
                    "exclusion_reason": None if np.isfinite(table[a, 0]) else "当前可用净值、滚动窗口或有效截面不足",
                    "factors": [{"factor_id": definition["id"], "revision": definition["revision"], "name": definition["name"],
                                 "raw_value": table[a, 3 + f * 3], "normalized_value": table[a, 4 + f * 3],
                                 "contribution": table[a, 5 + f * 3]}
                                for f, definition in enumerate(definitions)],
                })
            latest.sort(key=lambda row: (row["rank"] if np.isfinite(row["rank"]) else 1e9, row["code"]))
            warnings = [
                "净值口径研究模拟：未还原盘口、折溢价、容量或基金申赎确认；费用为显式假设。",
                "公告日期保守延后一交易日用于信号；当前快照无法证明每个历史数据修订版本。",
                "样本内剔除跨越样本外分界的标签；样本外区间由使用者事先指定，本次未自动寻优。",
                f"分组标签收益为{study['horizon']}日等长窗口，可能重叠，不能直接连乘成可交易净值。",
                "Top N 边界并列时同分产品全部等权纳入；分组也使用平均秩。",
            ]
            if study["universe_source"] == "manual_fixed":
                warnings.append("固定研究池存在样本选择与幸存者偏差，不能代表历史全市场可投资池。")
            if study["benchmark"]["return_basis"] == "price_index":
                warnings.append("比较基准是价格指数，未含分红；与复权净值收益口径存在差异。")
            if nk.mean_stats_kernel(path[start:, 7].copy())[1] != 1.0:
                warnings.append("有持仓路径缺失：不重选幸存产品，不再报告完整组合收益。")
            sample_codes = np.full(len(decisions), -1, dtype=np.int64)
            sample_codes[masks["in_sample"].astype(bool)] = 0
            sample_codes[masks["out_of_sample"].astype(bool)] = 1
            ic_window = study.get("ic_window", 12)
            ic_minimum = study.get("ic_min_periods", 6)
            rolling_ic = rk.rolling_ic_kernel(ic, sample_codes, ic_window, ic_minimum)
            rolling_rank_ic = rk.rolling_ic_kernel(rank_ic, sample_codes, ic_window, ic_minimum)
            rolling_rows = []
            periods = []
            for i, decision in enumerate(decisions):
                entry = decision + 1
                exit_index = entry + study["horizon"]
                periods.append({
                    "date": dates[decision].strftime("%Y-%m-%d"),
                    "entry_date": dates[entry].strftime("%Y-%m-%d") if entry < len(dates) else None,
                    "label_end": dates[exit_index].strftime("%Y-%m-%d") if exit_index < len(dates) else None,
                    "sample": "in_sample" if masks["in_sample"][i] else "out_of_sample" if masks["out_of_sample"][i] else "purged",
                    "ic": ic[i], "rank_ic": rank_ic[i], "pair_counts": pair_counts[i], "group_returns": groups[i],
                    "scores": scores[i], "raw_values": raw[i], "normalized_values": normalized[i],
                    "forward_returns": labels[i], "target_weights": target_weights[i],
                })
                if exit_index < len(dates) and sample_codes[i] >= 0:
                    rolling_rows.append({
                        "date": dates[exit_index].strftime("%Y-%m-%d"), "signal_date": periods[-1]["date"],
                        "sample": periods[-1]["sample"],
                        "ic": [mapping(STAT_KEYS, values) for values in rolling_ic[i]],
                        "rank_ic": [mapping(STAT_KEYS, values) for values in rolling_rank_ic[i]],
                    })
            curves = [{"date": dates[t].strftime("%Y-%m-%d"), "nav": path[t, 0], "benchmark_nav": path[t, 1],
                       "daily_return": path[t, 2], "turnover": path[t, 4], "cost": path[t, 5]}
                      for t in range(start, len(dates)) if np.isfinite(path[t, 7])]
            return self.artifacts.save("run", {
                "name": study["name"], "study_id": study["id"], "study_revision": study["revision"],
                "study_snapshot": study, "factor_snapshots": definitions, "data_lineage": data["lineage"],
                "engine_version": ENGINE_VERSION, "execution": self.execution(),
                "rolling_diagnostics": {"window": ic_window, "min_periods": ic_minimum,
                                        "date_basis": "label_end", "rows": rolling_rows},
                "as_of": dates[-1].strftime("%Y-%m-%d"), "status": "completed",
                "summaries": summaries, "periods": periods, "curves": curves, "latest_scores": latest,
                "data_quality": data["quality"], "warnings": warnings, "usage": "research_only",
            }, arrays={"prices": data["prices"], "available": data["available"], "days": data["days"],
                       "benchmark": data["benchmark"], "decisions": decisions, "parameters": parameters, "weights": weights})

    def list_runs(self):
        return {"items": self.artifacts.list("run")}

    def add_dataset(self, fields):
        value = DatasetFields.model_validate(fields).model_dump(mode="json")
        checksum = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        return self.artifacts.save("dataset", {**value, "checksum": checksum,
                                              "factor_names": ["MKT_RF", "SMB", "HML"],
                                              "dependent_return": "excess", "source_method": "external_import"})

    def datasets(self):
        return self.return_research.datasets()

    def run_attribution(self, fields):
        request = AttributionFields.model_validate(fields).model_dump(mode="json")
        with self.computing():
            dataset = None
            if request["model"] != "rbsa":
                dataset = self.return_research.dataset(request["dataset_id"])
                if dataset.get("market") != request["market"] or dataset.get("currency") != request["currency"]:
                    raise ValidationError("FACTOR_DATASET_MISMATCH", "因子数据集市场/币种不匹配当前产品；请提供匹配数据集。")
                if request["model"] == "ff3" and (set(dataset["factor_names"]) != {"MKT_RF", "SMB", "HML"}
                                                   or dataset["dependent_return"] != "excess"):
                    raise ValidationError("FACTOR_MODEL_COLUMNS", "FF3 需要 MKT_RF、SMB、HML 以及 RF；特征收益差额不能冒充 FF3。")
            data = load_panel(self.market_data_dir, request["product_kind"], request["targets"],
                              request["start_date"], request["end_date"],
                              indices=request["indices"] if request["model"] == "rbsa" else ())
            if request["model"] == "ff3" and any("债券" in item["fund_type"] or "货币" in item["fund_type"] for item in data["identities"]):
                raise ValidationError("FACTOR_MODEL_ASSET_MISMATCH", "FF3 权益模型不适用于纯债或货币产品，请选择匹配资产的风格代理。")
            start = int(data["dates"].searchsorted(pd.Timestamp(request["start_date"])))
            dates = data["dates"][start:]
            split = int(dates.searchsorted(pd.Timestamp(request["oos_date"])))
            returns = nk.returns_kernel(data["prices"])[start:].copy()
            if request["model"] == "rbsa":
                x = nk.returns_kernel(data["indices"])[start:].copy()
                rf = np.zeros(len(dates), dtype=np.float64)
                factor_names = request["indices"]
            else:
                frame = pd.DataFrame(dataset["rows"])
                frame["date"] = pd.to_datetime(frame["date"])
                frame = frame.set_index("date").reindex(dates)
                factor_names = ["MKT_RF", "SMB", "HML"] if request["model"] == "ff3" else dataset["factor_names"]
                x = np.ascontiguousarray(frame[factor_names].to_numpy(dtype=np.float64))
                rf = (np.ascontiguousarray(frame["RF"].to_numpy(dtype=np.float64))
                      if dataset["dependent_return"] == "excess" else np.zeros(len(dates), dtype=np.float64))
            validate_attribution_size(request, len(dates), len(request["targets"]), len(factor_names))
            if request["exposure_mode"] == "fixed":
                coefficients, stats = nk.attribution_kernel(returns, x, rf, split, 0 if request["model"] == "rbsa" else 1)
            else:
                coefficients = np.full((len(request["targets"]), len(factor_names) + 1), np.nan)
                stats = np.full((len(request["targets"]), 8), np.nan)
            attribution, coefficients, stats, attribution_arrays = build_contribution_analysis(
                request, data, dates, start, split, returns, x, rf, coefficients, stats,
                factor_names, dataset["dependent_return"] if dataset else "total")
            results = []
            for a, identity in enumerate(data["identities"]):
                code = int(stats[a, 7])
                results.append({**identity, "status": "ok" if code == 0 else "unavailable",
                                "reason": {0: None, 1: "样本不足", 2: "约束求解未收敛", 3: "解释变量秩亏或无有效变化"}[code],
                                "exposures": [{"factor": name, "value": coefficients[a, f]} for f, name in enumerate(factor_names)],
                                "daily_intercept": coefficients[a, len(factor_names)],
                                **mapping(("train_observations", "test_observations", "train_r2", "test_r2",
                                           "annualized_intercept", "train_residual_volatility", "test_residual_volatility"), stats[a, :7])})
            return self.artifacts.save("attribution", {
                "name": request["name"], "request": request, "results": results, "data_lineage": data["lineage"],
                "attribution": attribution,
                "dataset_snapshot": {k: v for k, v in dataset.items()
                                     if k not in {"rows", "diagnostics", "leg_returns", "formation_evidence", "formation_targets"}}
                                    if dataset else None,
                "factor_names": factor_names, "as_of": dates[-1].strftime("%Y-%m-%d"),
                "engine_version": ENGINE_VERSION, "execution": self.execution(),
                "dependent_return": dataset["dependent_return"] if dataset else "total",
                "warnings": ["收益归因解释已实现收益，不作为当时可得预测信号。",
                             "风格系数是收益估计，不能等同披露持仓；截距和残差不能直接等同经理能力。",
                             "指数代理可能高度相关或存在风格重叠，系数需结合样本外解释度判断。",
                             *( ["本次解释产品总收益，未减无风险收益；截距不是风险调整 Alpha。"]
                                if dataset and dataset["dependent_return"] == "total" else [] )],
            }, arrays={"returns": returns, "factors": x, "rf": rf,
                       "days": data["days"][start:].copy(), "split": np.array([split], dtype=np.int64),
                       **attribution_arrays})

    def publish(self, fields):
        value = ReleaseFields.model_validate(fields).model_dump(mode="json")
        run = self.artifacts.get(value["run_id"])
        if run["kind"] != "run" or run.get("status") != "completed":
            raise ValidationError("FACTOR_RELEASE_RUN", "只有已完成的特征研究运行可以发布。")
        if not any(row["status"] == "ranked" for row in run["latest_scores"]):
            raise ValidationError("FACTOR_RELEASE_EMPTY", "当前运行没有可发布的产品得分。")
        if value["effective_from"] < run["created_at"][:10] or value["effective_from"] < run["as_of"]:
            raise ValidationError("FACTOR_RELEASE_BACKDATE", "发布生效日不能早于研究生成日或信号日。")
        return self.artifacts.save("release", {**value, "usage": "research_only",
                                              "study_id": run["study_id"], "as_of": run["as_of"],
                                              "input_checksum": run["input_checksum"]})

    def release(self, object_id, usable=False, as_of=None):
        release = self.artifacts.get(object_id)
        if release["kind"] != "release":
            raise ValidationError("FACTOR_NOT_RELEASE", "请选择因子发布版本。")
        with self.events.locked():
            events = self.events.read_unlocked()["items"]
        retired = next((x for x in events if x["release_id"] == object_id), None)
        today = as_of or date.today().isoformat()
        state = "retired" if retired else "scheduled" if today < release["effective_from"] else "expired" if today > release["effective_to"] else "active"
        if state == "active" and (date.fromisoformat(today) - date.fromisoformat(release["as_of"])).days > 45:
            state = "stale"
        if usable and state != "active":
            raise ConflictError("FACTOR_RELEASE_INACTIVE", "发布尚未生效、已到期、信号超过45天或已停用，不能建立新的投研引用。")
        return {**release, "state": state, "retired_at": retired["created_at"] if retired else None}

    def releases(self, product_id=None):
        items = []
        for item in self.artifacts.list("release"):
            release = self.release(item["id"])
            if product_id:
                run = self.artifacts.get(release["run_id"])
                score = next((x for x in run["latest_scores"] if x["product_id"] == product_id), None)
                if score is None:
                    continue
                release["product_score"] = score
            items.append(release)
        return {"items": items}

    def retire(self, object_id):
        self.release(object_id)
        with self.events.locked():
            payload = self.events.read_unlocked()
            if not any(x["release_id"] == object_id for x in payload["items"]):
                payload["items"].append({"release_id": object_id, "created_at": utc_now(), "event": "retired"})
                self.events.write_unlocked(payload)
        return self.release(object_id)

    def get_bindings(self, context_type=None, context_id=None):
        with self.bindings.locked():
            items = self.bindings.read_unlocked()["items"]
        return {"items": [x for x in items if (not context_type or x["context_type"] == context_type)
                         and (not context_id or x["context_id"] == context_id)]}

    def bind(self, fields):
        value = BindingFields.model_validate(fields).model_dump(mode="json")
        release = self.release(value["release_id"], usable=True)
        current = {**value, "run_id": release["run_id"], "created_at": utc_now()}
        with self.bindings.locked():
            payload = self.bindings.read_unlocked()
            existing = next((x for x in payload["items"] if all(x[key] == value[key] for key in value)), None)
            if existing:
                return existing
            payload["items"].append(current)
            self.bindings.write_unlocked(payload)
        return current

    def monitor(self, object_id):
        release = self.release(object_id)
        original = self.artifacts.get(release["run_id"])
        latest_record = next((x for x in self.artifacts.list("run") if x["study_id"] == original["study_id"]), None)
        latest = self.artifacts.get(latest_record["id"]) if latest_record else original
        score_context = ("product_kind", "asset_class", "market", "currency", "targets", "universe_source", "normalization", "factors", "model", "dataset")
        comparable = all(original["study_snapshot"].get(key) == latest["study_snapshot"].get(key) for key in score_context) and original["factor_snapshots"] == latest["factor_snapshots"]
        drift = None
        if comparable:
            with self.computing():
                before = {x["product_id"]: x["score"] for x in original["latest_scores"]}
                after = {x["product_id"]: x["score"] for x in latest["latest_scores"]}
                codes = sorted(set(before) | set(after))
                old = np.ascontiguousarray([before.get(code) for code in codes], dtype=np.float64)
                new = np.ascontiguousarray([after.get(code) for code in codes], dtype=np.float64)
                drift = mapping(("score_correlation", "mean_absolute_score_change", "common_products", "coverage"), nk.drift_kernel(old, new))
        root = snapshot_directory(self.market_data_dir)
        names = [x["dataset"] for x in latest["data_lineage"]["files"]]
        changed = root.name != latest["data_lineage"]["snapshot"] or records(root, names) != latest["data_lineage"]["files"]
        return {"release": release, "latest_run_id": latest["id"], "latest_run_at": latest["created_at"],
                "data_changed_since_run": changed, "comparable": comparable, "drift": drift,
                "bindings": [x for x in self.get_bindings()["items"] if x["release_id"] == object_id],
                "note": "监控比较已保存运行；如数据已更新，请先重新运行方案。", "execution": nk.execution_audit()}

    def portfolio_profile(self, object_id, fields):
        request = PortfolioProfile.model_validate(fields).model_dump(mode="json")
        release = self.release(object_id, usable=True, as_of=request["as_of"])
        run = self.artifacts.get(release["run_id"])
        if request["as_of"] < run["as_of"]:
            raise ValidationError("FACTOR_PROFILE_LOOKAHEAD", "该发布的得分在持仓研究日尚不可得。")
        holdings = request["holdings"]
        if len({x["product_id"] for x in holdings}) != len(holdings) or not 0 < sum(x["weight"] for x in holdings) <= 1.000001:
            raise ValidationError("FACTOR_PROFILE_WEIGHTS", "持仓不能重复，权重之和须在 0–1 之间。")
        score_map = {x["product_id"]: x for x in run["latest_scores"]}
        values = np.ascontiguousarray([[score_map[h["product_id"]]["factors"][f]["normalized_value"]
                                       if h["product_id"] in score_map else np.nan
                                       for f in range(len(run["factor_snapshots"]))] for h in holdings], dtype=np.float64)
        weights = np.ascontiguousarray([h["weight"] for h in holdings], dtype=np.float64)
        with self.computing():
            profile = nk.portfolio_profile_kernel(values, weights)
            return self.artifacts.save("profile", {
                "name": "持仓因子画像", "release_id": object_id, "run_id": run["id"], "request": request,
                "factors": [{"name": factor["name"], "value": profile[f, 0], "covered_weight": profile[f, 1]}
                            for f, factor in enumerate(run["factor_snapshots"])],
                "meaning": "覆盖持仓的加权特征得分，不是回归风险暴露；未覆盖权重明确列示。",
                "execution": nk.execution_audit(),
            })

    def evaluation_plan(self, object_id):
        release = self.release(object_id, usable=True)
        run = self.artifacts.get(release["run_id"])
        return {"id": object_id, "revision": 1, "name": release["name"],
                "product_kind": run["study_snapshot"]["product_kind"], "source": "factor_release",
                "factor_run_id": run["id"]}

    def evaluation_run(self, object_id, as_of=None):
        release = self.release(object_id, usable=True)
        run = self.artifacts.get(release["run_id"])
        if as_of and as_of != run["as_of"]:
            raise ValidationError("FACTOR_FROZEN_AS_OF", f"发布是固定研究快照，得分日为 {run['as_of']}；其他日期请重新研究。")
        return {"result_id": run["id"], "as_of": run["as_of"], "data_generation": run["data_lineage"]["generation"],
                "rows": run["latest_scores"], "total": len(run["latest_scores"]),
                "ranked_count": len([x for x in run["latest_scores"] if x["status"] == "ranked"]),
                "excluded_count": len([x for x in run["latest_scores"] if x["status"] != "ranked"])}

    def evaluation_page(self, object_id, page, page_size):
        run = self.artifacts.get(object_id)
        rows = run.get("latest_scores", [])
        return {"rows": rows[(page - 1) * page_size:page * page_size], "total": len(rows)}
