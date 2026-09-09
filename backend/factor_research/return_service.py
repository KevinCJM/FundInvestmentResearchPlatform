"""Return-series domain: construction plans, immutable inputs, and datasets."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Callable

import numpy as np

from backend.custom_indicators.errors import ValidationError
from . import return_kernels as rk
from .catalog import ENGINE_VERSION
from .repository import ArtifactStore, VersionStore
from .return_contracts import FF3SourceFields, ReturnDatasetFields, ReturnPlanFields

GROUPS = ("SL", "SM", "SH", "BL", "BM", "BH")
FF3_NAMES = ["MKT_RF", "SMB", "HML"]
SUMMARY_KEYS = ("observations", "mean", "std", "mean_std_ratio", "positive_rate")
SOURCE_TEMPLATE = {
    "name": "股票时点面板名称", "source_url": "https://example.org/replace-with-real-source",
    "market": "CN", "currency": "CNY", "frequency": "daily", "units": "decimal_return",
    "construction": "请填写真实来源、历史成员与退市覆盖、复权、币种和修订口径。",
    "calendar": [], "formations": [], "returns": [], "rf": [],
}
DATASET_TEMPLATE = {
    "name": "因子收益数据集名称", "source_url": "https://example.org/replace-with-real-source",
    "market": "CN", "currency": "CNY", "frequency": "daily", "units": "decimal_return",
    "construction": "请填写真实市场覆盖、因子构造、收益口径及修订说明。",
    "factor_names": FF3_NAMES, "dependent_return": "excess", "rows": [],
}


class FactorReturnService:
    def __init__(self, root: Path, artifacts: ArtifactStore, computing: Callable, execution: Callable):
        self.plans = VersionStore(root / "return-plans.json", "factor-return-plan-")
        self.artifacts = artifacts
        self.computing = computing
        self.execution = execution

    def catalog(self):
        return {
            "methods": [
                {"id": "characteristic_spread", "name": "特征分组收益差额", "available": True,
                 "input": "已完成的特征研究及冻结净值", "output": "高分组日收益 − 低分组日收益"},
                {"id": "ff3_2x3", "name": "FF3 风格 · 2×3 双排序", "available": True,
                 "input": "上传的股票时点面板与逐日 RF", "output": "MKT_RF、SMB、HML、RF"},
                {"id": "native_stock_ff3", "name": "直接从本地股票数据构建", "available": False,
                 "reason": "本地股票行情、财务 PIT 与退市覆盖尚未验收；可先导入规范时点面板。"},
                {"id": "barra_cross_section", "name": "Barra 类横截面回归", "available": False,
                 "reason": "尚未实现完整暴露、行业约束、特异风险与协方差模型，不等同 FF3 双排序。"},
            ],
            "source_template": SOURCE_TEMPLATE, "dataset_template": DATASET_TEMPLATE,
            "source_fields": {
                "formations": "date, asset, market_cap, december_market_cap, book_equity, fiscal_year_end, announced_date, reference_member",
                "returns": "date, asset, return_value, lagged_market_cap, weight_date",
                "rf": "date, RF",
                "calendar": "从首个六月形成日开始的完整交易日历；所有日期为 YYYY-MM-DD。",
                "limits": "最多1000个资产、6000个交易日、30万条收益记录、200万日期×资产单元。",
            },
        }

    def source(self, object_id):
        value = self.artifacts.get(object_id)
        if value["kind"] != "return-source":
            raise ValidationError("FACTOR_RETURN_SOURCE_KIND", "请选择股票时点面板，而不是特征运行或收益数据集。")
        return value

    def import_source(self, fields):
        value = FF3SourceFields.model_validate(fields).model_dump(mode="json")
        checksum = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        result = self.artifacts.save("return-source", {**value, "checksum": checksum,
                                                       "schema": "ff3-formation-panel-v1"})
        return self.source_summary(result)

    @staticmethod
    def source_summary(value):
        return {key: value[key] for key in ("id", "name", "market", "currency", "source_url", "checksum")} | {
            "start_date": value["calendar"][0], "end_date": value["calendar"][-1],
            "observations": len(value["returns"]), "formation_rows": len(value["formations"]),
        }

    def sources(self):
        return {"items": [self.source_summary(self.source(row["id"]))
                          for row in self.artifacts.list("return-source")]}

    def save_plan(self, fields, object_id=None, revision=None):
        value = ReturnPlanFields.model_validate(fields).model_dump(mode="json")
        if value["method"] == "characteristic_spread":
            run = self.artifacts.get(value["source_run_id"])
            if run["kind"] != "run" or run.get("status") != "completed":
                raise ValidationError("FACTOR_RETURN_RUN_KIND", "请选择已完成的产品特征研究运行。")
            allowed = {"composite", *(row["id"] for row in run["factor_snapshots"])}
            if value["factor_key"] not in allowed:
                raise ValidationError("FACTOR_RETURN_FEATURE", "所选因子不在该历史运行中。")
            if value["quantiles"] > len(run["study_snapshot"]["targets"]):
                raise ValidationError("FACTOR_RETURN_GROUPS", "分组数不能超过研究产品数。")
        else:
            self.source(value["source_panel_id"])
        return self.plans.save(value, object_id, revision)

    def _diagnostics(self, dates, names, values):
        stats, correlation, cumulative = rk.return_diagnostics_kernel(values)
        return {
            "factors": [{"factor": name, **dict(zip(SUMMARY_KEYS, stats[f]))}
                        for f, name in enumerate(names)],
            "correlation": correlation,
            "cumulative": [{"date": day, "values": cumulative[t]} for t, day in enumerate(dates)],
            "cumulative_meaning": "日收益算术累计，不是可投资净值；中间缺失后不跨缺口累计。",
        }

    def import_dataset(self, fields):
        value = ReturnDatasetFields.model_validate(fields).model_dump(mode="json")
        value["rows"] = [{"date": row["date"], **row["values"]} for row in value["rows"]]
        values = np.ascontiguousarray([[row[name] for name in value["factor_names"]]
                                      for row in value["rows"]], dtype=np.float64)
        with self.computing():
            diagnostics = self._diagnostics([row["date"] for row in value["rows"]], value["factor_names"], values)
            checksum = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
            return self.artifacts.save("dataset", {
                **value, "checksum": checksum, "source_method": "external_import",
                "diagnostics": diagnostics, "execution": self.execution(),
                "warnings": ["导入数据的市场覆盖、历史时点与修订口径依赖提供者声明，系统不证明来源真实性。"],
            })

    def dataset(self, object_id):
        value = self.artifacts.get(object_id)
        if value["kind"] != "dataset":
            raise ValidationError("FACTOR_RETURN_DATASET_KIND", "请选择因子收益数据集。")
        # Read adapter only: never rewrites old FF3 artifacts.
        return {"factor_names": FF3_NAMES, "dependent_return": "excess",
                "source_method": "external_import", "warnings": [], **value}

    def datasets(self):
        result = []
        for record in self.artifacts.list("dataset"):
            item = self.dataset(record["id"])
            result.append({key: item.get(key) for key in (
                "id", "name", "market", "currency", "source_url", "factor_names", "dependent_return",
                "source_method", "created_at", "return_plan_id", "return_plan_revision", "checksum",
            )} | {"observations": len(item["rows"]), "start_date": item["rows"][0]["date"],
                 "end_date": item["rows"][-1]["date"]})
        return {"items": result}

    def export_csv(self, object_id):
        value = self.dataset(object_id)
        columns = ["date", *value["factor_names"]]
        if value["dependent_return"] == "excess":
            columns.append("RF")
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(value["rows"])
        return stream.getvalue()

    def run(self, object_id, revision):
        with self.computing():
            plan = self.plans.get(object_id, revision)
            if plan["method"] == "characteristic_spread":
                fields, arrays = self._spread(plan)
            else:
                fields, arrays = self._ff3(plan)
            names = fields["factor_names"]
            values = np.ascontiguousarray([[row[name] for name in names] for row in fields["rows"]], dtype=np.float64)
            if len(fields["rows"]) < 30:
                raise ValidationError("FACTOR_RETURN_SHORT", "至少需要30个收益日，请扩大构建区间。")
            diagnostics = self._diagnostics([row["date"] for row in fields["rows"]], names, values)
            if any(row["observations"] < 3 for row in diagnostics["factors"]):
                raise ValidationError("FACTOR_RETURN_EMPTY", "有效因子收益不足三个；请检查空分组和持仓数据缺失。")
            return self.artifacts.save("dataset", {
                **fields, "name": plan["name"], "frequency": "daily", "units": "decimal_return",
                "source_method": plan["method"], "return_plan_id": plan["id"],
                "return_plan_revision": plan["revision"], "plan_snapshot": plan,
                "diagnostics": diagnostics, "execution": self.execution(), "engine_version": ENGINE_VERSION,
            }, arrays=arrays)

    def _spread(self, plan):
        run = self.artifacts.get(plan["source_run_id"])
        if run["kind"] != "run":
            raise ValidationError("FACTOR_RETURN_RUN_KIND", "特征运行类型无效。")
        frozen = self.artifacts.load_arrays(run["id"])
        keys = ("prices", "days", "decisions")
        if not all(key in frozen for key in keys):
            raise ValidationError("FACTOR_RETURN_INPUT", "特征运行缺少必要冻结矩阵。")
        factor_key = plan["factor_key"]
        if factor_key == "composite":
            score_rows = [period["scores"] for period in run["periods"]]
        else:
            index = next((i for i, factor in enumerate(run["factor_snapshots"]) if factor["id"] == factor_key), None)
            if index is None:
                raise ValidationError("FACTOR_RETURN_FEATURE", "因子不属于该历史运行。")
            score_rows = [[row[index] for row in period["normalized_values"]] for period in run["periods"]]
        scores = np.ascontiguousarray(score_rows, dtype=np.float64)
        decisions, prices = frozen["decisions"], frozen["prices"]
        if scores.shape != (decisions.size, prices.shape[1]) or decisions.size == 0:
            raise ValidationError("FACTOR_RETURN_INPUT", "冻结的截面得分与交易日矩阵不一致。")
        path, targets = rk.spread_returns_kernel(prices, decisions, scores, plan["quantiles"], float(plan["cost_bps"]))
        start = int(decisions[0]) + 1
        dates = frozen["days"].astype("datetime64[D]").astype(str).tolist()
        name = plan["output_factor"]
        rows = [{"date": dates[t], name: path[t, 2]} for t in range(start, len(dates))]
        legs = [{"date": dates[t], **dict(zip(("low_return", "high_return", "gross_spread", "net_spread", "turnover", "cost"), path[t]))}
                for t in range(start, len(dates))]
        fields = {
            "factor_names": [name], "dependent_return": "total", "rows": rows, "leg_returns": legs,
            "market": run["study_snapshot"]["market"], "currency": run["study_snapshot"]["currency"],
            "source_url": "", "source_run_id": run["id"], "source_input_checksum": run["input_checksum"],
            "data_lineage": run["data_lineage"], "formation_targets": targets,
            "construction": "方向调整后的历史特征截面分组；下一交易日收盘等权进入，逐日持仓漂移，毛高分减低分收益。",
            "warnings": [*run.get("warnings", []),
                         "研究多空差额不等于可执行策略：基金可能不可做空，未计借券、融资与容量。",
                         "因子列是毛收益差额；费用仅列为扣费差额诊断。日收益不来自重叠前瞻标签。",
                         "没有无风险数据：通用归因解释产品总收益，截距不是风险调整 Alpha。"],
        }
        return fields, {**{key: frozen[key] for key in keys}, "scores": scores}

    def _ff3(self, plan):
        source = self.source(plan["source_panel_id"])
        original_fields = {key: source.get(key) for key in FF3SourceFields.model_fields}
        checksum = hashlib.sha256(json.dumps(original_fields, sort_keys=True).encode()).hexdigest()
        if checksum != source.get("checksum"):
            raise ValidationError("FACTOR_RETURN_SOURCE_CHECKSUM", "股票时点面板校验和不一致，已停止构建；请重新导入可核查的原始数据。")
        calendar = source["calendar"]
        assets = sorted({row["asset"] for row in source["returns"]} | {row["asset"] for row in source["formations"]})
        asset_index = {asset: i for i, asset in enumerate(assets)}
        day_index = {day: i for i, day in enumerate(calendar)}
        formation_dates = sorted({row["date"] for row in source["formations"]})
        formation_index = {day: i for i, day in enumerate(formation_dates)}
        returns = np.full((len(calendar), len(assets)), np.nan)
        caps = np.full(returns.shape, np.nan)
        descriptors = np.full((len(formation_dates), len(assets), 4), np.nan)
        rf = np.full(len(calendar), np.nan)
        for row in source["returns"]:
            t, a = day_index[row["date"]], asset_index[row["asset"]]
            returns[t, a] = row["return_value"] if row["return_value"] is not None else np.nan
            caps[t, a] = row["lagged_market_cap"]
        for row in source["formations"]:
            descriptors[formation_index[row["date"]], asset_index[row["asset"]]] = (
                row["market_cap"], row["december_market_cap"], row["book_equity"], float(row["reference_member"]))
        for row in source["rf"]:
            rf[day_index[row["date"]]] = row["RF"]
        formation_days = np.ascontiguousarray([day_index[day] for day in formation_dates], dtype=np.int64)
        values, counts, memberships = rk.ff3_returns_kernel(returns, caps, rf, formation_days, descriptors)
        for i, day in enumerate(formation_dates):
            if any(value == 0 for value in counts[i, :6]):
                raise ValidationError("FACTOR_FF3_EMPTY_GROUP", f"{day} 的2×3分组为空或参考池不足六个股票；请补齐样本，不能人为拆开并列值。")
        rows = [{"date": day, **dict(zip([*FF3_NAMES, "RF"], values[t, :4]))}
                for t, day in enumerate(calendar) if t > 0]
        evidence = [{"date": day, "counts": dict(zip(GROUPS, counts[i, :6])),
                     "size_break": counts[i, 6], "bm30": counts[i, 7], "bm70": counts[i, 8],
                     "memberships": [{"asset": asset, "group": GROUPS[memberships[i, a]]}
                                     for a, asset in enumerate(assets) if memberships[i, a] >= 0]}
                    for i, day in enumerate(formation_dates)]
        fields = {
            "factor_names": FF3_NAMES, "dependent_return": "excess", "rows": rows,
            "market": source["market"], "currency": source["currency"], "source_url": source["source_url"],
            "source_panel_id": source["id"], "source_checksum": source["checksum"],
            "formation_evidence": evidence,
            "leg_returns": [{"date": day, **dict(zip(GROUPS, values[t, 4:]))}
                            for t, day in enumerate(calendar) if t > 0],
            "construction": "FF3风格2×3：六月参考池规模中位数与B/M30%、70%断点；六组形成日市值加权及日后总收益漂移；市场因子使用前日市值与明确RF。",
            "warnings": [
                "自定义市场和参考池的 FF3 风格构造，不是 Kenneth French 官方数据的严格复制品。",
                "输入声明不证明历史全市场覆盖、退市完整性或修订 PIT；需核查数据提供者。",
                "六组股票固定持有至下一形成日，持仓缺失不剔除后重配；有缺失时相应因子为null。",
                "输出为毛研究因子收益，未扣交易、借券及融资费用。",
            ],
        }
        arrays = {"returns": returns, "lagged_caps": caps, "rf": rf, "descriptors": descriptors,
                  "formation_days": formation_days,
                  "days": np.ascontiguousarray(np.array(calendar, dtype="datetime64[D]").astype(np.int64))}
        return fields, arrays
