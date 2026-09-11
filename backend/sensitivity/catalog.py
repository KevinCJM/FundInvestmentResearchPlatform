"""One typed variable dictionary shared by transmission and product risk models."""
from __future__ import annotations

import copy
import csv
import io
import math
import re
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from backend.custom_indicators.errors import ValidationError
from backend.data_storage import guard_path
from backend.market_data import resolve_tushare_data_dir
from .contracts import SeriesImport
from .repository import ArtifactRepository, digest_json

UNIT_LABELS = {"return": "%", "bp": "bp", "pp": "个百分点", "points": "点"}
UNIT_CODES = {"return": 0, "bp": 1, "pp": 2, "points": 3}
FREQUENCY_LABELS = {"daily": "日频", "weekly": "周频", "monthly": "月频", "quarterly": "季频"}
TRANSFORM_CODES = {"price_return": 0, "percent_rate_change": 1, "difference": 2, "identity": 3}


def _variable(identifier, name, roles, unit, frequency, transform, source, category, basis):
    return {"id": identifier, "name": name, "roles": roles, "unit": unit,
            "frequency": frequency, "transform": transform, "source": source,
            "category": category, "basis": basis, "market": "CN", "currency": "CNY", "revision": 1}


def _index(code):
    return {"kind": "index", "ts_code": code, "source_api": "index_daily",
            "field": "close", "filename": "index_daily_df.parquet"}


def _macro(dataset, field):
    return {"kind": "macro", "dataset": dataset, "field": field,
            "filename": f"{dataset}_df.parquet"}


BUILTINS = [
    _variable("cn-equity-csi300", "沪深300价格收益", ["market"], "return", "daily", "price_return",
              _index("000300.SH"), "equity", "price_index"),
    _variable("cn-equity-csi500", "中证500价格收益", ["market"], "return", "daily", "price_return",
              _index("000905.SH"), "equity", "price_index"),
    _variable("cn-gold-etf", "黄金ETF净值收益（518880）", ["market"], "return", "daily", "price_return",
              {"kind": "etf", "ts_code": "518880.SH", "field": "adj_nav", "filename": "etf_daily_df.parquet"},
              "other", "adjusted_nav_proxy"),
    _variable("cn-gov-bond-etf", "国债ETF净值收益（511010）", ["market"], "return", "daily", "price_return",
              {"kind": "etf", "ts_code": "511010.SH", "field": "adj_nav", "filename": "etf_daily_df.parquet"},
              "credit", "adjusted_nav_proxy"),
    _variable("cn-shibor-1w", "一周Shibor变动", ["macro", "market"], "bp", "daily", "percent_rate_change",
              _macro("macro_shibor", "1w"), "policy", "percent_rate_level"),
    _variable("cn-lpr-1y", "一年期LPR变动", ["driver", "macro"], "bp", "monthly", "percent_rate_change",
              _macro("macro_lpr", "1y"), "policy", "percent_rate_level"),
    _variable("cn-cpi-yoy", "CPI同比增速变化", ["macro"], "pp", "monthly", "difference",
              _macro("macro_cn_cpi", "nt_yoy"), "inflation", "year_over_year_rate_change"),
    _variable("cn-ppi-yoy", "PPI同比增速变化", ["macro"], "pp", "monthly", "difference",
              _macro("macro_cn_ppi", "ppi_yoy"), "inflation", "year_over_year_rate_change"),
    _variable("cn-pmi", "制造业PMI变化", ["macro"], "points", "monthly", "difference",
              _macro("macro_cn_pmi", "pmi010000"), "activity", "index_level_change"),
    _variable("cn-m2-yoy", "M2同比增速变化", ["macro"], "pp", "monthly", "difference",
              _macro("macro_cn_money", "m2_yoy"), "credit", "year_over_year_rate_change"),
    _variable("cn-gdp-yoy", "GDP同比增速变化", ["macro"], "pp", "quarterly", "difference",
              _macro("macro_cn_gdp", "gdp_yoy"), "activity", "year_over_year_rate_change"),
    _variable("cn-gov-yield-bp", "人民币债券到期收益率平行变动", ["market"], "bp", "daily", "percent_rate_change",
              None, "policy", "parallel_yield_change_cashflow_valuation"),
]
BUILTIN_BY_ID = {item["id"]: item for item in BUILTINS}
EVENT_TEMPLATES = [
    {"id": "energy", "name": "能源成本冲击", "description": "先指定量化的能源驱动，再由已发布模型计算宏观影响。"},
    {"id": "policy", "name": "政策利率变化", "description": "指定政策驱动，不预先写死股市、通胀或长端利率方向。"},
    {"id": "credit", "name": "信用条件收紧", "description": "使用已研究的信用驱动；不能由事件名称直接生成损失。"},
    {"id": "custom", "name": "自定义事件", "description": "填写叙事，并明确驱动变量及变化路径。"},
]


def variable_contract(variable):
    fields = ("id", "revision", "unit", "frequency", "transform", "source", "basis", "market", "currency")
    return {key: variable[key] for key in fields}


def decorate(variable):
    result = copy.deepcopy(variable)
    result["unit_label"] = UNIT_LABELS[result["unit"]]
    result["reference_move"] = "+1%" if result["unit"] == "return" else "+100bp" if result["unit"] == "bp" else f'+1{result["unit_label"]}'
    result["contract_hash"] = digest_json(variable_contract(result))
    return result


class VariableRegistry:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.imports = ArtifactRepository(self.data_dir / "model_variables")

    def get(self, identifier):
        guard_path(self.data_dir)
        if identifier in BUILTIN_BY_ID:
            return decorate(BUILTIN_BY_ID[identifier])
        item = self.imports.get(identifier, "series")
        return decorate({**item["variable"], "id": item["id"], "source": {
            "kind": "import", "artifact_id": item["id"], "content_hash": item["content_hash"],
        }})

    def list(self):
        guard_path(self.data_dir)
        items = [self.get(item["id"]) for item in BUILTINS]
        items.extend(self.get(item["id"]) for item in self.imports.list("series"))
        root = resolve_tushare_data_dir(self.data_dir)
        for item in items:
            source = item["source"]
            if source is None:
                item["availability"] = {"available": False, "reason": "没有历史时序；可用于固定现金流单期估值，回归研究需先导入相应时序。"}
            elif source["kind"] == "import":
                item["availability"] = {"available": True, "reason": "自有时序；不认证为官方或完整历史PIT数据。"}
            else:
                path = root / source["filename"]
                available = False
                reason = f'缺少 {source["filename"]}，请在数据同步中补齐。'
                if path.is_file():
                    try:
                        available = source["field"] in pq.ParquetFile(path).schema_arrow.names
                        reason = "源文件和字段可用；计算时检查对象、区间及公告时间。" if available else f'源文件缺少字段 {source["field"]}。'
                    except (OSError, ValueError):
                        reason = "源文件暂不可读，请检查数据质量。"
                item["availability"] = {"available": available, "reason": reason}
        return items

    def import_series(self, raw):
        request = SeriesImport.model_validate(raw)
        guard_path(self.data_dir, write=True)
        reader = csv.DictReader(io.StringIO(request.csv_text.lstrip("\ufeff")))
        if not reader.fieldnames or set(reader.fieldnames) not in ({"date", "value"}, {"date", "value", "available_at"}):
            raise ValidationError("SERIES_CSV_HEADERS", "CSV 列名须为 date,value，可增加 available_at；日期使用 YYYY-MM-DD。")
        rows = []
        seen = set()
        today = datetime.now(timezone.utc).date()
        finite_count = 0
        for line, row in enumerate(reader, start=2):
            if len(rows) >= 20_000 or None in row:
                raise ValidationError("SERIES_CSV_SIZE", "CSV 最多 20,000 行，每行列数须一致。")
            try:
                text = (row.get("date") or "").strip()
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
                    raise ValueError("date")
                observed = date.fromisoformat(text)
                known = date.fromisoformat(row["available_at"].strip()) if row.get("available_at", "").strip() else today
                if observed in seen or known < observed or observed > today or known > today:
                    raise ValueError("dates")
                number = float(row["value"]) if row.get("value", "").strip() else np.nan
                if row.get("value", "").strip() and not math.isfinite(number):
                    raise ValueError("number")
                if math.isfinite(number) and request.transform == "price_return" and number <= 0:
                    raise ValueError("price")
                if math.isfinite(number) and request.transform == "identity" and request.unit == "return" and number <= -1:
                    raise ValueError("return")
            except (TypeError, ValueError) as exc:
                raise ValidationError("SERIES_CSV_ROW", f"第 {line} 行无效：检查日期、重复日期、可得日及有限数值；空值请留空。") from exc
            finite_count += int(math.isfinite(number))
            seen.add(observed)
            rows.append((observed, known, number))
        if finite_count < 2:
            raise ValidationError("SERIES_CSV_EMPTY", "至少需要两个有效观察值。")
        rows.sort(key=lambda row: row[0])
        days = np.array([row[0].isoformat() for row in rows], dtype="datetime64[D]").astype(np.int64)
        available = np.array([row[1].isoformat() for row in rows], dtype="datetime64[D]").astype(np.int64)
        values = np.array([row[2] for row in rows], dtype=np.float64)
        variable = {"name": request.name, "roles": request.roles, "unit": request.unit,
                    "frequency": request.frequency, "transform": request.transform,
                    "market": "CN", "currency": "CNY", "revision": 1, "category": request.category,
                    "basis": "user_declared_series", "source_label": request.source_label}
        item = self.imports.save("series", {"name": request.name, "variable": variable,
            "source_label": request.source_label, "observations": len(rows),
            "first_date": rows[0][0].isoformat(), "last_date": rows[-1][0].isoformat(),
            "pit_verified": False, "availability_basis": "user_declared_or_import_time"},
            {"days": days, "available": available, "values": values})
        return self.get(item["id"])
