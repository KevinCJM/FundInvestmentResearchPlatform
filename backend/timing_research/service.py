"""Bounded research jobs; Python only orchestrates I/O and fixed NJIT plans."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import logging
import math
from pathlib import Path
import threading
import numpy as np

from backend.custom_indicators.errors import ValidationError, NotFoundError, ConflictError
from .catalog import build_catalog, templates
from .contracts import Definition, RunRequest
from .data import load_etf_bars
from .graph import GraphRuntime
from .repository import TimingRepository
from . import numeric
from .panel import warm_panel_kernels
from .learning import warm_learning_kernels
from .training_runtime import prepare_candidates, fit_and_route
from .context import bind_baskets, validate_baskets

LOG = logging.getLogger(__name__)
RESTRICTIONS = [
    "固定规则研究，未训练或自动选参；样本外区间必须在查看结果之前预留。重复试参后不再是未触碰样本外。",
    "分段复核是固定算法的独立空仓起点检验，不宣称已完成训练型 walk-forward。",
    "仅境内日线 ETF：收盘信号、次日开盘、T+1；日内双触碰按止损优先。未模拟停牌成交队列、涨跌停与冲击容量。",
    "复权价格是研究单位，不是历史可下单报价；现有快照不保证历史修订版本，结果不得直接作为实盘或正式 PIT 发布依据。",
    "每个区间允许执行前一交易日已知信号；区间结束不强制平仓。持有基准不扣策略交易费用。",
    "年度/月度使用连续账户，已平仓交易按退出日归组，可能跨越区间；信号未来收益仅为检验标签，不参与入场。",
]


def _number(value):
    number = float(value)
    return number if math.isfinite(number) else None


def _metrics(values):
    return {key: _number(value) for key, value in zip(numeric.SUMMARY_COLUMNS, values)}


def _day(value):
    return str(np.datetime64(int(value), "D"))


def _index(dates, value):
    # Date-axis selection is an I/O boundary, not a numerical calculation.
    return int(np.searchsorted(dates, np.datetime64(str(value), "D").astype(np.int64)))


def _groups(dates, start, end, width):
    """Calendar labels select basic intervals; no gathered numerical arrays."""
    beginning, current = start, None
    for index in range(start, end):
        label = _day(dates[index])[:width]
        if current is not None and label != current:
            yield current, beginning, index
            beginning = index
        current = label
    if current is not None:
        yield current, beginning, end


class TimingResearchService:
    def __init__(self, base: Path, *, market_data_dir: Path | None = None, indicator_service=None, loader=load_etf_bars):
        self.repository = TimingRepository(base)
        self.market_data_dir = market_data_dir or base
        self.indicator_service = indicator_service
        self.loader = loader
        self.graph = GraphRuntime(build_catalog(indicator_service))
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="timing-research")
        self._slots = threading.BoundedSemaphore(4)
        self._jobs = OrderedDict()
        self._lock = threading.RLock()
        self._warm = False
        self._training_plans = OrderedDict()

    def warm(self):
        status = numeric.warm_timing_kernels()
        warm_panel_kernels()
        warm_learning_kernels()
        for item in templates():
            self._prepare(Definition.model_validate(item["definition"]))
        saved = self.repository.list_definitions()
        for item in saved:
            self._prepare(Definition.model_validate({key: item[key] for key in Definition.model_fields if key in item}))
        self._warm = True
        return {"complete": True, "numeric": status, "templates": len(templates()), "saved_definitions": len(saved)}

    def close(self):
        self._executor.shutdown(wait=True, cancel_futures=False)

    def catalog(self):
        self.graph.catalog = build_catalog(self.indicator_service)
        operators = [{key: value for key, value in meta.items() if not key.startswith("_")} for meta in self.graph.catalog.values()]
        return {"operators": operators, "templates": templates(), "limits": {"products": 12, "bars": 12000, "steps": 128, "math_nodes": 512, "queued_jobs": 4, "training_candidates": 108, "basket_members": 12}}

    def prepare(self, definition: Definition):
        self.catalog()
        prepared = self._prepare(definition)
        return {"compile_token": prepared.digest, "definition_hash": prepared.digest, "execution": self.graph.audit(prepared)}

    def _prepare(self, definition):
        prepared = self.graph.prepare(definition)
        with self._lock:
            if definition.training and prepared.digest not in self._training_plans:
                candidates = prepare_candidates(definition, self.graph)
                self._training_plans[prepared.digest] = candidates
                while len(self._training_plans) > 16:
                    self._training_plans.popitem(last=False)
                # Preparing up to 108 variants must not evict the user's token.
                prepared = self.graph.prepare(definition)
        return prepared

    def save_definition(self, definition, identifier=None, revision=None):
        self.prepare(definition)
        return self.repository.save_definition(definition.model_dump(mode="json"), identifier, revision)

    def submit(self, request: RunRequest):
        if not self._warm:
            raise ConflictError("TIMING_NOT_READY", "择时计算尚未完成启动预热，请稍后重试。")
        prepared = self.graph.get(request.definition, request.compile_token)
        validate_baskets(request, prepared)
        with self._lock:
            candidates = self._training_plans.get(prepared.digest, ())
        if prepared.definition.training and not candidates:
            raise ConflictError("TIMING_PREPARE_REQUIRED", "训练计划已过期，请重新准备算法。")
        if not self._slots.acquire(blocking=False):
            raise ConflictError("TIMING_QUEUE_FULL", "已有 4 个研究任务等待处理，请完成后再试。")
        try:
            snapshot = request.model_dump(mode="json", exclude={"compile_token"})
            identifier = self.repository.create_run(prepared.definition.model_dump(mode="json"), snapshot)
            job = {"id": identifier, "status": "queued", "progress": 0, "total": len(request.targets)}
            with self._lock:
                self._jobs[identifier] = job
                # Completed job metadata is bounded; durable runs remain available.
                for old in list(self._jobs):
                    if len(self._jobs) <= 128:
                        break
                    if self._jobs[old]["status"] in {"completed", "failed"}:
                        del self._jobs[old]
            self._executor.submit(self._work, identifier, request.model_copy(deep=True), prepared, candidates)
            return dict(job)
        except Exception:
            self._slots.release()
            raise

    def job(self, identifier):
        with self._lock:
            if identifier not in self._jobs:
                raise NotFoundError("TIMING_JOB_NOT_FOUND", "任务已结束或服务已重启，请从研究记录查看已保存结果。")
            return dict(self._jobs[identifier])

    def _update(self, identifier, **fields):
        with self._lock:
            self._jobs[identifier].update(fields)

    def _work(self, identifier, request, prepared, candidates=()):
        self._update(identifier, status="running")
        try:
            successes = 0
            inputs = {}
            def load(code):
                if code not in inputs:
                    inputs[code] = self.loader(self.market_data_dir, code, None, str(request.end_date), request.price_basis)
                return inputs[code]
            for progress, target in enumerate(request.targets, 1):
                try:
                    result, arrays = self._product(request, prepared, target.product_id, candidates=candidates, load=load)
                except Exception as exc:
                    LOG.exception("Timing product failed: %s", target.product_id)
                    result = {"product_id": target.product_id, "status": "error", "error": getattr(exc, "message", "该产品计算失败，请检查数据与公式；详细原因已记录到服务日志。")}
                    arrays = None
                self.repository.save_product(identifier, target.product_id, result, arrays)
                successes += result["status"] == "ok"
                # Release the product's large arrays before loading the next.
                del result, arrays
                self._update(identifier, progress=progress)
            self.repository.finish_run(identifier, {
                "name": prepared.definition.name, "execution": self.graph.audit(prepared),
                "restrictions": (["ETF 改编训练研究：仅使用训练期已完成交易，样本外冻结；历史股票实验绩效不适用。", "样本外分段沿用同一冻结政策，不重复拟合。"] + RESTRICTIONS[2:]) if prepared.definition.training else RESTRICTIONS, "status": "completed", "successful_products": successes,
                "engine_version": numeric.ENGINE_VERSION,
            })
            self._update(identifier, status="completed", run_id=identifier)
        except Exception:
            LOG.exception("Timing job failed: %s", identifier)
            self._update(identifier, status="failed", error="研究结果保存失败。已写入的产品证据保留，请检查数据磁盘后重新运行。")
        finally:
            self._slots.release()

    @staticmethod
    def _simulate(bars, entry, exit_, start, end, execution):
        path, trades, count, status = numeric.simulate_kernel(
            bars.open, bars.high, bars.low, bars.close, entry, exit_, start, end,
            execution.max_holding_bars, execution.cooldown_bars, execution.fee_bps,
            execution.slippage_bps, execution.take_profit, execution.stop_loss,
        )
        if status:
            label = "缺失" if status == 1 else "不合法"
            raise ValidationError("TIMING_MARKET_DATA_INVALID", f"研究区间存在{label}的 OHLC 行情；未跳过交易日计算，请先检查数据质量。")
        path.setflags(write=False)
        trades.setflags(write=False)
        return path, trades, count

    def _product(self, request, prepared, code, *, candidates=(), load=None):
        if load is None:
            cache = {}
            def load(identifier):
                if identifier not in cache:
                    cache[identifier] = self.loader(self.market_data_dir, identifier, None, str(request.end_date), request.price_basis)
                return cache[identifier]
        bars = load(code)
        start, split, end = _index(bars.dates, request.start_date), _index(bars.dates, request.holdout_start), len(bars.dates)
        if not 0 <= start < split < end or end > 12000:
            raise ValidationError("TIMING_INSUFFICIENT_DATA", "产品上市日期或行情覆盖不足以划分样本内外，请调整研究日期。")
        if numeric.availability_status_kernel(bars.dates, bars.available_days, 0, end):
            raise ValidationError("TIMING_NOT_CAUSAL", "存在晚于行情日期才可得的数据，不能在历史当日形成信号。")
        if numeric.execution_volume_status_kernel(bars.volume, start, end):
            raise ValidationError("TIMING_NO_TRADING_VOLUME", "研究区间存在缺失或非正成交量，无法确认按日成交；首期不支持停牌或无成交日，请检查数据或缩小区间。")
        validate_baskets(request, prepared)
        baskets, basket_lineage = bind_baskets(request, prepared, bars, load)
        channels = self.graph.evaluate(prepared, bars, baskets)
        entry = channels[prepared.definition.entry]
        training_audit = None
        if prepared.definition.training:
            if not candidates:
                candidates = self._training_plans.get(prepared.digest, ())
            if not candidates:
                raise ConflictError("TIMING_PREPARE_REQUIRED", "请先准备完整训练计划，再运行研究。")
            entry, training_audit = fit_and_route(prepared.definition.training, candidates, self.graph, bars, baskets, channels, start, split, self._simulate)
        exit_ = channels.get(prepared.definition.exit)
        if exit_ is None:
            exit_ = np.zeros(end, dtype=np.int64)
            exit_.setflags(write=False)
        execution = prepared.definition.execution
        path, trades, count = self._simulate(bars, entry, exit_, start, end, execution)
        summary = {"all": _metrics(numeric.analyze_kernel(path, trades, count, start, end)),
                   "in_sample": _metrics(numeric.analyze_kernel(path, trades, count, start, split))}
        month_ids = np.asarray([int(_day(day)[:7].replace("-", "")) for day in bars.dates], dtype=np.int64)
        month_ids.setflags(write=False)
        diagnostics = {"all": {key: _number(value) for key, value in zip(numeric.DIAGNOSTIC_COLUMNS, numeric.diagnostics_kernel(entry, month_ids, path, start, end))}}
        oos_path, oos_trades, oos_count = self._simulate(bars, entry, exit_, split, end, execution)
        summary["out_of_sample"] = _metrics(numeric.analyze_kernel(oos_path, oos_trades, oos_count, split, end))
        diagnostics["out_of_sample"] = {key: _number(value) for key, value in zip(numeric.DIAGNOSTIC_COLUMNS, numeric.diagnostics_kernel(entry, month_ids, oos_path, split, end))}
        del oos_path, oos_trades
        periods = {}
        for name, width in (("yearly", 4), ("monthly", 7)):
            periods[name] = [{"period": label, **_metrics(numeric.analyze_kernel(path, trades, count, a, b))}
                             for label, a, b in _groups(bars.dates, start, end, width)]
        folds = []
        span = end - split
        if span < request.walk_forward_splits:
            raise ValidationError("TIMING_OOS_TOO_SHORT", "样本外交易日少于分段数量，请减少分段或扩大日期范围。")
        for index in range(request.walk_forward_splits):
            a = split + span * index // request.walk_forward_splits
            b = split + span * (index + 1) // request.walk_forward_splits
            p, t, n = self._simulate(bars, entry, exit_, a, b, execution)
            folds.append({"period": f"{_day(bars.dates[a])}～{_day(bars.dates[b-1])}", **_metrics(numeric.analyze_kernel(p, t, n, a, b))})
            del p, t
        quality = numeric.signal_quality_kernel(bars.open, bars.high, bars.low, bars.close, entry, split, end)
        dates = [_day(value) for value in bars.dates[start:end]]
        curve = [{"date": date, "close": _number(bars.close[start + i]), "nav": _number(row[0]), "buy_hold_nav": _number(row[1]),
                  "position": _number(row[4]), "action": int(row[7]), "reason": "entry_signal" if row[7] == 1 else numeric.REASON_LABELS[int(row[8])]}
                 for i, (date, row) in enumerate(zip(dates, path[start:end]))]
        trade_rows = [{"signal_date": _day(bars.dates[int(row[0])]), "entry_date": _day(bars.dates[int(row[1])]), "exit_date": _day(bars.dates[int(row[2])]),
                       **{key: _number(row[i]) for i, key in enumerate(numeric.TRADE_COLUMNS) if i >= 3 and i != 7}, "reason": numeric.REASON_LABELS[int(row[7])]}
                      for row in trades[:count]]
        labels = {node.id: node.label for node in prepared.definition.nodes}
        channel_rows = [{"id": ref, "label": f"{labels[ref.split('.')[0]]} · {ref.split('.')[1]}", "type": prepared.ports[ref],
                         "values": [None if prepared.ports[ref] == "condition" and value == -1 else _number(value) for value in values[start:end]]}
                        for ref, values in channels.items() if values.ndim == 1]
        # Panel previews keep each real member identifiable; no positional blend
        # or a fabricated single line stands in for a multi-member output.
        for ref, values in channels.items():
            if values.ndim != 2:
                continue
            kind = "condition" if prepared.ports[ref] == "condition_panel" else "series"
            members = request.context_baskets[prepared.panel_groups[ref]]
            for index, member in enumerate(members):
                channel_rows.append({"id": f"{ref}[{member}]", "label": f"{labels[ref.split('.')[0]]} · {member}", "type": kind,
                    "values": [None if kind == "condition" and value == -1 else _number(value) for value in values[index, start:end]]})
        if training_audit:
            channel_rows.append({"id": "training.entry", "label": "冻结训练政策 · 最终入场", "type": "condition", "values": [None if value == -1 else int(value) for value in entry[start:end]]})
        result = {"product_id": code, "status": "ok", "detail_loaded": True, "summary": summary, "diagnostics": diagnostics, **periods, "walk_forward": folds,
                  "signal_quality": [{key: _number(value) for key, value in zip(numeric.QUALITY_COLUMNS, row)} for row in quality],
                  "curve": curve, "trades": trade_rows, "channels": channel_rows, "lineage": {**bars.lineage, "context_baskets": basket_lineage},
                  "warnings": list(bars.warnings) + ["样本外结果从空仓独立运行；图表与年度/月度表展示完整研究区间账户。", "5/10/15 期信号质量只统计样本外，未扣费用，可重叠；不等于实际交易业绩。", "买入持有是未扣费的市场基准；策略已双边计入费用与滑点。"],
                  "execution": self.graph.audit(prepared)}
        if training_audit:
            result["training"] = training_audit
            result["warnings"] += training_audit["warnings"] + ["节点预览展示草稿默认参数；实际所选参数记录在训练审计，最终入场请查看“冻结训练政策”。"]
        if basket_lineage:
            result["warnings"].append("环境篮子由本次研究固定选择，不代表历史可投资成分，可能存在幸存者与选择偏差；不是官方全市场广度。")
        arrays = {name: getattr(bars, name) for name in ("dates", "open", "high", "low", "close", "volume", "available_days", "raw_open", "raw_high", "raw_low", "raw_close")}
        arrays.update(entry=entry, exit=exit_, path=path, trades=trades[:count])
        arrays.update({f"channel_{index}": value for index, value in enumerate(channels.values())})
        arrays.update({f"basket_{group}": value for group, value in baskets.items()})
        return result, arrays

    def compare(self, identifiers):
        if len(set(identifiers)) != len(identifiers):
            raise ValidationError("TIMING_DUPLICATE_RUN", "请比较不同的研究记录。")
        runs = [self.repository.get_run(identifier, limit=1) for identifier in identifiers]
        def comparison_key(run):
            request = run["request_snapshot"]
            costs = run["definition_snapshot"]["execution"]
            market_hashes = tuple((product["product_id"], product.get("lineage", {}).get("source_hash"), str(product.get("lineage", {}).get("context_baskets", {}))) for product in run["products"])
            return (tuple(str(request.get(key)) for key in ("targets", "start_date", "end_date", "holdout_start", "price_basis", "context_baskets")),
                    costs["fee_bps"], costs["slippage_bps"], market_hashes)
        if any(comparison_key(run) != comparison_key(runs[0]) for run in runs[1:]):
            raise ValidationError("TIMING_COMPARISON_MISMATCH", "请使用相同产品、日期、样本外起点、价格口径、费用及数据快照进行对比。")
        return {"items": [{"run_id": run["id"], "name": run["name"], "products": [{key: value for key, value in product.items() if key not in {"curve", "trades", "channels"}} for product in run["products"]]} for run in runs]}
