from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
import pandas as pd
from pathlib import Path
from datetime import datetime
from functools import lru_cache
from fastapi import FastAPI, Query, Request
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from starlette.responses import HTMLResponse, JSONResponse, FileResponse

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))
# Canonical package names embedded in NJIT caches must resolve from both the
# repository root and `cd backend; uvicorn app:app` startup modes.
if str(BACKEND_DIR.parent) not in sys.path:
    sys.path.append(str(BACKEND_DIR.parent))

from optimizer import warm_optimizer_numba_kernels
from fit import compute_rolling_corr_classes
from strategy import (
    compute_risk_budget_weights,
    scale_weights_percent,
    strategy_execution_audit,
)
from fit import (
    ClassSpec,
    ETFSpec,
    _load_adj_nav,
    _pick_series,
    compute_classes_nav,
    serialize_rolling_correlation_payload,
)
from market_data import resolve_market_data_file, resolve_tushare_data_dir
from backend.data_storage import StorageError, StorageManager, storage_lifespan
from pit.context import PitContextError, ResearchContext, resolve_request_context
from product_pools.membership import universe_pit_lineage
from cal_indicators.typed_numeric_backend import warm_typed_numeric_backend



class SaveRequest(BaseModel):
    asset_alloc_name: str
    classes: List[FitClassIn]
    # Provenance. Without these the saved allocation cannot answer "which
    # product pool, which data vintage, which research day produced this",
    # which is exactly what an audit asks first.
    universe_snapshot_id: Optional[str] = None
    data_release_id: Optional[str] = None
    as_of: Optional[str] = None
    run_mode: str = "RESEARCH"


class ETFIn(BaseModel):
    code: str
    name: str
    riskContribution: float = Field(ge=0, description="风险贡献占比 0~100")


class SolveRequest(BaseModel):
    assetClassId: str
    riskMetric: str = "vol"
    maxLeverage: float = 0.0
    etfs: List[ETFIn]


class SolveResponse(BaseModel):
    weights: List[float]
    execution: dict


class FitETFIn(BaseModel):
    code: str
    name: str
    weight: float


class FitClassIn(BaseModel):
    id: str
    name: str
    etfs: List[FitETFIn]


class RollingResponse(BaseModel):
    dates: List[str]
    series: dict
    metrics: List[dict]
    execution: dict


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Fail closed until every main-process and worker NJIT lane is hot."""

    from backend.data_sources.etl_executor import reconnect
    from backend.data_sources.etl_store import EtlStore
    from backend.data_sources.store import SourceStore
    _app.state.etl_runtime = reconnect(EtlStore(SourceStore()))
    optimizer_status = warm_optimizer_numba_kernels()
    typed_status = warm_typed_numeric_backend()
    from backtest_numba import warm_backtest_numba_kernels
    from fit_numba import warm_fit_numba_kernels
    from auto_class_numba import warm_auto_class_numba_kernels
    from historical_regimes.numba_kernels import (
        warm_historical_regime_numba_kernels,
    )
    from historical_regimes.v2_numba import warm_regime_graph_numba_kernels
    from historical_regimes.taa import warm_taa_numba_kernels
    from custom_indicators.portfolio_numba import warm_portfolio_numba_kernels
    from scenario_stress.numba_kernels import warm_scenario_numba_kernels
    from synthetic_series_numba import warm_synthetic_series_numba_kernel
    from instrument_analytics_numba import warm_instrument_analytics_numba_kernels
    from product_analysis_numba import warm_product_analysis_numba_kernels
    from business_numeric_numba import warm_business_numeric_kernels
    from research_series.numba_kernels import warm_research_series_numba_kernels
    from strategy import warm_strategy_numba_kernels
    from services.custom_indicator_routes import indicator_service
    from services.historical_regime_routes import regime_graph_v2_service

    backtest_status = warm_backtest_numba_kernels()
    fit_status = warm_fit_numba_kernels()
    auto_class_status = warm_auto_class_numba_kernels()
    historical_regime_status = warm_historical_regime_numba_kernels()
    regime_graph_status = warm_regime_graph_numba_kernels()
    taa_status = warm_taa_numba_kernels()
    from services.tactical_allocation_routes import tactical_service
    tactical_status = tactical_service.warm()
    if tactical_status.get("complete") is not True:
        raise RuntimeError("战术资产配置 NJIT 启动预热未完成")
    portfolio_status = warm_portfolio_numba_kernels()
    scenario_stress_status = warm_scenario_numba_kernels()
    synthetic_series_status = warm_synthetic_series_numba_kernel()
    instrument_analytics_status = warm_instrument_analytics_numba_kernels()
    product_analysis_status = warm_product_analysis_numba_kernels()
    business_numeric_status = warm_business_numeric_kernels()
    research_series_status = warm_research_series_numba_kernels()
    strategy_status = warm_strategy_numba_kernels()
    from services.factor_research_routes import factor_service
    factor_status = factor_service.warm()
    from services.timing_research_routes import timing_service
    timing_status = timing_service.warm()
    from backend.sensitivity.kernels import warm_sensitivity_kernels
    sensitivity_status = warm_sensitivity_kernels()
    if sensitivity_status.get("complete") is not True:
        raise RuntimeError("风险模型与已发布情景计算内核未完成启动预热")
    if factor_status.get("complete") is not True:
        raise RuntimeError("因子研究中心 NJIT 启动预热未完成")
    regime_graph_plan_status = regime_graph_v2_service.prewarm_saved_definitions()
    if regime_graph_plan_status.get("complete") is not True:
        raise RuntimeError("历史情景 V2 已保存定义未能全部完成启动预热")
    from backend.data_sources.resolution_kernels import warm_resolution_kernels
    resolution_status = warm_resolution_kernels()
    if not resolution_status["complete"]:
        raise RuntimeError("Multi-source resolution NJIT warmup incomplete")
    indicator_service.start_compute_engine()
    _app.state.numba_warmup = {
        "complete": True,
        "optimizer": optimizer_status,
        "indicators": typed_status,
        "backtest": backtest_status,
        "fit_analytics": fit_status,
        "auto_asset_class": auto_class_status,
        "historical_regimes": historical_regime_status,
        "regime_graph_v2": regime_graph_status,
        "regime_graph_v2_saved_plans": regime_graph_plan_status,
        "taa": taa_status,
        "tactical_allocation": tactical_status,
        "portfolio_research": portfolio_status,
        "scenario_stress": scenario_stress_status,
        "synthetic_product_series": synthetic_series_status,
        "instrument_analytics": instrument_analytics_status,
        "product_analysis": product_analysis_status,
        "business_numeric": business_numeric_status,
        "research_series": research_series_status,
        "portfolio_strategy": strategy_status,
        "workers": indicator_service.compute_engine.status(),
        "source_resolution": resolution_status,
        "factor_research": factor_status,
        "timing_research": timing_status,
        "published_sensitivity": sensitivity_status,
    }
    # Share the PIT page's single background scan, after all NJIT workers warm.
    # This is diagnostic cache priming, not a waiver of PIT/publication checks.
    from pit.audit import start_scan
    try:
        start_scan(DATA_DIR)
        _app.state.pit_audit_start_error = None
    except Exception as exc:  # diagnostics must never prevent API startup
        _app.state.pit_audit_start_error = type(exc).__name__
    try:
        yield
    finally:
        timing_service.close()
        indicator_service.close_compute_engine()


app = FastAPI(title="Fund Investment Research Platform", lifespan=storage_lifespan(lifespan))


@app.middleware('http')
async def storage_readiness(request: Request, call_next):
    # Keep the diagnostic endpoint reachable if a mounted data disk disconnects.
    if not request.url.path.startswith('/api/data-storage'):
        try:
            StorageManager().guard()
        except (StorageError, OSError) as exc:
            return JSONResponse(status_code=503, content={'detail': {
                'code': exc.code if isinstance(exc, StorageError) else 'STORAGE_IO_ERROR',
                'message': exc.message if isinstance(exc, StorageError) else '数据磁盘无法访问，请重新连接。',
            }})
    return await call_next(request)

def _system_pit() -> "ResearchContext":
    """The system-level PIT口径 for a request that states none of its own.

    `DATA_DIR` is defined further down this module; the lookup happens when a
    request calls this, long after import finishes. Measured at 0.08 ms, so it
    runs per request rather than being cached into staleness.
    """

    return resolve_request_context(DATA_DIR)



def _cors_origins() -> List[str]:
    """Read a comma-separated CORS allowlist without permitting wildcard origins."""

    configured = os.getenv("CORS_ALLOW_ORIGINS", "")
    if configured.strip():
        return [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]

    if os.getenv("APP_ENV", "development").lower() != "production":
        return [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]

    # The built frontend is served from this same FastAPI origin in production,
    # so cross-origin requests are unnecessary unless explicitly configured.
    return []


# Added before CORS so CORS stays the outermost layer: a rejected PIT header
# still has to come back as a readable 400 to a cross-origin dev frontend.
from services.pit_routes import PitViewOverrideMiddleware

app.add_middleware(PitViewOverrideMiddleware)

# CORS is only needed for separate frontend deployments. Production defaults to
# same-origin access and never falls back to an unrestricted wildcard.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包括模块化路由器（策略，分析）
try:
    from services.strategy_routes import router as strategy_router

    app.include_router(strategy_router)
except Exception:
    pass
try:
    from services.analytics_routes import router as analytics_router

    app.include_router(analytics_router)
except Exception:
    pass
from services.data_routes import router as data_router
from services.data_model_routes import router as data_model_router
from services.data_source_routes import router as data_source_router
from services.etl_routes import router as etl_router
from services.custom_indicator_routes import router as custom_indicator_router
from services.instrument_analytics import (
    build_legacy_etf_analytics_response,
    build_legacy_etf_trend_response,
    instrument_analytics_execution_audit,
)
from services.instrument_routes import (
    instrument_product_detail as unified_instrument_product_detail,
    instrument_products as unified_instrument_products,
    router as instrument_router,
)
from services.index_routes import router as index_router
from services.portfolio_routes import router as portfolio_router
from services.historical_regime_routes import router as historical_regime_router
from services.scenario_stress_routes import router as scenario_stress_router
from services.business_numeric_routes import router as business_numeric_router
from services.research_series_routes import router as research_series_router
from services.auto_class_routes import router as auto_class_router
from services.product_pool_routes import router as product_pool_router
from services.pit_routes import router as pit_router
from services.factor_research_routes import router as factor_research_router
from services.timing_research_routes import router as timing_research_router
from services.tactical_allocation_routes import router as tactical_allocation_router
from services.localization_routes import router as localization_router

app.include_router(data_router)
app.include_router(pit_router)
app.include_router(data_model_router)
app.include_router(data_source_router)
from services.storage_routes import router as storage_router
app.include_router(storage_router)
app.include_router(etl_router)
app.include_router(custom_indicator_router)
app.include_router(instrument_router)
app.include_router(index_router)
app.include_router(portfolio_router)
app.include_router(historical_regime_router)
app.include_router(scenario_stress_router)
app.include_router(business_numeric_router)
app.include_router(research_series_router)
app.include_router(auto_class_router)
app.include_router(product_pool_router)
app.include_router(factor_research_router)
app.include_router(timing_research_router)
app.include_router(tactical_allocation_router)
from services.risk_model_routes import risk_model_router, transmission_router
from services.published_scenario_routes import router as published_scenario_router
app.include_router(risk_model_router)
app.include_router(transmission_router)
app.include_router(published_scenario_router)
app.include_router(localization_router)


@app.get("/api/health")
def health():
    from pit.audit import scan_status
    pit_status = scan_status()
    start_error = getattr(app.state, "pit_audit_start_error", None)
    if start_error:
        pit_status = {**pit_status, "state": "failed", "error": start_error}
    return {
        "ok": True,
        "numba_warmup": getattr(app.state, "numba_warmup", {"complete": False}),
        "pit_audit": {**pit_status, "complete": pit_status["state"] == "ready"},
    }


@app.post("/api/risk-parity/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    """Solve covariance risk budgets only through precompiled NJIT kernels."""

    if not req.etfs:
        return JSONResponse(status_code=400, content={"detail": "风险预算至少需要一个产品"})
    if req.riskMetric != "vol":
        return JSONResponse(
            status_code=400,
            content={"detail": "当前风险预算求解仅支持波动率；VaR/ES 不会被静默当作波动率处理"},
        )
    try:
        _pit = _system_pit()
        data = _load_adj_nav(
            DATA_DIR,
            [item.code for item in req.etfs],
            [item.name for item in req.etfs],
            as_of=_pit.as_of, run_mode=_pit.run_mode,
        )
    except (FileNotFoundError, ValueError) as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    series: list[pd.Series] = []
    missing: list[str] = []
    for index, item in enumerate(req.etfs):
        selected = _pick_series(data, item.code, item.name)
        if selected is None or selected.empty:
            missing.append(item.name or item.code)
            continue
        series.append(selected.rename(f"asset_{index}"))
    if missing:
        return JSONResponse(
            status_code=400,
            content={"detail": f"以下产品缺少真实净值，禁止用预算权重降级替代：{', '.join(missing)}"},
        )

    nav_wide = pd.concat(series, axis=1, join="inner").dropna(axis=0, how="any")
    if len(nav_wide.index) < 3:
        return JSONResponse(status_code=400, content={"detail": "完整交集净值样本不足，无法求解风险预算"})
    budgets = [float(item.riskContribution) for item in req.etfs]
    try:
        normalized = compute_risk_budget_weights(
            nav_wide,
            {"metric": "vol"},
            budgets,
        )
        weights = scale_weights_percent(normalized, float(req.maxLeverage))
        execution = strategy_execution_audit()
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    return SolveResponse(weights=weights, execution=execution)


class RollingClassesRequest(BaseModel):
    startDate: str
    window: int = 60
    targetClassName: str
    classes: List[FitClassIn]


@app.post("/api/rolling-corr-classes", response_model=RollingResponse)
def rolling_corr_classes(req: RollingClassesRequest):
    try:
        start = pd.to_datetime(req.startDate)
    except Exception:
        raise ValueError("startDate 格式错误，应为 YYYY-MM-DD")
    classes = [
        ClassSpec(
            id=c.id,
            name=c.name,
            etfs=[ETFSpec(code=e.code, name=e.name, weight=float(e.weight)) for e in c.etfs],
        )
        for c in req.classes
    ]
    _pit = _system_pit()
    idx, series_map, metrics = compute_rolling_corr_classes(DATA_DIR, classes, start,
                                                            int(req.window),
                                                            req.targetClassName,
                                                            as_of=_pit.as_of, run_mode=_pit.run_mode)
    return RollingResponse(
        **serialize_rolling_correlation_payload(idx, series_map, metrics)
    )


@app.post("/api/save-allocation")
def save_allocation(req: SaveRequest):
    alloc_name = (req.asset_alloc_name or "").strip()
    if not alloc_name:
        raise ValueError("配置名称不能为空")
    if not req.classes:
        raise ValueError("资产大类配置不能为空")

    info_path = DATA_DIR / "asset_alloc_info.parquet"
    nv_path = DATA_DIR / "asset_nv.parquet"
    now = datetime.now()

    # 1. 校验并保存配置信息
    if info_path.exists():
        info_df = pd.read_parquet(info_path)
        if alloc_name in info_df["asset_alloc_name"].unique():
            return JSONResponse(status_code=400, content={"detail": f"配置名称 '{alloc_name}' 已存在"})
    else:
        info_df = pd.DataFrame()

    try:
        context = resolve_request_context(DATA_DIR, req.as_of, req.run_mode, req.data_release_id)
        # 这条大类净值会被下游回测当成行情读，所以搭它的产品池是不是"事后筛出来的"
        # 必须在落盘之前判掉——存进去就再也分不清了。
        universe = universe_pit_lineage(DATA_DIR, req.universe_snapshot_id, context)
    except PitContextError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    new_rows = []
    for ac in req.classes:
        for etf in ac.etfs:
            new_rows.append({
                "asset_alloc_name": alloc_name,
                "asset_name": ac.name,
                "etf_code": etf.code,
                "etf_name": etf.name,
                "etf_weight": etf.weight,
                "creat_time": now,
                "universe_snapshot_id": req.universe_snapshot_id or None,
                "data_release_id": context.data_release_id,
                "as_of": context.as_of,
                "run_mode": context.run_mode,
            })

    new_info_df = pd.DataFrame(new_rows)
    updated_info_df = pd.concat([info_df, new_info_df], ignore_index=True)
    updated_info_df.to_parquet(info_path, index=False)

    # 2. 计算并保存虚拟净值
    try:
        start_date = pd.to_datetime("2010-01-01")  # 从一个较早的日期开始计算以获取完整历史
        classes_spec = [
            ClassSpec(id=c.id, name=c.name, etfs=[ETFSpec(code=e.code, name=e.name, weight=e.weight) for e in c.etfs])
            for c in req.classes]
        fit_result = compute_classes_nav(
            DATA_DIR, classes_spec, start_date, as_of=context.as_of, run_mode=context.run_mode
        )
        NAV = fit_result.nav
        nav_lineage = fit_result.lineage
        nav_availability = fit_result.available_at

        # 将宽表 NAV 转换为长表
        nav_long = NAV.reset_index().melt(id_vars=["date"], var_name="asset_name", value_name="nv")
        nav_long["asset_alloc_name"] = alloc_name
        nav_long["creat_time"] = now
        # 这条序列是"哪一天算出来的"和"每一天什么时候才可知"——缺了这两列，
        # 2026 年用全历史算出的净值会被当成 2018 年就存在的行情来回测。
        nav_long["as_of"] = context.as_of
        nav_long["run_mode"] = context.run_mode
        nav_dates = pd.to_datetime(nav_long["date"])
        nav_long["available_at"] = (
            nav_dates.map(nav_availability) if len(nav_availability) else pd.Series(pd.NaT, index=nav_long.index)
        )
        nav_long["available_at"] = pd.to_datetime(nav_long["available_at"]).fillna(nav_dates)
        # 重新排序字段
        nav_long = nav_long[
            [
                "asset_alloc_name",
                "asset_name",
                "date",
                "nv",
                "creat_time",
                "as_of",
                "run_mode",
                "available_at",
            ]
        ]

        if nv_path.exists():
            nv_df = pd.read_parquet(nv_path)
            updated_nv_df = pd.concat([nv_df, nav_long], ignore_index=True)
        else:
            updated_nv_df = nav_long
        updated_nv_df.to_parquet(nv_path, index=False)

    except Exception as e:
        # 如果净值计算失败，为了数据一致性，回滚已保存的配置信息
        if info_path.exists():
            info_df_rollback = pd.read_parquet(info_path)
            info_df_rollback = info_df_rollback[info_df_rollback["asset_alloc_name"] != alloc_name]
            if info_df_rollback.empty:
                info_path.unlink()
            else:
                info_df_rollback.to_parquet(info_path, index=False)
        from backend.research_input_checks import ResearchInputError
        if isinstance(e, ResearchInputError):
            return JSONResponse(status_code=422, content={"detail": e.detail()})
        return JSONResponse(status_code=500, content={"detail": f"计算并保存净值时出错: {e}"})

    return {
        "ok": True,
        "message": f"配置 '{alloc_name}' 已成功保存",
        "lineage": {
            "universe_snapshot_id": req.universe_snapshot_id or None,
            "data_release_id": context.data_release_id,
            "as_of": context.as_of,
            "run_mode": context.run_mode,
            "pit": {**nav_lineage, "universe": universe},
        },
    }


@app.get("/api/list-allocations")
def list_allocations():
    info_path = DATA_DIR / "asset_alloc_info.parquet"
    if not info_path.exists():
        return []
    df = pd.read_parquet(info_path)
    return sorted(df["asset_alloc_name"].unique().tolist())


@app.get("/api/allocation-lineage")
def allocation_lineage(name: str):
    """Provenance of one saved allocation.

    Allocations saved before the lineage columns existed answer with nulls and
    `traceable: false` — an honest "we do not know" beats inventing a snapshot.
    """

    info_path = DATA_DIR / "asset_alloc_info.parquet"
    if not info_path.exists():
        return JSONResponse(status_code=404, content={"detail": "配置文件不存在"})
    df = pd.read_parquet(info_path)
    rows = df[df["asset_alloc_name"] == name]
    if rows.empty:
        return JSONResponse(status_code=404, content={"detail": f"未找到名为 '{name}' 的配置"})

    def first(column: str):
        if column not in rows.columns:
            return None
        values = rows[column].dropna()
        return str(values.iloc[0]) if not values.empty else None

    lineage = {
        "asset_alloc_name": name,
        "universe_snapshot_id": first("universe_snapshot_id"),
        "data_release_id": first("data_release_id"),
        "as_of": first("as_of"),
        "run_mode": first("run_mode"),
        "created_at": first("creat_time"),
    }
    lineage["traceable"] = bool(lineage["universe_snapshot_id"] and lineage["as_of"])
    return lineage


@app.get("/api/load-allocation")
def load_allocation(name: str):
    info_path = DATA_DIR / "asset_alloc_info.parquet"
    if not info_path.exists():
        return JSONResponse(status_code=404, content={"detail": "配置文件不存在"})

    df = pd.read_parquet(info_path)
    alloc_df = df[df["asset_alloc_name"] == name]
    if alloc_df.empty:
        return JSONResponse(status_code=404, content={"detail": f"未找到名为 '{name}' 的配置"})

    # 从扁平表重建嵌套结构
    classes_map = {}
    for _, row in alloc_df.iterrows():
        class_name = row["asset_name"]
        if class_name not in classes_map:
            classes_map[class_name] = {
                "id": f"loaded-{class_name}-{datetime.now().timestamp()}",
                "name": class_name,
                "mode": "custom",  # 默认导入为自定义权重模式
                "etfs": [],
                "riskMetric": "vol",
                "maxLeverage": 0,
            }
        classes_map[class_name]["etfs"].append({
            "code": row["etf_code"],
            "name": row["etf_name"],
            "weight": row["etf_weight"],
        })

    return list(classes_map.values())


# -------------------- ETF Universe from data/ --------------------
DATA_DIR = (Path(__file__).resolve().parents[1] / "data").resolve()


def _load_universe() -> List[dict]:
    """Load the ETF universe through the active Tushare manifest.

    Priority: active etf_info_df.parquet -> legacy JSON fallback -> empty list
    Expected fields: ts_code/code and name
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pq_path = resolve_market_data_file("etf_info_df.parquet", DATA_DIR)
    if pq_path.exists():
        try:
            df = pd.read_parquet(pq_path)
            code_col = "ts_code" if "ts_code" in df.columns else ("code" if "code" in df.columns else None)
            name_col = "name" if "name" in df.columns else ("fund_name" if "fund_name" in df.columns else None)
            mgmt_col = "management" if "management" in df.columns else ("manager" if "manager" in df.columns else None)
            fd_col = "found_date" if "found_date" in df.columns else (
                "foundation_date" if "foundation_date" in df.columns else None
            )
            if code_col and name_col:
                cols = [code_col, name_col]
                if mgmt_col:
                    cols.append(mgmt_col)
                if fd_col:
                    cols.append(fd_col)
                out_df = df[cols].dropna(subset=[code_col, name_col]).drop_duplicates()
                items: List[dict] = []
                for _, r in out_df.iterrows():
                    items.append(
                        {
                            "code": str(r[code_col]),
                            "name": str(r[name_col]),
                            "management": None if not mgmt_col else (
                                None if pd.isna(r[mgmt_col]) else str(r[mgmt_col])),
                            "found_date": _normalize_date(None if not fd_col else r[fd_col]),
                        }
                    )
                return items
        except Exception:
            pass
    json_path = resolve_market_data_file("etf_universe.json", DATA_DIR)
    if json_path.exists():
        try:
            arr = json.loads(json_path.read_text(encoding="utf-8"))
            out = []
            for x in arr:
                code = x.get("code") or x.get("ts_code")
                name = x.get("name") or x.get("fund_name") or ""
                mgmt = x.get("management") or x.get("manager")
                fd = x.get("found_date") or x.get("foundation_date")
                if code and name:
                    out.append(
                        {
                            "code": str(code),
                            "name": str(name),
                            "management": None if mgmt is None else str(mgmt),
                            "found_date": _normalize_date(fd),
                        }
                    )
            return out
        except Exception:
            pass
    return []


def _normalize_date(v) -> Optional[str]:
    if v is None:
        return None
    try:
        # handle 20180101 or '2018-01-01'
        s = str(v)
        if s.isdigit() and len(s) == 8:
            return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
        dt = pd.to_datetime(v, errors="coerce")
        if pd.isna(dt):
            return None
        return str(dt.date())
    except Exception:
        return None


@lru_cache(maxsize=1)
def _cached_universe_with_mtime(identity: tuple[tuple[str, int, int], ...]) -> List[dict]:  # noqa: ARG001
    return _load_universe()


def _get_universe() -> List[dict]:
    # Invalidate cache when files change
    identities = []
    for fname in ("etf_universe.json", "etf_info_df.parquet"):
        p = resolve_market_data_file(fname, DATA_DIR)
        if p.exists():
            stat = p.stat()
            identities.append((str(p), stat.st_mtime_ns, stat.st_size))
    return _cached_universe_with_mtime(tuple(identities))


@app.get("/api/etf/search")
def etf_search(
        q: Optional[str] = Query(default=""),
        k: Optional[int] = Query(default=None),  # deprecated by page/page_size
        sort_by: str = Query(default="name"),  # one of: name, code, found_date, management
        sort_dir: str = Query(default="asc"),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=10, ge=1, le=200),
):
    arr = _get_universe()
    qnorm = (q or "").strip().lower()
    filtered: List[dict]
    if not qnorm:
        filtered = arr
    else:
        filtered = []
        for x in arr:
            hay = f"{x.get('code', '')} {x.get('name', '')} {x.get('management', '')}".lower()
            if qnorm in hay:
                filtered.append(x)
    reverse = sort_dir.lower() == "desc"
    key = (lambda x: (x.get(sort_by) or "")) if sort_by in {"name", "code", "management", "found_date"} else (
        lambda x: x.get("name") or "")
    filtered.sort(key=key, reverse=reverse)
    total = len(filtered)
    if k is not None and k > 0:
        # compatibility: take top-k of filtered then apply pagination
        filtered = filtered[:k]
    # pagination
    start = (page - 1) * page_size
    end = start + page_size
    items = filtered[start:end]
    return JSONResponse(
        {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "execution": instrument_analytics_execution_audit(),
        }
    )


@app.get("/api/etf/analytics")
def etf_analytics():
    payload = build_legacy_etf_analytics_response(
        data_dir=resolve_tushare_data_dir(DATA_DIR),
    )
    if payload is None:
        return JSONResponse(status_code=404, content={"detail": "未找到 etf_info_df 数据文件"})
    return JSONResponse(payload)


@app.get("/api/etf/analytics/list_trend")
def etf_list_trend(
    dimension: str = Query("all", description="筛选维度，可选 all/type/invest_type/fund_type/management"),
    values: Optional[List[str]] = Query(None, description="筛选值，可传多个"),
):
    payload = build_legacy_etf_trend_response(
        dimension=dimension,
        values=values,
        data_dir=resolve_tushare_data_dir(DATA_DIR),
    )
    if payload is None:
        return JSONResponse(status_code=404, content={"detail": "未找到 etf_info_df 数据文件"})
    return JSONResponse(payload)


@app.get("/api/etf/products")
def etf_products(
    q: Optional[str] = Query(default=""),
    fund_type: Optional[List[str]] = Query(default=None),
    organization_type: Optional[List[str]] = Query(default=None, alias="type"),
    invest_type: Optional[List[str]] = Query(default=None),
    market: Optional[List[str]] = Query(default=None),
    status: Optional[List[str]] = Query(default=None),
    management: Optional[List[str]] = Query(default=None),
    custodian: Optional[List[str]] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    sort_by: str = Query(default="issue_amount"),
    sort_dir: str = Query(default="desc"),
):
    payload = unified_instrument_products(
        kind="etf",
        q=q or "",
        fund_type=fund_type,
        fund_category=organization_type,
        invest_type=invest_type,
        market=market,
        status=status,
        management=management,
        custodian=custodian,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_dir="asc" if sort_dir.lower() == "asc" else "desc",
        conditions=None,
        snapshot_metrics=None,
    )
    return JSONResponse(payload)


@app.get("/api/etf/products/{product_id}")
def etf_product_detail(product_id: str):
    payload = unified_instrument_product_detail(product_id, kind="etf")
    return payload if isinstance(payload, JSONResponse) else JSONResponse(payload)


# ---- 静态页面托管（可选：将前端构建产物放到 frontend/dist 下） ----
DIST_DIR = (Path(__file__).resolve().parents[1] / "frontend" / "dist").resolve()
app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True, check_dir=False), name="static")


@app.exception_handler(404)
async def spa_fallback(request: Request, exc):  # noqa: ARG001
    """Serve the built frontend for unknown non-API routes."""
    path = request.url.path
    if path.startswith("/api"):
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    # Attempt to serve asset files directly when 静态资源已经生成
    last_segment = path.rsplit("/", 1)[-1]
    if "." in last_segment:
        suffix = Path(last_segment).suffix.lower()
        asset_suffixes = {
            ".js",
            ".css",
            ".ico",
            ".png",
            ".jpg",
            ".jpeg",
            ".svg",
            ".webp",
            ".json",
            ".txt",
            ".map",
            ".woff",
            ".woff2",
            ".ttf",
        }
        if suffix in asset_suffixes:
            asset_path = (DIST_DIR / path.lstrip("/")).resolve()
            try:
                asset_path.relative_to(DIST_DIR)
            except ValueError:
                return JSONResponse({"detail": "Not Found"}, status_code=404)
            if asset_path.exists() and asset_path.is_file():
                return FileResponse(asset_path)
            return JSONResponse({"detail": "Not Found"}, status_code=404)
    index_file = DIST_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    return JSONResponse({"detail": "Not Found"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Run FastAPI app (direct mode)")
    parser.add_argument(
        "--port", "-p", type=int,
        default=int(os.getenv("APP_PORT") or os.getenv("PORT") or 8000),
        help="Port to listen on (env: APP_PORT or PORT). Default 8000.",
    )
    args = parser.parse_args()

    uvicorn.run("app:app", host="127.0.0.1", port=args.port, reload=True, proxy_headers=False)
