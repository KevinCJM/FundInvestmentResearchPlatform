from __future__ import annotations

import json
import math
import os
import sys
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from functools import lru_cache
from fastapi import FastAPI, Query, Request
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Tuple, Dict, Any
from starlette.responses import HTMLResponse, JSONResponse, FileResponse

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))
# Canonical package names embedded in NJIT caches must resolve from both the
# repository root and `cd backend; uvicorn app:app` startup modes.
if str(BACKEND_DIR.parent) not in sys.path:
    sys.path.append(str(BACKEND_DIR.parent))

from optimizer import (
    calculate_efficient_frontier_exploration,
    returns_from_nav_matrix,
    warm_optimizer_numba_kernels,
)
from backtest_engine import backtest_portfolio, gen_rebalance_dates
from fit import compute_rolling_corr_classes, compute_class_consistency
from strategy import (
    compute_risk_budget_weights,
    compute_target_weights,
    normalize_explicit_weights,
    scale_weights_percent,
    strategy_execution_audit,
)
from fit import (
    ClassSpec,
    ETFSpec,
    _load_adj_nav,
    _pick_series,
    compute_classes_nav,
    compute_nav_performance_payload,
    compute_rolling_corr,
    serialize_rolling_correlation_payload,
)
from market_data import resolve_market_data_file, resolve_tushare_data_dir
from cal_indicators.typed_numeric_backend import warm_typed_numeric_backend


class FrontierRequest(BaseModel):
    alloc_name: str
    start_date: str
    end_date: str
    return_metric: Dict[str, Any]
    risk_metric: Dict[str, Any]
    risk_free_rate: float = 0.0  # 年化无风险利率（小数），用于夏普率
    constraints: Optional[
        Dict[str, Any]] = None  # { single_limits: {name:{lo,hi}}, group_limits: [{assets:[name], lo, hi}] }
    exploration: Optional[Dict[str, Any]] = None  # { rounds: [{samples:int, step:float, buckets:int}] }
    quantization: Optional[Dict[str, Any]] = None  # { step: float|null }
    refine: Optional[Dict[str, Any]] = None  # { use_slsqp: bool, count: int }


class SaveRequest(BaseModel):
    asset_alloc_name: str
    classes: List[FitClassIn]


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


class FitRequest(BaseModel):
    startDate: str
    classes: List[FitClassIn]


class FitResponse(BaseModel):
    dates: List[str]
    navs: dict
    corr: List[List[Optional[float]]]
    corr_labels: List[str]
    metrics: List[dict]
    consistency: List[dict]
    annual_metrics: dict
    execution: dict


class RollingRequest(BaseModel):
    startDate: str
    window: int = 60
    targetCode: str
    targetName: str
    etfs: List[FitETFIn]


class RollingResponse(BaseModel):
    dates: List[str]
    series: dict
    metrics: List[dict]
    execution: dict


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Fail closed until every main-process and worker NJIT lane is hot."""

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
    portfolio_status = warm_portfolio_numba_kernels()
    scenario_stress_status = warm_scenario_numba_kernels()
    synthetic_series_status = warm_synthetic_series_numba_kernel()
    instrument_analytics_status = warm_instrument_analytics_numba_kernels()
    product_analysis_status = warm_product_analysis_numba_kernels()
    business_numeric_status = warm_business_numeric_kernels()
    research_series_status = warm_research_series_numba_kernels()
    strategy_status = warm_strategy_numba_kernels()
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
    }
    try:
        yield
    finally:
        indicator_service.close_compute_engine()


app = FastAPI(title="Fund Investment Research Platform", lifespan=lifespan)


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

app.include_router(data_router)
app.include_router(data_model_router)
app.include_router(data_source_router)
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


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "numba_warmup": getattr(app.state, "numba_warmup", {"complete": False}),
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
        data = _load_adj_nav(
            DATA_DIR,
            [item.code for item in req.etfs],
            [item.name for item in req.etfs],
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


# Registered by services.analytics_routes; this remains an unregistered
# compatibility callable so there is only one HTTP numerical path.
def fit_classes(req: FitRequest):
    try:
        start = pd.to_datetime(req.startDate)
    except Exception:
        raise ValueError("startDate 格式错误，应为 YYYY-MM-DD")
    if not req.classes:
        raise ValueError("classes 不能为空")
    classes = [
        ClassSpec(
            id=c.id,
            name=c.name,
            etfs=[ETFSpec(code=e.code, name=e.name, weight=float(e.weight)) for e in c.etfs],
        )
        for c in req.classes
    ]
    NAV, corr, metrics = compute_classes_nav(DATA_DIR, classes, start)
    consistency_rows = compute_class_consistency(DATA_DIR, classes, start)
    performance = compute_nav_performance_payload(NAV)

    def finite_or_none(x: float):
        try:
            if x is None:
                return None
            if isinstance(x, (int, float)) and (not (x != x) and abs(x) != float('inf')):
                return float(x)
        except Exception:
            pass
        # 处理 NaN/Inf
        try:
            import math
            if isinstance(x, (int, float)) and (math.isfinite(x)):
                return float(x)
        except Exception:
            pass
        return None

    dates = [d.strftime("%Y-%m-%d") for d in NAV.index]
    navs = {col: [finite_or_none(float(x)) for x in NAV[col].tolist()] for col in NAV.columns}
    corr_labels = list(corr.columns)
    corr_vals = [[finite_or_none(float(v)) for v in row] for row in corr.values.tolist()]
    metrics_out = []
    for name, row in metrics.iterrows():
        metrics_out.append({
            "name": str(name),
            "cumulative_return": performance["cumulative_returns"].get(str(name)),
            "annual_return": finite_or_none(row.get("年化收益率", None)),
            "annual_vol": finite_or_none(row.get("年化波动率", None)),
            "sharpe": finite_or_none(row.get("夏普比率", None)),
            "var99": finite_or_none(row.get("99%VaR(日)", None)),
            "es99": finite_or_none(row.get("99%ES(日)", None)),
            "max_drawdown": finite_or_none(row.get("最大回撤", None)),
            "calmar": finite_or_none(row.get("卡玛比率", None)),
        })
    # consistency sanitize
    cons_out = []
    for row in consistency_rows:
        cons_out.append({
            "name": str(row.get("name")),
            "mean_corr": None if not isinstance(row.get("mean_corr"), (int, float)) or not (
                    row.get("mean_corr") == row.get("mean_corr")) else float(row.get("mean_corr")),
            "pca_evr1": None if not isinstance(row.get("pca_evr1"), (int, float)) or not (
                    row.get("pca_evr1") == row.get("pca_evr1")) else float(row.get("pca_evr1")),
            "max_te": None if not isinstance(row.get("max_te"), (int, float)) or not (
                    row.get("max_te") == row.get("max_te")) else float(row.get("max_te")),
        })
    return FitResponse(
        dates=dates,
        navs=navs,
        corr=corr_vals,
        corr_labels=corr_labels,
        metrics=metrics_out,
        consistency=cons_out,
        annual_metrics=performance["annual_metrics"],
        execution=performance["execution"],
    )


# Registered by services.analytics_routes.
def rolling_corr(req: RollingRequest):
    try:
        start = pd.to_datetime(req.startDate)
    except Exception:
        raise ValueError("startDate 格式错误，应为 YYYY-MM-DD")
    etfs = [ETFSpec(code=e.code, name=e.name, weight=float(e.weight)) for e in req.etfs]
    idx, series_map, metrics = compute_rolling_corr(DATA_DIR, etfs, start, int(req.window),
                                                    req.targetCode,
                                                    req.targetName)
    return RollingResponse(
        **serialize_rolling_correlation_payload(idx, series_map, metrics)
    )


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
    idx, series_map, metrics = compute_rolling_corr_classes(DATA_DIR, classes, start,
                                                            int(req.window),
                                                            req.targetClassName)
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
        NAV, _, _ = compute_classes_nav(DATA_DIR, classes_spec, start_date)

        # 将宽表 NAV 转换为长表
        nav_long = NAV.reset_index().melt(id_vars=["date"], var_name="asset_name", value_name="nv")
        nav_long["asset_alloc_name"] = alloc_name
        nav_long["creat_time"] = now
        # 重新排序字段
        nav_long = nav_long[["asset_alloc_name", "asset_name", "date", "nv", "creat_time"]]

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
        return JSONResponse(status_code=500, content={"detail": f"计算并保存净值时出错: {e}"})

    return {"ok": True, "message": f"配置 '{alloc_name}' 已成功保存"}


@app.get("/api/list-allocations")
def list_allocations():
    info_path = DATA_DIR / "asset_alloc_info.parquet"
    if not info_path.exists():
        return []
    df = pd.read_parquet(info_path)
    return sorted(df["asset_alloc_name"].unique().tolist())


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


# Registered by services.analytics_routes.
def post_efficient_frontier(req: FrontierRequest):
    nv_path = DATA_DIR / "asset_nv.parquet"
    if not nv_path.exists():
        return JSONResponse(status_code=404, content={"detail": "净值数据文件 asset_nv.parquet 不存在"})

    df = pd.read_parquet(nv_path)

    # 1. 筛选数据
    alloc_df = df[df["asset_alloc_name"] == req.alloc_name].copy()
    if alloc_df.empty:
        return JSONResponse(status_code=404, content={"detail": f"未找到名为 '{req.alloc_name}' 的配置的净值数据"})

    alloc_df['date'] = pd.to_datetime(alloc_df['date'])
    mask = (alloc_df['date'] >= pd.to_datetime(req.start_date)) & (alloc_df['date'] <= pd.to_datetime(req.end_date))
    alloc_df = alloc_df.loc[mask]

    if alloc_df.empty:
        return JSONResponse(status_code=400, content={"detail": "在选定日期区间内没有数据"})

    # 2. 准备收益率宽表
    nav_wide = alloc_df.pivot_table(index='date', columns='asset_name', values='nv').sort_index().dropna(axis=0, how='any')
    if len(nav_wide.index) < 2:
        return JSONResponse(status_code=400, content={"detail": "完整交集净值样本不足，无法计算有效前沿"})

    return_type = req.return_metric.get('type', 'simple')
    try:
        return_values = returns_from_nav_matrix(
            nav_wide.to_numpy(dtype=np.float64),
            return_type=return_type,
        )
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    returns_df = pd.DataFrame(
        return_values,
        index=nav_wide.index[1:],
        columns=nav_wide.columns,
    )

    # 3. 调用核心计算函数
    # Map constraints by asset order
    asset_names = list(nav_wide.columns)
    single_limits = []
    if req.constraints and isinstance(req.constraints.get('single_limits', None), dict):
        m = req.constraints['single_limits']
        for nm in asset_names:
            v = m.get(nm, None)
            lo = float(v.get('lo', 0.0)) if isinstance(v, dict) else 0.0
            hi = float(v.get('hi', 1.0)) if isinstance(v, dict) else 1.0
            single_limits.append((max(0.0, lo), min(1.0, hi)))
    else:
        single_limits = [(0.0, 1.0) for _ in asset_names]

    group_limits = {}
    if req.constraints and isinstance(req.constraints.get('group_limits', None), list):
        for g in req.constraints['group_limits']:
            assets = g.get('assets', [])
            idxs = tuple(i for i, nm in enumerate(asset_names) if nm in assets)
            if not idxs:
                continue
            lo = float(g.get('lo', 0.0))
            hi = float(g.get('hi', 1.0))
            group_limits[idxs] = (lo, hi)

    rounds = None
    if req.exploration and isinstance(req.exploration.get('rounds', None), list):
        rounds = []
        for r in req.exploration['rounds']:
            rounds.append({
                'samples': int(r.get('samples', 100)),
                'step': float(r.get('step', 0.5)),
                'buckets': int(r.get('buckets', 50)),
            })

    quant_step = None
    if req.quantization:
        try:
            v = req.quantization.get('step', None)
            quant_step = None if v in (None, 'none') else float(v)
        except Exception:
            quant_step = None

    use_refine = False
    refine_count = 0
    if req.refine:
        use_refine = bool(req.refine.get('use_slsqp', False))
        refine_count = int(req.refine.get('count', 0))

    results = calculate_efficient_frontier_exploration(
        asset_returns=returns_df,
        return_config=req.return_metric,
        risk_config=req.risk_metric,
        single_limits=single_limits,
        group_limits=group_limits,
        rounds=rounds,
        quantize_step=quant_step,
        use_slsqp_refine=use_refine,
        refine_count=refine_count,
        risk_free_rate=float(getattr(req, 'risk_free_rate', 0.0) or 0.0),
    )

    # 4. 数据净化，防止 NaN/Infinity 导致前端JSON解析或渲染失败
    def extract_value(obj):
        if obj is None:
            return None
        if isinstance(obj, (list, tuple)):
            val = obj
        else:
            val = obj.get("value")
        if not (isinstance(val, (list, tuple)) and len(val) == 2):
            return None
        x, y = val
        return (x, y)

    def is_finite_point(obj):
        v = extract_value(obj)
        return v is not None and math.isfinite(v[0]) and math.isfinite(v[1])

    clean_results = {
        "asset_names": results.get("asset_names", []),
        "scatter": [p for p in results.get("scatter", []) if is_finite_point(p)],
        "frontier": sorted([p for p in results.get("frontier", []) if is_finite_point(p)],
                           key=lambda o: extract_value(o)[0]),
        "max_sharpe": results.get("max_sharpe") if is_finite_point(results.get("max_sharpe")) else None,
        "min_variance": results.get("min_variance") if is_finite_point(results.get("min_variance")) else None,
        "max_return": results.get("max_return") if is_finite_point(results.get("max_return")) else None,
        "execution": results.get("execution"),
    }

    return clean_results


# ---------------- Strategy: compute weights and backtest ----------------

class StrategyClassItem(BaseModel):
    name: str
    weight: Optional[float] = None
    budget: Optional[float] = None


class StrategySpec(BaseModel):
    type: str  # fixed | risk_budget | target
    name: Optional[str] = None
    classes: List[StrategyClassItem]
    # rebalancing (optional)
    rebalance: Optional[Dict[str, Any]] = None  # {enabled, mode, which, N, unit, fixedInterval}
    # optional model config for dynamic recalculation on rebalance
    model: Optional[Dict[str, Any]] = None
    # risk budget params
    risk_metric: Optional[str] = None
    return_type: Optional[str] = None  # simple|log for risk calc
    confidence: Optional[float] = None
    days: Optional[int] = None
    window: Optional[int] = None
    # target params
    target: Optional[str] = None  # min_risk|max_return|max_sharpe|risk_min_given_return|return_max_given_risk
    return_metric: Optional[str] = None
    risk_free_rate: Optional[float] = None
    target_return: Optional[float] = None
    target_risk: Optional[float] = None
    # constraints
    constraints: Optional[Dict[str, Any]] = None


class ComputeWeightsRequest(BaseModel):
    alloc_name: str
    strategy: StrategySpec
    data_len: Optional[int] = None  # e.g., 30, 60, ... None=all
    window_mode: Optional[str] = None  # 'all'|'firstN'|'rollingN'


# Registered by services.strategy_routes.
def api_compute_weights(req: ComputeWeightsRequest):
    nv_path = DATA_DIR / "asset_nv.parquet"
    if not nv_path.exists():
        return JSONResponse(status_code=404, content={"detail": "净值数据文件 asset_nv.parquet 不存在"})
    df = pd.read_parquet(nv_path)
    df = df[df["asset_alloc_name"] == req.alloc_name]
    if df.empty:
        return JSONResponse(status_code=404, content={"detail": f"未找到名为 '{req.alloc_name}' 的配置的净值数据"})
    nav_wide = df.pivot_table(index='date', columns='asset_name', values='nv').sort_index()
    # Filter to requested classes order
    class_names = [c.name for c in req.strategy.classes]
    nav_wide = nav_wide[class_names].dropna(how='all').dropna(axis=0)
    # 窗口裁剪逻辑由下游 compute_* 函数处理，避免重复裁剪导致结果一致

    if req.strategy.type == 'fixed':
        if any(c.weight is None for c in req.strategy.classes):
            return JSONResponse(status_code=400, content={"detail": "固定权重必须逐项提供，禁止以等权补缺"})
        try:
            weights = normalize_explicit_weights(
                [float(c.weight) for c in req.strategy.classes]
            )
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return {"weights": weights, "execution": strategy_execution_audit()}

    if req.strategy.type == 'risk_budget':
        budgets = [float(c.budget or 0.0) for c in req.strategy.classes]
        # risk config
        risk_cfg = {"metric": req.strategy.risk_metric or "vol"}
        if req.strategy.risk_metric in {"annual_vol", "ewm_vol"}:
            if req.strategy.days is not None:
                risk_cfg["days"] = int(req.strategy.days)
        if req.strategy.risk_metric == "ewm_vol":
            if req.strategy.window is not None:
                risk_cfg["window"] = int(req.strategy.window)
            if req.strategy.confidence is not None:  # not used here; kept for interface consistency
                pass
        if req.strategy.risk_metric in {"var", "es"}:
            if req.strategy.confidence is not None:
                risk_cfg["confidence"] = float(req.strategy.confidence)
        weights = compute_risk_budget_weights(nav_wide, risk_cfg, budgets, window_len=req.data_len,
                                              window_mode=(req.window_mode or 'firstN'))
        return {"weights": weights, "execution": strategy_execution_audit()}

    if req.strategy.type == 'target':
        risk_cfg = {"metric": req.strategy.risk_metric or "vol"}
        if req.strategy.risk_metric in {"annual_vol", "ewm_vol"} and req.strategy.days is not None:
            risk_cfg["days"] = int(req.strategy.days)
        if req.strategy.risk_metric == "ewm_vol" and req.strategy.window is not None:
            risk_cfg["window"] = int(req.strategy.window)
        if req.strategy.risk_metric in {"var", "es"} and req.strategy.confidence is not None:
            risk_cfg["confidence"] = float(req.strategy.confidence)
        ret_cfg = {"metric": req.strategy.return_metric or "annual", "days": int(req.strategy.days or 252)}
        # map constraints
        asset_names = list(nav_wide.columns)
        single_limits: List[Tuple[float, float]] = [(0.0, 1.0) for _ in asset_names]
        group_limits: Dict[Tuple[int, ...], Tuple[float, float]] = {}
        if req.strategy.constraints and isinstance(req.strategy.constraints.get('single_limits', None), dict):
            sl = req.strategy.constraints['single_limits']
            single_limits = []
            for nm in asset_names:
                v = sl.get(nm, {})
                lo = float(v.get('lo', 0.0)) if isinstance(v, dict) else 0.0
                hi = float(v.get('hi', 1.0)) if isinstance(v, dict) else 1.0
                single_limits.append((lo, hi))
        if req.strategy.constraints and isinstance(req.strategy.constraints.get('group_limits', None), list):
            for g in req.strategy.constraints['group_limits']:
                assets = g.get('assets', [])
                idxs = tuple(i for i, nm in enumerate(asset_names) if nm in assets)
                if idxs:
                    lo = float(g.get('lo', 0.0));
                    hi = float(g.get('hi', 1.0))
                    group_limits[idxs] = (lo, hi)

        weights = compute_target_weights(
            nav_wide,
            ret_cfg,
            risk_cfg,
            target=req.strategy.target or 'min_risk',
            window_len=req.data_len,
            window_mode=(req.window_mode or 'firstN'),
            single_limits=single_limits,
            group_limits=group_limits,
            risk_free_rate=float(req.strategy.risk_free_rate or 0.0),
            target_return=req.strategy.target_return,
            target_risk=req.strategy.target_risk,
        )
        return {"weights": weights, "execution": strategy_execution_audit()}

    return JSONResponse(status_code=400, content={"detail": "未知策略类型"})


class BacktestRequest(BaseModel):
    alloc_name: str
    start_date: Optional[str] = None
    strategies: List[StrategySpec]


# Registered by services.strategy_routes.
def api_backtest(req: BacktestRequest):
    nv_path = DATA_DIR / "asset_nv.parquet"
    if not nv_path.exists():
        return JSONResponse(status_code=404, content={"detail": "净值数据文件 asset_nv.parquet 不存在"})
    df = pd.read_parquet(nv_path)
    df = df[df["asset_alloc_name"] == req.alloc_name]
    if df.empty:
        return JSONResponse(status_code=404, content={"detail": f"未找到名为 '{req.alloc_name}' 的配置的净值数据"})
    nav_wide = df.pivot_table(index='date', columns='asset_name', values='nv').sort_index()

    # Build strategies weights in class order
    class_names = list(nav_wide.columns)
    strat_list = []
    for s in req.strategies:
        cls_map = {c.name: c for c in s.classes}
        weights = [float(cls_map.get(n).weight) if (n in cls_map and cls_map[n].weight is not None) else 0.0 for n in
                   class_names]
        # pass rebalance info forward (ensure dict form)
        rb = s.rebalance if isinstance(s.rebalance, dict) else None
        sdict = {"name": s.name or s.type, "type": s.type, "weights": weights, "rebalance": rb,
                 "classes": [c.dict() for c in s.classes]}
        if s.model:
            sdict["model"] = s.model
        strat_list.append(sdict)

    res = backtest_portfolio(nav_wide, strat_list, start_date=req.start_date)
    return res


# --------- Compute schedule weights for recalc ahead of backtest ---------
from concurrent.futures import ProcessPoolExecutor, as_completed


class ComputeScheduleRequest(BaseModel):
    alloc_name: str
    start_date: Optional[str] = None
    strategy: StrategySpec


def _compute_weight_for_date(args: Dict[str, Any]) -> Dict[str, Any]:
    import pandas as pd
    from strategy import compute_risk_budget_weights, compute_target_weights
    nav_split = args['nav_split']
    nav = pd.DataFrame(nav_split['data'], index=pd.to_datetime(nav_split['index']), columns=nav_split['columns'])
    up_to = pd.to_datetime(args['date'])
    nav = nav.loc[nav.index <= up_to]
    stype = args['stype']
    model = args['model'] or {}
    # windowing
    window_mode = model.get('window_mode') or 'rollingN'
    n = int(model.get('data_len') or 0)
    if window_mode != 'all' and n > 0:
        nav = nav.tail(n)
    asset_names = list(nav.columns)
    if stype == 'risk_budget':
        budgets = args['budgets']
        risk_cfg = {'metric': model.get('risk_metric') or 'vol'}
        if model.get('days') is not None:
            risk_cfg['days'] = int(model.get('days'))
        if model.get('window') is not None:
            risk_cfg['window'] = int(model.get('window'))
        if model.get('confidence') is not None:
            risk_cfg['confidence'] = float(model.get('confidence'))
        w = compute_risk_budget_weights(nav, risk_cfg, budgets, window_len=None)
        return {'date': args['date'], 'weights': [float(x) for x in w]}
    else:  # target
        ret_cfg = {
            'metric': model.get('return_metric') or 'annual',
            'days': int(model.get('days') or 252),
            'alpha': model.get('ret_alpha'),
            'window': model.get('ret_window'),
        }
        risk_cfg = {
            'metric': model.get('risk_metric') or 'vol',
            'days': model.get('risk_days'),
            'alpha': model.get('risk_alpha'),
            'window': model.get('risk_window'),
            'confidence': model.get('risk_confidence'),
        }
        # constraints map
        single_limits = []
        sl = (model.get('constraints') or {}).get('single_limits', {})
        for nm in asset_names:
            v = sl.get(nm, {})
            lo = float(v.get('lo', 0.0)) if isinstance(v, dict) else 0.0
            hi = float(v.get('hi', 1.0)) if isinstance(v, dict) else 1.0
            single_limits.append((lo, hi))
        group_limits = {}
        for g in (model.get('constraints') or {}).get('group_limits', []) or []:
            assets = g.get('assets', [])
            idxs = tuple(i for i, nm in enumerate(asset_names) if nm in assets)
            if idxs:
                group_limits[idxs] = (float(g.get('lo', 0.0)), float(g.get('hi', 1.0)))
        w = compute_target_weights(
            nav, ret_cfg, risk_cfg,
            target=str(model.get('target') or 'min_risk'),
            window_len=None, window_mode=None,
            single_limits=single_limits, group_limits=group_limits,
            risk_free_rate=float(model.get('risk_free_rate') or 0.0),
            target_return=model.get('target_return'), target_risk=model.get('target_risk'),
            use_exploration=False,
        )
        return {'date': args['date'], 'weights': [float(x) for x in w]}


# Registered by services.strategy_routes.
def api_compute_schedule_weights(req: ComputeScheduleRequest):
    nv_path = DATA_DIR / "asset_nv.parquet"
    if not nv_path.exists():
        return JSONResponse(status_code=404, content={"detail": "净值数据文件 asset_nv.parquet 不存在"})
    df = pd.read_parquet(nv_path)
    df = df[df["asset_alloc_name"] == req.alloc_name]
    if df.empty:
        return JSONResponse(status_code=404, content={"detail": f"未找到名为 '{req.alloc_name}' 的配置的净值数据"})
    nav_wide = df.pivot_table(index='date', columns='asset_name', values='nv').sort_index()
    if req.start_date:
        nav_wide = nav_wide[nav_wide.index >= pd.to_datetime(req.start_date)]
    # align to classes order
    class_names = [c.name for c in req.strategy.classes]
    nav_wide = nav_wide[class_names].dropna(how='all').dropna(axis=0)
    asset_names = list(nav_wide.columns)

    rb = req.strategy.rebalance or {}
    if not rb.get('enabled') or not rb.get('recalc'):
        # only compute one snapshot at start
        dates = [nav_wide.index[0].date().isoformat()]
    else:
        mode = str(rb.get('mode', 'monthly'))
        which = str(rb.get('which', 'nth'))
        N = int(rb.get('N', 1))
        unit = str(rb.get('unit', 'trading'))
        fixed_interval = int(rb.get('fixedInterval', 20)) if mode == 'fixed' else None
        rset = gen_rebalance_dates(nav_wide.index, mode, N=N, which=which, unit=unit, fixed_interval=fixed_interval)
        rset = sorted([d for d in rset if d in nav_wide.index])
        if not rset or rset[0] != nav_wide.index[0]:
            rset = [nav_wide.index[0]] + rset
        dates = [d.date().isoformat() for d in rset]

    # Prepare args for processes
    nav_split = {'index': [d.isoformat() for d in nav_wide.index], 'columns': asset_names,
                 'data': nav_wide.values.tolist()}
    tasks = []
    if req.strategy.type == 'risk_budget':
        budgets = [float(c.budget or 0.0) for c in req.strategy.classes]
        model = {
            'risk_metric': req.strategy.risk_metric or 'vol',
            'days': req.strategy.days,
            'window': req.strategy.window,
            'confidence': req.strategy.confidence,
            # window config for workers
            'window_mode': (req.strategy.return_metric or 'rollingN'),
            'data_len': None,
        }
        model.update({k: v for k, v in (req.strategy.constraints or {}).items()})
        for d in dates:
            tasks.append({'date': d, 'nav_split': nav_split, 'stype': 'risk_budget',
                          'model': {'risk_metric': model['risk_metric'], 'days': model.get('days'),
                                    'window': model.get('window'), 'confidence': model.get('confidence'),
                                    'window_mode': req.strategy.return_metric, 'data_len': None}, 'budgets': budgets})
    else:
        model = {
            'target': req.strategy.target,
            'return_metric': req.strategy.return_metric or 'annual',
            'return_type': req.strategy.return_type or 'simple',
            'days': req.strategy.days or 252,
            'ret_alpha': None,
            'ret_window': None,
            'risk_metric': req.strategy.risk_metric or 'vol',
            'risk_days': req.strategy.days,
            'risk_alpha': None,
            'risk_window': req.strategy.window,
            'risk_confidence': req.strategy.confidence,
            'risk_free_rate': req.strategy.risk_free_rate or 0.0,
            'constraints': req.strategy.constraints or {},
            'window_mode': (req.strategy.return_metric or 'rollingN'),
            'data_len': None,
            'target_return': req.strategy.target_return,
            'target_risk': req.strategy.target_risk,
        }
        for d in dates:
            tasks.append({'date': d, 'nav_split': nav_split, 'stype': 'target', 'model': model, 'budgets': None})

    # Parallel compute with fallback sequential on error
    results: List[Dict[str, Any]] = []
    try:
        max_workers = min(4, (os.cpu_count() or 2))
    except Exception:
        max_workers = 2
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_compute_weight_for_date, t) for t in tasks]
            for f in as_completed(futs):
                results.append(f.result())
    except Exception:
        # fallback sequential
        results = [_compute_weight_for_date(t) for t in tasks]
    # order by date
    results.sort(key=lambda x: x['date'])
    return {"asset_names": asset_names, "dates": [r['date'] for r in results],
            "weights": [r['weights'] for r in results]}


# Registered by services.strategy_routes.
def api_default_start(alloc_name: str):
    """Return the default backtest start date for an allocation: 
    take the maximum of each asset's first available NAV date (ensures all series have data).
    """
    nv_path = DATA_DIR / "asset_nv.parquet"
    if not nv_path.exists():
        return JSONResponse(status_code=404, content={"detail": "净值数据文件 asset_nv.parquet 不存在"})
    df = pd.read_parquet(nv_path)
    df = df[df["asset_alloc_name"] == alloc_name]
    if df.empty:
        return JSONResponse(status_code=404, content={"detail": f"未找到名为 '{alloc_name}' 的配置的净值数据"})
    df = df.dropna(subset=["date"]).copy()
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime
    first_dates = df.groupby("asset_name")["date"].min()
    if first_dates.empty:
        return {"default_start": None, "count": 0}
    default_start_ts = first_dates.max()
    default_start = default_start_ts.date().isoformat()
    # total available trading days count (from earliest overall start), or from default_start?
    nav_wide = df.pivot_table(index='date', columns='asset_name', values='nv').sort_index()
    nav_wide = nav_wide[nav_wide.index >= default_start_ts]
    count = int(len(nav_wide.index))
    return {"default_start": default_start, "count": count}


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

    uvicorn.run("app:app", host="0.0.0.0", port=args.port, reload=True)
