from __future__ import annotations

import json
import math
import re
import sys
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

from optimizer import calculate_efficient_frontier_exploration
from backtest_engine import backtest_portfolio, gen_rebalance_dates
from fit import compute_rolling_corr_classes, compute_class_consistency
from strategy import compute_risk_budget_weights, compute_target_weights
from fit import ClassSpec, ETFSpec, compute_classes_nav, compute_rolling_corr


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
    corr: List[List[float]]
    corr_labels: List[str]
    metrics: List[dict]
    consistency: List[dict]


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


app = FastAPI(title="Risk Parity Backend")

# 允许前端本地开发联调（Vite 默认 5173 端口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*",
    ],
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


@app.get("/api/health")
def health():
    return {"ok": True}


def _round2(n: float) -> float:
    return round(n + 1e-8, 2)


@app.post("/api/risk-parity/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    """
    风险平价求解（使用 data/etf_daily_df.parquet 的 adj_nav 复权净值计算日收益 → 协方差矩阵）：
    - 支持目标风险预算（来自前端 riskContribution，占比合计 100）；若未提供则等预算。
    - 非负权重，权重和=1；若 maxLeverage>0，则线性放大到 1+maxLeverage。
    - 目前仅实现基于波动率的风险度量（riskMetric 参数暂不影响计算）。
    """

    # 读取并准备收益序列
    pq = DATA_DIR / "etf_daily_df.parquet"
    if not pq.exists():
        # 回退为占比分配
        rc = [max(0.0, float(x.riskContribution)) for x in req.etfs]
        s = sum(rc)
        if s <= 0:
            return SolveResponse(weights=[0.0 for _ in rc])
        base = np.array(rc, dtype=float) / s
        scale = 1.0 + max(0.0, float(req.maxLeverage))
        return SolveResponse(weights=[_round2(float(w * 100 * scale)) for w in base])

    df = pd.read_parquet(pq)
    # 规范列
    for c in ["adj_nav", "ts_code", "name", "date"]:
        if c not in df.columns:
            raise ValueError(f"parquet 缺少必要列：{c}")
    df = df[["ts_code", "name", "date", "adj_nav"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "adj_nav"]).sort_values("date")

    # 根据前端给的 code/name 匹配 ts_code/name
    series_list: List[pd.Series] = []
    labels: List[str] = []

    def _code_nosfx(x: str) -> str:
        xs = (x or "").strip()
        if "." in xs:
            return xs.split(".")[0]
        return xs

    for item in req.etfs:
        code = (item.code or "").strip()
        code_ns = _code_nosfx(code)
        name = (item.name or "").strip()
        ts = df["ts_code"].astype(str)
        nm = df["name"].astype(str)
        sub = df[(ts == code) | (ts.str.split(".").str[0] == code_ns) | (nm == name)]
        if sub.empty and code_ns:
            # 松匹配：code 片段
            sub = df[ts.str.contains(code_ns, na=False)]
        if sub.empty:
            continue
        # 以日期为索引，计算日收益率（使用复权净值）
        s = (
            sub.sort_values("date")["adj_nav"].reset_index(drop=True).pct_change().dropna()
        )
        if s.empty:
            continue
        # 用原始 ts_code 作为列标签，便于对齐
        label = str(sub.iloc[0]["ts_code"])  # 使用 ts_code 作为内部标签
        s.index = range(len(s))  # 统一索引，后续用等长拼接
        series_list.append(s)
        labels.append(label)

    n = len(series_list)
    if n == 0:
        # 回退：预算分配
        rc = [max(0.0, float(x.riskContribution)) for x in req.etfs]
        s = sum(rc)
        if s <= 0:
            return SolveResponse(weights=[0.0 for _ in rc])
        base = np.array(rc, dtype=float) / s
        scale = 1.0 + max(0.0, float(req.maxLeverage))
        return SolveResponse(weights=[_round2(float(w * 100 * scale)) for w in base])

    # 对齐为相同长度：取最短序列长度
    min_len = min(len(s) for s in series_list)
    X = np.column_stack([s.iloc[-min_len:].to_numpy() for s in series_list])
    # 协方差矩阵
    S = np.cov(X, rowvar=False, ddof=1)

    # 目标预算 b, 构建预算向量（与 labels 同序），同时兼容 code 无后缀/带后缀；name 作为兜底
    budget_map = {}
    for it in req.etfs:
        c = (it.code or "").strip()
        budget_map[c] = float(it.riskContribution)
        budget_map[_code_nosfx(c)] = float(it.riskContribution)
        budget_map[(it.name or "").strip()] = float(it.riskContribution)
    b = []
    for lab in labels:
        b.append(float(budget_map.get(lab, 0.0) or 0.0))
    b = np.array(b, dtype=float)
    if b.sum() <= 0:
        b = np.ones(n, dtype=float)
    b = b / b.sum()

    # 数值稳定：若协方差半正定，做轻微对角线缩减
    try:
        # 尝试 Cholesky 以检测正定性
        np.linalg.cholesky(S + 1e-12 * np.eye(n))
    except np.linalg.LinAlgError:
        S = S + 1e-6 * np.eye(n)

    # 固定点迭代：w_new ∝ b / (S w)
    eps = 1e-12
    w = b.copy()
    w = np.maximum(w, eps)
    w = w / w.sum()
    for _ in range(5000):
        Sw = S @ w
        denom = np.maximum(Sw, eps)
        w_new = b / denom
        w_new = np.maximum(w_new, eps)
        w_new = w_new / w_new.sum()
        if np.linalg.norm(w_new - w, ord=1) < 1e-8:
            w = w_new
            break
        w = 0.7 * w_new + 0.3 * w  # 适度松弛以增强稳定性
    # 计算风险贡献以检验贴合度（非必须）
    Sw = S @ w
    port_var = float(w @ Sw)
    if port_var > 0:
        rc = (w * Sw) / port_var
        # 若偏差很大，做最后一次牛顿修正（可选，略）

    # Map weights back to original input order; unknown assets get 0
    weight_map = {lab: float(wi) for lab, wi in zip(labels, w)}
    out = []
    for it in req.etfs:
        code = (it.code or "").strip()
        key = code if code in weight_map else _code_nosfx(code)
        if key not in weight_map:
            key = (it.name or "").strip()
        out.append(weight_map.get(key, 0.0))

    # 杠杆缩放
    scale = 1.0 + max(0.0, float(req.maxLeverage))
    out = [float(x * 100 * scale) for x in out]
    return SolveResponse(weights=[_round2(x) for x in out])


@app.post("/api/fit-classes", response_model=FitResponse)
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
    corr_vals = [[finite_or_none(float(v)) or 0.0 for v in row] for row in corr.values.tolist()]
    metrics_out = []
    for name, row in metrics.iterrows():
        metrics_out.append({
            "name": str(name),
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
    return FitResponse(dates=dates, navs=navs, corr=corr_vals, corr_labels=corr_labels, metrics=metrics_out,
                       consistency=cons_out)


@app.post("/api/rolling-corr", response_model=RollingResponse)
def rolling_corr(req: RollingRequest):
    try:
        start = pd.to_datetime(req.startDate)
    except Exception:
        raise ValueError("startDate 格式错误，应为 YYYY-MM-DD")
    etfs = [ETFSpec(code=e.code, name=e.name, weight=float(e.weight)) for e in req.etfs]
    idx, series_map, metrics = compute_rolling_corr(DATA_DIR, etfs, start, int(req.window),
                                                    req.targetCode,
                                                    req.targetName)
    dates = [d.strftime("%Y-%m-%d") for d in idx]
    # 清理 NaN/Inf
    safe_series = {
        k: [float(x) if isinstance(x, (int, float)) and (x == x) and abs(x) != float('inf') else 0.0 for x in v] for
        k, v in series_map.items()}
    for m in metrics:
        for k in list(m.keys()):
            if k == 'name':
                continue
            v = m[k]
            try:
                if not (isinstance(v, (int, float)) and v == v and abs(v) != float('inf')):
                    m[k] = 0.0
            except Exception:
                m[k] = 0.0
    return RollingResponse(dates=dates, series=safe_series, metrics=metrics)


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
    dates = [d.strftime("%Y-%m-%d") for d in idx]
    safe_series = {
        k: [float(x) if isinstance(x, (int, float)) and (x == x) and abs(x) != float('inf') else 0.0 for x in v] for
        k, v in series_map.items()}
    for m in metrics:
        for k in list(m.keys()):
            if k == 'name':
                continue
            v = m[k]
            try:
                if not (isinstance(v, (int, float)) and v == v and abs(v) != float('inf')):
                    m[k] = 0.0
            except Exception:
                m[k] = 0.0
    return RollingResponse(dates=dates, series=safe_series, metrics=metrics)


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


@app.post("/api/efficient-frontier")
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
    nav_wide = alloc_df.pivot_table(index='date', columns='asset_name', values='nv').sort_index()

    return_type = req.return_metric.get('type', 'simple')
    if return_type == 'log':
        returns_df = np.log(nav_wide / nav_wide.shift(1)).dropna()
    else:
        returns_df = nav_wide.pct_change().dropna()

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


@app.post("/api/strategy/compute-weights")
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
        weights = []
        for c in req.strategy.classes:
            w = 0.0 if c.weight is None else float(c.weight)
            weights.append(w)
        s = sum(weights)
        if s <= 0:
            n = len(weights)
            weights = [1.0 / n for _ in range(n)]
        else:
            weights = [w / s for w in weights]
        return {"weights": weights}

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
        return {"weights": weights}

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
        return {"weights": weights}

    return JSONResponse(status_code=400, content={"detail": "未知策略类型"})


class BacktestRequest(BaseModel):
    alloc_name: str
    start_date: Optional[str] = None
    strategies: List[StrategySpec]


@app.post("/api/strategy/backtest")
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


@app.post("/api/strategy/compute-schedule-weights")
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


@app.get("/api/strategy/default-start")
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
ETF_INFO_FILENAMES = ("etf_info_df.parquet",)


DEMO_ETF_INFO_ROWS = [
    {
        "ts_code": "510050.SH",
        "code": "510050",
        "name": "上证50ETF",
        "management": "华夏基金",
        "custodian": "中国银行",
        "trustee": "中国银行",
        "fund_type": "场内ETF",
        "type": "公募基金",
        "invest_type": "宽基指数",
        "market": "上交所",
        "status": "上市交易",
        "issue_amount": 450000,
        "m_fee": 0.5,
        "c_fee": 0.1,
        "exp_return": 12.3,
        "duration_year": 18.0,
        "list_date": "2004-02-23",
        "found_date": "2004-02-05",
        "issue_date": "2004-02-10",
        "due_date": None,
        "purc_startdate": "2004-02-12",
        "redm_startdate": "2004-02-12",
    },
    {
        "ts_code": "512000.SH",
        "code": "512000",
        "name": "券商ETF",
        "management": "国泰基金",
        "custodian": "招商银行",
        "trustee": "招商银行",
        "fund_type": "场内ETF",
        "type": "公募基金",
        "invest_type": "行业指数",
        "market": "上交所",
        "status": "上市交易",
        "issue_amount": 160000,
        "m_fee": 0.6,
        "c_fee": 0.2,
        "exp_return": 11.5,
        "duration_year": 9.0,
        "list_date": "2015-04-23",
        "found_date": "2015-04-15",
        "issue_date": "2015-04-20",
        "due_date": None,
        "purc_startdate": "2015-04-21",
        "redm_startdate": "2015-04-21",
    },
    {
        "ts_code": "159915.SZ",
        "code": "159915",
        "name": "创业板ETF",
        "management": "易方达基金",
        "custodian": "中国建设银行",
        "trustee": "中国建设银行",
        "fund_type": "场内ETF",
        "type": "公募基金",
        "invest_type": "宽基指数",
        "market": "深交所",
        "status": "上市交易",
        "issue_amount": 320000,
        "m_fee": 0.5,
        "c_fee": 0.1,
        "exp_return": 15.2,
        "duration_year": 12.0,
        "list_date": "2011-06-13",
        "found_date": "2011-05-20",
        "issue_date": "2011-05-25",
        "due_date": None,
        "purc_startdate": "2011-05-26",
        "redm_startdate": "2011-05-26",
    },
    {
        "ts_code": "560800.BJ",
        "code": "560800",
        "name": "科创精选ETF",
        "management": "工银瑞信",
        "custodian": "交通银行",
        "trustee": "交通银行",
        "fund_type": "场内ETF",
        "type": "公募基金",
        "invest_type": "科技创新",
        "market": "北交所",
        "status": "上市交易",
        "issue_amount": 38000,
        "m_fee": 0.8,
        "c_fee": 0.15,
        "exp_return": 18.4,
        "duration_year": 3.0,
        "list_date": "2022-08-30",
        "found_date": "2022-08-10",
        "issue_date": "2022-08-18",
        "due_date": None,
        "purc_startdate": "2022-08-19",
        "redm_startdate": "2022-08-19",
    },
    {
        "ts_code": "513050.SH",
        "code": "513050",
        "name": "中概互联网ETF",
        "management": "华夏基金",
        "custodian": "中国银行",
        "trustee": "中国银行",
        "fund_type": "场内ETF",
        "type": "公募基金",
        "invest_type": "主题策略",
        "market": "上交所",
        "status": "上市交易",
        "issue_amount": 240000,
        "m_fee": 0.6,
        "c_fee": 0.15,
        "exp_return": 16.8,
        "duration_year": 6.0,
        "list_date": "2017-09-11",
        "found_date": "2017-08-25",
        "issue_date": "2017-09-05",
        "due_date": None,
        "purc_startdate": "2017-09-06",
        "redm_startdate": "2017-09-06",
    },
    {
        "ts_code": "588000.SH",
        "code": "588000",
        "name": "科创50ETF",
        "management": "华夏基金",
        "custodian": "中国银行",
        "trustee": "中国银行",
        "fund_type": "场内ETF",
        "type": "公募基金",
        "invest_type": "科技创新",
        "market": "上交所",
        "status": "上市交易",
        "issue_amount": 280000,
        "m_fee": 0.6,
        "c_fee": 0.15,
        "exp_return": 17.5,
        "duration_year": 4.0,
        "list_date": "2019-11-05",
        "found_date": "2019-10-15",
        "issue_date": "2019-10-25",
        "due_date": None,
        "purc_startdate": "2019-10-28",
        "redm_startdate": "2019-10-28",
    },
    {
        "ts_code": "159949.SZ",
        "code": "159949",
        "name": "创业板50ETF",
        "management": "嘉实基金",
        "custodian": "中国工商银行",
        "trustee": "中国工商银行",
        "fund_type": "场内ETF",
        "type": "公募基金",
        "invest_type": "宽基指数",
        "market": "深交所",
        "status": "上市交易",
        "issue_amount": 210000,
        "m_fee": 0.55,
        "c_fee": 0.1,
        "exp_return": 14.8,
        "duration_year": 8.0,
        "list_date": "2013-09-30",
        "found_date": "2013-09-02",
        "issue_date": "2013-09-12",
        "due_date": None,
        "purc_startdate": "2013-09-13",
        "redm_startdate": "2013-09-13",
    },
    {
        "ts_code": "513500.SH",
        "code": "513500",
        "name": "标普500ETF",
        "management": "南方基金",
        "custodian": "中国银行",
        "trustee": "中国银行",
        "fund_type": "场内ETF",
        "type": "公募基金",
        "invest_type": "海外指数",
        "market": "上交所",
        "status": "上市交易",
        "issue_amount": 180000,
        "m_fee": 0.7,
        "c_fee": 0.12,
        "exp_return": 9.6,
        "duration_year": 7.0,
        "list_date": "2013-06-28",
        "found_date": "2013-06-06",
        "issue_date": "2013-06-18",
        "due_date": None,
        "purc_startdate": "2013-06-19",
        "redm_startdate": "2013-06-19",
    },
    {
        "ts_code": "560010.BJ",
        "code": "560010",
        "name": "绿色发展ETF",
        "management": "中金基金",
        "custodian": "中国农业银行",
        "trustee": "中国农业银行",
        "fund_type": "场内ETF",
        "type": "公募基金",
        "invest_type": "ESG主题",
        "market": "北交所",
        "status": "存续",
        "issue_amount": 42000,
        "m_fee": 0.75,
        "c_fee": 0.18,
        "exp_return": 13.2,
        "duration_year": 2.0,
        "list_date": "2021-12-15",
        "found_date": "2021-12-01",
        "issue_date": "2021-12-08",
        "due_date": None,
        "purc_startdate": "2021-12-09",
        "redm_startdate": "2021-12-09",
    },
]


_COMMON_NULLS = {"", "nan", "none", "null", "-", "--", "暂无", "缺失"}


def _normalize_numeric_value(
    value: Any,
    *,
    percent: bool = False,
    amount_to_wan: bool = False,
    strip_suffixes: Optional[Tuple[str, ...]] = None,
) -> Optional[float]:
    """Coerce messy numeric strings to float.

    - percent=True will drop trailing '%' but keep the numeric in percentage points (0.5 -> 0.5).
    - amount_to_wan=True will normalise Chinese units to "万" (e.g. 3亿 -> 30000).
    - strip_suffixes removes textual units such as '年'.
    """

    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            if math.isnan(value):
                return None
        except Exception:
            pass
        return float(value)

    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in _COMMON_NULLS:
        return None

    if percent and text.endswith("%"):
        text = text[:-1]

    if strip_suffixes:
        for suffix in strip_suffixes:
            if text.endswith(suffix):
                text = text[: -len(suffix)]
                break

    multiplier = 1.0
    if amount_to_wan:
        unit_map = {
            "亿元": 10000.0,
            "亿": 10000.0,
            "万元": 1.0,
            "万": 1.0,
            "元": 0.0001,
        }
        for suffix, factor in unit_map.items():
            if text.endswith(suffix):
                text = text[: -len(suffix)]
                multiplier = factor
                break

    text = text.replace(",", "").replace("，", "").replace(" ", "")
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    try:
        number = float(match.group())
    except ValueError:
        return None
    return number * multiplier


@lru_cache(maxsize=1)
def _cached_etf_info_df(_mtime: float) -> pd.DataFrame:  # noqa: ARG001
    """Load the ETF info DataFrame from parquet with caching."""
    for fname in ETF_INFO_FILENAMES:
        path = DATA_DIR / fname
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
            if not isinstance(df, pd.DataFrame):
                continue
            df = df.copy()
            df.columns = [str(c) for c in df.columns]
            date_cols = [
                "found_date",
                "due_date",
                "list_date",
                "issue_date",
                "delist_date",
                "purc_startdate",
                "redm_startdate",
            ]
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            numeric_cols = [
                "issue_amount",
                "m_fee",
                "c_fee",
                "duration_year",
                "p_value",
                "min_amount",
                "exp_return",
            ]
            for col in numeric_cols:
                if col in df.columns:
                    if col in {"issue_amount", "min_amount"}:
                        df[col] = df[col].apply(lambda x: _normalize_numeric_value(x, amount_to_wan=True))
                    elif col in {"m_fee", "c_fee", "exp_return"}:
                        df[col] = df[col].apply(lambda x: _normalize_numeric_value(x, percent=True))
                    elif col == "duration_year":
                        df[col] = df[col].apply(lambda x: _normalize_numeric_value(x, strip_suffixes=("年", "yrs", "year", "years")))
                    else:
                        df[col] = df[col].apply(_normalize_numeric_value)
            return df
        except Exception:
            continue
    return pd.DataFrame()


def _build_demo_etf_info_df() -> pd.DataFrame:
    if not DEMO_ETF_INFO_ROWS:
        return pd.DataFrame()
    df = pd.DataFrame(DEMO_ETF_INFO_ROWS)
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    date_cols = [
        "found_date",
        "due_date",
        "list_date",
        "issue_date",
        "delist_date",
        "purc_startdate",
        "redm_startdate",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    numeric_cols = [
        "issue_amount",
        "m_fee",
        "c_fee",
        "duration_year",
        "p_value",
        "min_amount",
        "exp_return",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _ensure_demo_etf_info_df() -> pd.DataFrame:
    df = _build_demo_etf_info_df()
    if df.empty:
        return df
    path = DATA_DIR / "etf_info_df.parquet"
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            try:
                df.to_parquet(path, index=False)
            except Exception:
                pass
    except Exception:
        pass
    return df


def _load_etf_info_df() -> pd.DataFrame:
    mtimes = []
    for fname in ETF_INFO_FILENAMES:
        path = DATA_DIR / fname
        if path.exists():
            mtimes.append(path.stat().st_mtime)
    if not mtimes:
        return _ensure_demo_etf_info_df()
    return _cached_etf_info_df(max(mtimes))


def _load_universe() -> List[dict]:
    """Load ETF universe from JSON or Parquet under data/.
    Priority: etf_universe.json -> etf_info_df.parquet -> empty list
    Expected fields: ts_code/code and name
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DATA_DIR / "etf_universe.json"
    pq_path = DATA_DIR / "etf_info_df.parquet"
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
def _cached_universe_with_mtime(mtime: float) -> List[dict]:  # noqa: ARG001
    return _load_universe()


def _get_universe() -> List[dict]:
    # Invalidate cache when files change
    mtimes = []
    for fname in ("etf_universe.json", "etf_info_df.parquet"):
        p = DATA_DIR / fname
        if p.exists():
            mtimes.append(p.stat().st_mtime)
    m = max(mtimes) if mtimes else 0.0
    return _cached_universe_with_mtime(m)


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
    return JSONResponse({"items": items, "total": total, "page": page, "page_size": page_size})


def _safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _value_distribution(series: Optional[pd.Series], top_n: Optional[int] = None) -> List[Dict[str, Any]]:
    if series is None:
        return []
    clean = series.fillna("未知").astype(str)
    clean = clean.replace({"": "未知", "nan": "未知"})
    counts = clean.value_counts()
    if top_n is not None:
        counts = counts.head(top_n)
    return [{"name": str(idx), "value": int(val)} for idx, val in counts.items()]


def _build_list_trend_series(df: pd.DataFrame) -> List[Dict[str, Any]]:
    list_trend: List[Dict[str, Any]] = []
    if "list_date" not in df.columns or df.empty:
        return list_trend

    working = df.dropna(subset=["list_date"]).copy()
    if working.empty:
        return list_trend

    working["year"] = working["list_date"].dt.year
    year_counts = working.groupby("year").size().sort_index()
    if "issue_amount" in working.columns:
        year_issue = working.groupby("year")["issue_amount"].sum()
    else:
        year_issue = pd.Series(dtype=float)

    for year, count in year_counts.items():
        list_trend.append(
            {
                "year": int(year),
                "count": int(count),
                "total_issue_amount": _safe_float(year_issue.get(year)) if not year_issue.empty else None,
            }
        )

    return list_trend


def _build_list_trend_filters(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "type": _value_distribution(df.get("type"), None),
        "invest_type": _value_distribution(df.get("invest_type"), None),
        "fund_type": _value_distribution(df.get("fund_type"), None),
        "management": _value_distribution(df.get("management"), None),
    }


@app.get("/api/etf/analytics")
def etf_analytics():
    df = _load_etf_info_df()
    if df.empty:
        return JSONResponse(status_code=404, content={"detail": "未找到 etf_info_df 数据文件"})

    working = df.copy()

    issue_amount_total = _safe_float(working.get("issue_amount").sum()) if "issue_amount" in working.columns else None
    avg_m_fee = _safe_float(working.get("m_fee").mean()) if "m_fee" in working.columns else None
    avg_c_fee = _safe_float(working.get("c_fee").mean()) if "c_fee" in working.columns else None
    avg_exp_return = _safe_float(working.get("exp_return").mean()) if "exp_return" in working.columns else None
    avg_duration_year = _safe_float(working.get("duration_year").mean()) if "duration_year" in working.columns else None

    status_series = working.get("status")
    active_count: Optional[int] = None
    if status_series is not None:
        clean_status = status_series.fillna("未知").astype(str)
        inactive_keywords = ("终止", "退市", "清盘", "暂停", "到期")
        active_mask = ~clean_status.str.contains("|".join(inactive_keywords), case=False, na=False)
        active_count = int(active_mask.sum())

    management_summary: List[Dict[str, Any]] = []
    if "management" in working.columns:
        mgmt_df = working.copy()
        mgmt_df["management"] = mgmt_df["management"].fillna("未知")
        mgmt_counts = mgmt_df.groupby("management").size().sort_values(ascending=False).head(10)
        mgmt_issue = (
            mgmt_df.groupby("management")["issue_amount"].sum()
            if "issue_amount" in working.columns
            else pd.Series(dtype=float)
        )
        for name, count in mgmt_counts.items():
            management_summary.append(
                {
                    "name": str(name),
                    "count": int(count),
                    "total_issue_amount": _safe_float(mgmt_issue.get(name)) if not mgmt_issue.empty else None,
                }
            )

    organization_type_distribution = _value_distribution(working.get("type"), None)
    fund_type_distribution = _value_distribution(working.get("fund_type"), None)
    invest_type_distribution = _value_distribution(working.get("invest_type"), None)
    market_distribution = _value_distribution(working.get("market"), None)
    status_distribution = _value_distribution(status_series, None)

    market_issue_summary: List[Dict[str, Any]] = []
    if "market" in working.columns:
        market_df = working.copy()
        market_df["market"] = market_df["market"].fillna("未知")
        grouped = market_df.groupby("market")
        issue_series = grouped["issue_amount"].sum() if "issue_amount" in market_df.columns else pd.Series(dtype=float)
        for market, sub_df in grouped:
            market_issue_summary.append(
                {
                    "market": str(market),
                    "count": int(len(sub_df)),
                    "total_issue_amount": _safe_float(issue_series.get(market)) if not issue_series.empty else None,
                }
            )
        market_issue_summary.sort(
            key=lambda item: (item.get("total_issue_amount") or 0.0, item.get("count") or 0), reverse=True
        )

    list_trend = _build_list_trend_series(working)
    list_trend_filters = _build_list_trend_filters(df)

    fee_by_fund_type: List[Dict[str, Any]] = []
    if "fund_type" in working.columns and ("m_fee" in working.columns or "c_fee" in working.columns):
        fee_df = working.copy()
        fee_df["fund_type"] = fee_df["fund_type"].fillna("未知")
        fee_groups = fee_df.groupby("fund_type")
        m_fee_series = fee_groups["m_fee"].mean() if "m_fee" in fee_df.columns else pd.Series(dtype=float)
        c_fee_series = fee_groups["c_fee"].mean() if "c_fee" in fee_df.columns else pd.Series(dtype=float)
        order_series = m_fee_series if not m_fee_series.empty else c_fee_series
        for fund_type in order_series.sort_values(ascending=False).index:
            fee_by_fund_type.append(
                {
                    "fund_type": str(fund_type),
                    "avg_m_fee": _safe_float(m_fee_series.get(fund_type)) if not m_fee_series.empty else None,
                    "avg_c_fee": _safe_float(c_fee_series.get(fund_type)) if not c_fee_series.empty else None,
                }
            )

    top_issue_amount: List[Dict[str, Any]] = []
    if "issue_amount" in working.columns:
        cols = [c for c in ["ts_code", "name", "issue_amount", "list_date", "market"] if c in working.columns]
        top_issue_df = working.dropna(subset=["issue_amount"]).sort_values("issue_amount", ascending=False).head(10)
        for _, row in top_issue_df.iterrows():
            item = {c: row.get(c) for c in cols}
            item["issue_amount"] = _safe_float(item.get("issue_amount"))
            if "list_date" in item:
                if pd.notna(item["list_date"]):
                    item["list_date"] = str(pd.to_datetime(item["list_date"]).date())
                else:
                    item["list_date"] = None
            top_issue_amount.append(item)

    recent_listings: List[Dict[str, Any]] = []
    if "list_date" in working.columns:
        recent_cols = [c for c in ["ts_code", "name", "list_date", "market", "issue_amount"] if c in working.columns]
        recent_df = working.dropna(subset=["list_date"]).sort_values("list_date", ascending=False).head(10)
        for _, row in recent_df.iterrows():
            item = {c: row.get(c) for c in recent_cols}
            if "list_date" in item:
                if pd.notna(item["list_date"]):
                    item["list_date"] = str(pd.to_datetime(item["list_date"]).date())
                else:
                    item["list_date"] = None
            if "issue_amount" in item:
                item["issue_amount"] = _safe_float(item.get("issue_amount"))
            recent_listings.append(item)

    response = {
        "summary": {
            "total_count": int(len(working)),
            "active_count": active_count,
            "unique_managements": int(working["management"].nunique(dropna=True)) if "management" in working.columns else None,
            "total_issue_amount": issue_amount_total,
            "avg_m_fee": avg_m_fee,
            "avg_c_fee": avg_c_fee,
            "avg_exp_return": avg_exp_return,
            "avg_duration_year": avg_duration_year,
        },
        "top_management": management_summary,
        "organization_type_distribution": organization_type_distribution,
        "fund_type_distribution": fund_type_distribution,
        "invest_type_distribution": invest_type_distribution,
        "market_distribution": market_distribution,
        "market_issue_summary": market_issue_summary,
        "status_breakdown": status_distribution,
        "list_trend": list_trend,
        "list_trend_filters": list_trend_filters,
        "fee_by_fund_type": fee_by_fund_type,
        "top_issue_amount": top_issue_amount,
        "recent_listings": recent_listings,
    }

    return JSONResponse(response)


@app.get("/api/etf/analytics/list_trend")
def etf_list_trend(
    dimension: str = Query("all", description="筛选维度，可选 all/type/invest_type/fund_type/management"),
    values: Optional[List[str]] = Query(None, description="筛选值，可传多个"),
):
    df = _load_etf_info_df()
    if df.empty:
        return JSONResponse(status_code=404, content={"detail": "未找到 etf_info_df 数据文件"})

    dimension = (dimension or "all").lower()
    dimension_map = {
        "all": None,
        "type": "type",
        "invest_type": "invest_type",
        "fund_type": "fund_type",
        "management": "management",
    }

    column = dimension_map.get(dimension)
    working = df.copy()

    applied_values = _coerce_filter_list(values)
    if column and applied_values:
        normalized = (
            working[column]
            .fillna("未知")
            .astype(str)
            .replace({"": "未知", "nan": "未知"})
        )
        working = working[normalized.isin(applied_values)]
    elif column:
        # 没有提供具体值时直接返回空序列，避免误导
        working = working.iloc[0:0]

    series = _build_list_trend_series(working)

    return JSONResponse(
        {
            "dimension": dimension,
            "values": applied_values,
            "list_trend": series,
            "filters": _build_list_trend_filters(df),
        }
    )


def _coerce_filter_list(raw: Optional[List[str]]) -> List[str]:
    values: List[str] = []
    if not raw:
        return values
    for entry in raw:
        if entry is None:
            continue
        text = str(entry)
        parts = [text] if "," not in text else text.split(",")
        for part in parts:
            norm = part.strip()
            if norm:
                values.append(norm)
    return values


def _build_filter_options(df: pd.DataFrame, column: str, *, top_n: Optional[int] = None) -> List[Dict[str, Any]]:
    if column not in df.columns:
        return []
    series = df[column].fillna("未知").astype(str)
    series = series.replace({"": "未知", "nan": "未知"})
    counts = series.value_counts()
    if top_n is not None:
        counts = counts.head(top_n)
    return [
        {"value": str(idx), "label": str(idx), "count": int(count)}
        for idx, count in counts.items()
    ]


def _serialize_value(value: Any) -> Any:
    try:
        if isinstance(value, (pd.Timestamp, datetime)):
            if pd.isna(value):
                return None
            return value.strftime("%Y-%m-%d")
        if isinstance(value, np.datetime64):
            ts = pd.to_datetime(value, errors="coerce")
            if pd.isna(ts):
                return None
            return ts.strftime("%Y-%m-%d")
        if isinstance(value, (np.floating, float)):
            if pd.isna(value):
                return None
            return float(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
    except Exception:
        return None
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None


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
    df = _load_etf_info_df()
    if df.empty:
        return JSONResponse(status_code=404, content={"detail": "未找到 etf_info_df 数据文件"})

    working = df.copy()

    # Apply text search across 核心信息
    keyword = (q or "").strip()
    if keyword:
        lowered = keyword.lower()
        search_cols = [col for col in ["ts_code", "code", "name", "management"] if col in working.columns]
        if search_cols:
            mask = pd.Series(False, index=working.index)
            for col in search_cols:
                series = working[col].astype(str).str.lower()
                mask = mask | series.str.contains(lowered, na=False)
            working = working[mask]

    filters = {
        "fund_type": _coerce_filter_list(fund_type),
        "type": _coerce_filter_list(organization_type),
        "invest_type": _coerce_filter_list(invest_type),
        "market": _coerce_filter_list(market),
        "status": _coerce_filter_list(status),
        "management": _coerce_filter_list(management),
        "custodian": _coerce_filter_list(custodian),
    }

    for column, values in filters.items():
        if not values or column not in working.columns:
            continue
        candidates = {v.lower() for v in values}
        series = working[column].fillna("未知").astype(str)
        series = series.replace({"": "未知", "nan": "未知"})
        mask = series.str.lower().isin(candidates)
        working = working[mask]

    sortable_map = {
        "issue_amount": "issue_amount",
        "m_fee": "m_fee",
        "c_fee": "c_fee",
        "exp_return": "exp_return",
        "duration_year": "duration_year",
        "list_date": "list_date",
        "found_date": "found_date",
        "issue_date": "issue_date",
        "name": "name",
    }
    sort_col = sortable_map.get(sort_by, "issue_amount")
    ascending = sort_dir.lower() == "asc"
    if sort_col in working.columns:
        working = working.sort_values(by=sort_col, ascending=ascending, na_position="last", kind="mergesort")

    total_filtered = int(len(working))
    start = (page - 1) * page_size
    end = start + page_size
    paged = working.iloc[start:end].copy()

    preferred_cols = [
        "ts_code",
        "code",
        "name",
        "management",
        "custodian",
        "fund_type",
        "type",
        "invest_type",
        "market",
        "status",
        "benchmark",
        "issue_amount",
        "m_fee",
        "c_fee",
        "exp_return",
        "duration_year",
        "list_date",
        "found_date",
        "issue_date",
        "due_date",
        "purc_startdate",
        "redm_startdate",
    ]
    present_cols = [col for col in preferred_cols if col in paged.columns]
    items: List[Dict[str, Any]] = []
    for _, row in paged.iterrows():
        record: Dict[str, Any] = {}
        for col in present_cols:
            record[col] = _serialize_value(row[col])
        items.append(record)

    def _safe_series_mean(column: str) -> Optional[float]:
        if column not in working.columns:
            return None
        return _safe_float(working[column].mean())

    def _safe_series_sum(column: str) -> Optional[float]:
        if column not in working.columns:
            return None
        return _safe_float(working[column].sum())

    def _safe_series_median(column: str) -> Optional[float]:
        if column not in working.columns:
            return None
        try:
            val = working[column].median()
            if pd.isna(val):
                return None
            return float(val)
        except Exception:
            return None

    active_count: Optional[int] = None
    if "status" in working.columns:
        status_series = working["status"].fillna("未知").astype(str)
        inactive_keywords = ("终止", "退市", "清盘", "暂停", "到期")
        active_mask = ~status_series.str.contains("|".join(inactive_keywords), case=False, na=False)
        active_count = int(active_mask.sum())

    recent_listings: Optional[int] = None
    if "list_date" in working.columns:
        today = pd.Timestamp.today().normalize()
        twelve_months_ago = today - pd.DateOffset(months=12)
        list_dates = pd.to_datetime(working["list_date"], errors="coerce")
        recent_mask = (list_dates.notna()) & (list_dates >= twelve_months_ago)
        recent_listings = int(recent_mask.sum())

    summary = {
        "universe_total": int(len(df)),
        "filtered_total": total_filtered,
        "active_count": active_count,
        "recent_listings_12m": recent_listings,
        "avg_m_fee": _safe_series_mean("m_fee"),
        "avg_c_fee": _safe_series_mean("c_fee"),
        "avg_exp_return": _safe_series_mean("exp_return"),
        "avg_duration_year": _safe_series_mean("duration_year"),
        "total_issue_amount": _safe_series_sum("issue_amount"),
        "median_issue_amount": _safe_series_median("issue_amount"),
        "unique_managements": int(working["management"].nunique()) if "management" in working.columns else None,
    }

    available_filters = {
        "fund_type": _build_filter_options(df, "fund_type"),
        "type": _build_filter_options(df, "type"),
        "invest_type": _build_filter_options(df, "invest_type"),
        "market": _build_filter_options(df, "market"),
        "status": _build_filter_options(df, "status"),
        "management": _build_filter_options(df, "management", top_n=30),
        "custodian": _build_filter_options(df, "custodian", top_n=30),
    }

    response = {
        "items": items,
        "page": page,
        "page_size": page_size,
        "total": total_filtered,
        "summary": summary,
        "available_filters": available_filters,
        "sort_by": sort_col,
        "sort_dir": "asc" if ascending else "desc",
    }
    return JSONResponse(response)


def _match_product_by_identifier(df: pd.DataFrame, identifier: str) -> Optional[pd.Series]:
    if df.empty:
        return None
    token = (identifier or "").strip()
    if not token:
        return None
    lowered = token.lower()
    lowered_nosfx = lowered.split(".")[0]

    def _match_series(column: str) -> Optional[pd.Series]:
        if column not in df.columns:
            return None
        series = df[column].astype(str).str.lower()
        exact = df[series == lowered]
        if not exact.empty:
            return exact.iloc[0]
        if lowered_nosfx and lowered_nosfx != lowered:
            nosfx_match = series.str.split(".").str[0] == lowered_nosfx
            subset = df[nosfx_match]
            if not subset.empty:
                return subset.iloc[0]
        contains = df[series.str.contains(lowered, na=False)]
        if not contains.empty:
            return contains.iloc[0]
        return None

    for col in ["ts_code", "code"]:
        record = _match_series(col)
        if record is not None:
            return record

    if "name" in df.columns:
        names = df["name"].astype(str).str.lower()
        name_match = df[names == lowered]
        if not name_match.empty:
            return name_match.iloc[0]
        contains = df[names.str.contains(lowered, na=False)]
        if not contains.empty:
            return contains.iloc[0]

    return None


def _generate_synthetic_price_series(
    identifier: str,
    issue_amount: Optional[float] = None,
    start_date: Optional[str] = None,
    periods: Optional[int] = 120,
) -> List[Dict[str, Any]]:
    seed = abs(hash(identifier)) % (2 ** 32)
    rng = np.random.default_rng(seed)

    base_price = 1.0
    if issue_amount is not None:
        try:
            base_price = float(issue_amount) / 5000.0
            base_price = float(np.clip(base_price, 0.8, 8.0))
        except Exception:
            base_price = 1.0

    today = pd.Timestamp.today().normalize()
    start_dt: Optional[pd.Timestamp] = None
    if start_date:
        try:
            parsed = pd.to_datetime(start_date, errors="coerce")
        except Exception:
            parsed = None
        if parsed is not None:
            if getattr(parsed, "tzinfo", None) is not None:
                parsed = parsed.tz_convert(None)
            start_dt = parsed
            if start_dt > today:
                start_dt = today

    if start_dt is not None:
        dates = pd.bdate_range(start=start_dt, end=today)
    else:
        if not periods or periods <= 0:
            periods = 120
        dates = pd.bdate_range(end=today, periods=periods)
    if len(dates) == 0:
        return []

    closes = np.empty(len(dates))
    closes[0] = max(base_price, 0.5)
    for i in range(1, len(dates)):
        drift = rng.normal(0.0006, 0.018)
        closes[i] = max(0.2, closes[i - 1] * (1 + drift))

    opens = np.empty_like(closes)
    opens[0] = closes[0] * (1 + rng.normal(0.0, 0.004))
    for i in range(1, len(dates)):
        opens[i] = closes[i - 1] * (1 + rng.normal(0.0, 0.006))

    highs = np.maximum(opens, closes) * (1 + rng.uniform(0.002, 0.02, size=len(dates)))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0.002, 0.02, size=len(dates)))

    volume_base = 80000.0
    if issue_amount is not None:
        try:
            volume_base = float(issue_amount) * 120.0
            volume_base = float(np.clip(volume_base, 20000.0, 5_000_000.0))
        except Exception:
            volume_base = 80000.0
    volumes = volume_base * rng.uniform(0.5, 1.6, size=len(dates))

    series: List[Dict[str, Any]] = []
    for idx, dt in enumerate(dates):
        high = float(max(highs[idx], opens[idx], closes[idx]))
        low = float(min(lows[idx], opens[idx], closes[idx]))
        if high < low:
            high = low
        low = max(low, 0.1)
        series.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "open": round(float(opens[idx]), 4),
                "close": round(float(closes[idx]), 4),
                "high": round(high, 4),
                "low": round(low, 4),
                "volume": float(np.round(volumes[idx], 0)),
            }
        )
    return series


@app.get("/api/etf/products/{product_id}")
def etf_product_detail(product_id: str):
    df = _load_etf_info_df()
    if df.empty:
        return JSONResponse(status_code=404, content={"detail": "未找到 etf_info_df 数据文件"})

    record = _match_product_by_identifier(df, product_id)
    if record is None:
        return JSONResponse(status_code=404, content={"detail": f"未找到编号为 {product_id} 的产品"})

    preferred_info_cols = [
        "ts_code",
        "code",
        "name",
        "management",
        "custodian",
        "trustee",
        "fund_type",
        "type",
        "invest_type",
        "market",
        "status",
        "benchmark",
        "found_date",
        "issue_date",
        "list_date",
        "due_date",
        "purc_startdate",
        "redm_startdate",
    ]
    metric_cols = [
        "issue_amount",
        "m_fee",
        "c_fee",
        "exp_return",
        "duration_year",
        "p_value",
        "min_amount",
    ]

    base_info: Dict[str, Any] = {}
    for col in preferred_info_cols + metric_cols:
        if col in record.index:
            base_info[col] = _serialize_value(record[col])

    metrics = {
        "issue_amount": _safe_float(record.get("issue_amount")),
        "m_fee": _safe_float(record.get("m_fee")),
        "c_fee": _safe_float(record.get("c_fee")),
        "exp_return": _safe_float(record.get("exp_return")),
        "duration_year": _safe_float(record.get("duration_year")),
    }

    identifier = base_info.get("ts_code") or base_info.get("code") or product_id
    start_date = (
        base_info.get("list_date")
        or base_info.get("found_date")
        or base_info.get("issue_date")
        or base_info.get("purc_startdate")
        or None
    )
    time_series = _generate_synthetic_price_series(
        str(identifier),
        issue_amount=metrics.get("issue_amount"),
        start_date=str(start_date) if start_date else None,
    )

    response = {
        "product_id": identifier,
        "name": base_info.get("name") or identifier,
        "management": base_info.get("management"),
        "custodian": base_info.get("custodian"),
        "status": base_info.get("status"),
        "base_info": base_info,
        "metrics": metrics,
        "timeseries": time_series,
    }
    return JSONResponse(response)


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
