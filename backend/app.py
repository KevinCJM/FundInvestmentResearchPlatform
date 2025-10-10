from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from functools import lru_cache
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Tuple, Dict, Any
from starlette.responses import HTMLResponse, JSONResponse
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


# ---- 静态页面托管（可选：将前端构建产物放到 frontend/dist 下） ----
DIST_DIR = (Path(__file__).resolve().parents[1] / "frontend" / "dist").resolve()
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="static")


@app.get("/{full_path:path}")
def spa_fallback(full_path: str):
    print(f"[DEBUG] SPA Fallback triggered for path: {full_path}")
    if full_path.startswith("api"):
        return {"detail": "Not Found"}
    index_file = DIST_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>Frontend not built. Run: cd frontend && npm install && npm run build</h3>",
                        status_code=200)


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
