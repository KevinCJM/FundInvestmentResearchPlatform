from __future__ import annotations

from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import JSONResponse

from fit import ClassSpec, ETFSpec, compute_classes_nav, compute_rolling_corr, compute_rolling_corr_classes, compute_class_consistency
from optimizer import calculate_efficient_frontier_exploration


DATA_DIR = (Path(__file__).resolve().parents[2] / "data").resolve()

router = APIRouter(prefix="/api", tags=["analytics"])


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


@router.post("/fit-classes", response_model=FitResponse)
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
    cons_out = []
    for row in consistency_rows:
        cons_out.append({
            "name": str(row.get("name")),
            "mean_corr": None if not isinstance(row.get("mean_corr"), (int,float)) or not (row.get("mean_corr") == row.get("mean_corr")) else float(row.get("mean_corr")),
            "pca_evr1": None if not isinstance(row.get("pca_evr1"), (int,float)) or not (row.get("pca_evr1") == row.get("pca_evr1")) else float(row.get("pca_evr1")),
            "max_te": None if not isinstance(row.get("max_te"), (int,float)) or not (row.get("max_te") == row.get("max_te")) else float(row.get("max_te")),
        })
    return FitResponse(dates=dates, navs=navs, corr=corr_vals, corr_labels=corr_labels, metrics=metrics_out, consistency=cons_out)


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


@router.post("/rolling-corr", response_model=RollingResponse)
def rolling_corr(req: RollingRequest):
    try:
        start = pd.to_datetime(req.startDate)
    except Exception:
        raise ValueError("startDate 格式错误，应为 YYYY-MM-DD")
    etfs = [ETFSpec(code=e.code, name=e.name, weight=float(e.weight)) for e in req.etfs]
    idx, series_map, metrics = compute_rolling_corr(DATA_DIR, etfs, start, int(req.window), req.targetCode, req.targetName)
    dates = [d.strftime("%Y-%m-%d") for d in idx]
    safe_series = {k: [float(x) if isinstance(x, (int, float)) and (x == x) and abs(x) != float('inf') else 0.0 for x in v] for k, v in series_map.items()}
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


class FrontierRequest(BaseModel):
    alloc_name: str
    start_date: str
    end_date: str
    return_metric: Dict[str, Any]
    risk_metric: Dict[str, Any]
    risk_free_rate: float = 0.0
    constraints: Optional[Dict[str, Any]] = None
    exploration: Optional[Dict[str, Any]] = None
    quantization: Optional[Dict[str, Any]] = None
    refine: Optional[Dict[str, Any]] = None


@router.post("/efficient-frontier")
def post_efficient_frontier(req: FrontierRequest):
    nv_path = DATA_DIR / "asset_nv.parquet"
    if not nv_path.exists():
        return JSONResponse(status_code=404, content={"detail": "净值数据文件 asset_nv.parquet 不存在"})
    df = pd.read_parquet(nv_path)
    alloc_df = df[df["asset_alloc_name"] == req.alloc_name].copy()
    if alloc_df.empty:
        return JSONResponse(status_code=404, content={"detail": f"未找到名为 '{req.alloc_name}' 的配置的净值数据"})
    alloc_df['date'] = pd.to_datetime(alloc_df['date'])
    mask = (alloc_df['date'] >= pd.to_datetime(req.start_date)) & (alloc_df['date'] <= pd.to_datetime(req.end_date))
    alloc_df = alloc_df.loc[mask]
    if alloc_df.empty:
        return JSONResponse(status_code=400, content={"detail": "在选定日期区间内没有数据"})
    nav_wide = alloc_df.pivot_table(index='date', columns='asset_name', values='nv').sort_index()
    return_type = req.return_metric.get('type', 'simple')
    if return_type == 'log':
        returns_df = np.log(nav_wide / nav_wide.shift(1)).dropna()
    else:
        returns_df = nav_wide.pct_change().dropna()

    # constraints mapping
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
            rounds.append({'samples': int(r.get('samples', 100)), 'step': float(r.get('step', 0.5)), 'buckets': int(r.get('buckets', 50))})
    quant_step = None
    if req.quantization:
        v = req.quantization.get('step', None)
        quant_step = None if v in (None, 'none') else float(v)
    use_refine = bool(req.refine.get('use_slsqp', False)) if req.refine else False
    refine_count = int(req.refine.get('count', 0)) if req.refine else 0

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

    import math
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
        "frontier": sorted([p for p in results.get("frontier", []) if is_finite_point(p)], key=lambda o: extract_value(o)[0]),
        "max_sharpe": results.get("max_sharpe") if is_finite_point(results.get("max_sharpe")) else None,
        "min_variance": results.get("min_variance") if is_finite_point(results.get("min_variance")) else None,
    }
    return clean_results
