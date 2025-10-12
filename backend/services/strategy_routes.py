from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from collections import OrderedDict
from threading import Lock
import hashlib
import json

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import JSONResponse

from strategy import compute_risk_budget_weights, compute_target_weights
from backtest_engine import backtest_portfolio, gen_rebalance_dates, slice_fit_data, ensure_valid_rebalance_window


DATA_DIR = (Path(__file__).resolve().parents[2] / "data").resolve()

router = APIRouter(prefix="/api/strategy", tags=["strategy"])


_SCHEDULE_CACHE_SIZE = 64
_schedule_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_schedule_cache_lock = Lock()


def _normalize_for_cache(value: Any) -> Any:
    if isinstance(value, dict):
        normalized_items = []
        for k in sorted(value.keys()):
            v = value[k]
            if v is None:
                continue
            normalized_items.append((k, _normalize_for_cache(v)))
        return {k: v for k, v in normalized_items}
    if isinstance(value, list):
        return [_normalize_for_cache(v) for v in value]
    return value


def _schedule_cache_key(spec: Dict[str, Any]) -> str:
    canonical = _normalize_for_cache(spec)
    payload = json.dumps(canonical, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _store_schedule_cache(key: Optional[str], payload: Dict[str, Any]) -> None:
    if not key:
        return
    with _schedule_cache_lock:
        _schedule_cache[key] = payload
        _schedule_cache.move_to_end(key)
        while len(_schedule_cache) > _SCHEDULE_CACHE_SIZE:
            _schedule_cache.popitem(last=False)


def _get_schedule_cache(key: Optional[str]) -> Optional[Dict[str, Any]]:
    if not key:
        return None
    with _schedule_cache_lock:
        payload = _schedule_cache.get(key)
        if payload is None:
            return None
        _schedule_cache.move_to_end(key)
        return payload

class StrategyClassItem(BaseModel):
    name: str
    weight: Optional[float] = None
    budget: Optional[float] = None


class StrategySpec(BaseModel):
    type: str  # fixed | risk_budget | target
    name: Optional[str] = None
    classes: List[StrategyClassItem]
    rebalance: Optional[Dict[str, Any]] = None
    model: Optional[Dict[str, Any]] = None
    precomputed: Optional[str] = None
    # risk budget params (legacy compute-weights)
    risk_metric: Optional[str] = None
    return_type: Optional[str] = None
    confidence: Optional[float] = None
    days: Optional[int] = None
    window: Optional[int] = None
    # target params (legacy compute-weights)
    target: Optional[str] = None
    return_metric: Optional[str] = None
    risk_free_rate: Optional[float] = None
    target_return: Optional[float] = None
    target_risk: Optional[float] = None
    constraints: Optional[Dict[str, Any]] = None


def _effective_model_config(strategy: StrategySpec) -> Dict[str, Any]:
    base = dict(strategy.model or {})
    if strategy.type == 'risk_budget':
        model = dict(base)
        model.setdefault('risk_metric', strategy.risk_metric or 'vol')
        if model.get('risk_metric') in {'annual_vol', 'ewm_vol'} and model.get('days') is None and strategy.days is not None:
            model['days'] = int(strategy.days)
        if model.get('risk_metric') == 'ewm_vol' and model.get('window') is None and strategy.window is not None:
            model['window'] = int(strategy.window)
        if model.get('risk_metric') in {'var', 'es'} and model.get('confidence') is None and strategy.confidence is not None:
            model['confidence'] = float(strategy.confidence)
        model.setdefault('window_mode', model.get('window_mode') or 'rollingN')
        if 'data_len' not in model and strategy.model and strategy.model.get('data_len') is not None:
            model['data_len'] = strategy.model.get('data_len')
        return model

    model = dict(base)
    if strategy.type == 'target':
        model.setdefault('target', strategy.target or 'min_risk')
        model.setdefault('return_metric', strategy.return_metric or 'annual')
        model.setdefault('return_type', strategy.return_type or 'simple')
        if model.get('days') is None and strategy.days is not None:
            model['days'] = int(strategy.days)
        model.setdefault('risk_metric', strategy.risk_metric or 'vol')
        if model.get('risk_free_rate') is None and strategy.risk_free_rate is not None:
            model['risk_free_rate'] = float(strategy.risk_free_rate)
        if model.get('constraints') is None and strategy.constraints is not None:
            model['constraints'] = strategy.constraints
        if model.get('target_return') is None and strategy.target_return is not None:
            model['target_return'] = strategy.target_return
        if model.get('target_risk') is None and strategy.target_risk is not None:
            model['target_risk'] = strategy.target_risk
    model.setdefault('window_mode', model.get('window_mode') or 'rollingN')
    if 'data_len' not in model and strategy.model and strategy.model.get('data_len') is not None:
        model['data_len'] = strategy.model.get('data_len')
    return model


def _build_schedule_spec(
    alloc_name: str,
    start_date: Optional[str],
    strategy: StrategySpec,
    model: Dict[str, Any],
) -> Dict[str, Any]:
    if strategy.type == 'risk_budget':
        classes = [
            {'name': c.name, 'budget': float(c.budget or 0.0)}
            for c in strategy.classes
        ]
    else:
        classes = [{'name': c.name} for c in strategy.classes]
    rebalance = strategy.rebalance or {}
    return {
        'alloc_name': alloc_name,
        'start_date': start_date,
        'type': strategy.type,
        'rebalance': rebalance,
        'model': model,
        'classes': classes,
    }


class ComputeWeightsRequest(BaseModel):
    alloc_name: str
    strategy: StrategySpec
    data_len: Optional[int] = None
    window_mode: Optional[str] = None  # 'all'|'rollingN'


@router.post("/compute-weights")
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

    if nav_wide.empty:
        return JSONResponse(status_code=400, content={"detail": "净值数据不足，无法计算权重"})

    window_mode = req.window_mode or (req.strategy.model or {}).get('window_mode') or 'all'
    data_len = req.data_len if req.data_len is not None else (req.strategy.model or {}).get('data_len')
    nav_fit = slice_fit_data(nav_wide, nav_wide.index[-1], window_mode, data_len)
    if len(nav_fit) < 2 or (window_mode.lower() == 'rollingn' and data_len and len(nav_fit) < int(data_len)):
        return JSONResponse(status_code=400, content={"detail": "样本不足，无法根据当前窗口计算权重"})

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
        risk_cfg = {"metric": req.strategy.risk_metric or "vol"}
        if req.strategy.risk_metric in {"annual_vol", "ewm_vol"} and req.strategy.days is not None:
            risk_cfg["days"] = int(req.strategy.days)
        if req.strategy.risk_metric == "ewm_vol" and req.strategy.window is not None:
            risk_cfg["window"] = int(req.strategy.window)
        if req.strategy.risk_metric in {"var", "es"} and req.strategy.confidence is not None:
            risk_cfg["confidence"] = float(req.strategy.confidence)
        weights = compute_risk_budget_weights(nav_fit, risk_cfg, budgets, window_len=None, window_mode=None)
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
        asset_names = list(nav_fit.columns)
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
                    lo = float(g.get('lo', 0.0)); hi = float(g.get('hi', 1.0))
                    group_limits[idxs] = (lo, hi)

        weights = compute_target_weights(
            nav_fit,
            ret_cfg,
            risk_cfg,
            target=req.strategy.target or 'min_risk',
            window_len=None,
            window_mode=None,
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


@router.post("/backtest")
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
        weights = [float(cls_map.get(n).weight) if (n in cls_map and cls_map[n].weight is not None) else 0.0 for n in class_names]
        rb = s.rebalance if isinstance(s.rebalance, dict) else None
        model_cfg = _effective_model_config(s)
        cache_spec = _build_schedule_spec(req.alloc_name, req.start_date, s, model_cfg)
        computed_key = _schedule_cache_key(cache_spec)
        sdict = {
            "name": s.name or s.type,
            "type": s.type,
            "weights": weights,
            "rebalance": rb,
            "classes": [c.dict() for c in s.classes],
        }
        if model_cfg:
            sdict["model"] = model_cfg
        precomputed_weights: Optional[Dict[pd.Timestamp, Any]] = None
        if s.precomputed and s.precomputed == computed_key:
            cached = _get_schedule_cache(s.precomputed)
            if cached and cached.get('asset_names') == class_names:
                weight_map = cached.get('weights') or {}
                precomputed_weights = {
                    pd.to_datetime(k): v for k, v in weight_map.items()
                }
        if precomputed_weights:
            sdict["precomputed_weights"] = precomputed_weights
        strat_list.append(sdict)

    res = backtest_portfolio(nav_wide, strat_list, start_date=req.start_date)
    return res


@router.get("/default-start")
def api_default_start(alloc_name: str):
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
    nav_wide = df.pivot_table(index='date', columns='asset_name', values='nv').sort_index()
    nav_wide = nav_wide[nav_wide.index >= default_start_ts]
    count = int(len(nav_wide.index))
    return {"default_start": default_start, "count": count}


class ComputeScheduleRequest(BaseModel):
    alloc_name: str
    start_date: Optional[str] = None
    strategy: StrategySpec


from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def _compute_weight_for_date(args: Dict[str, Any]) -> Dict[str, Any]:
    import pandas as pd
    from strategy import compute_risk_budget_weights, compute_target_weights
    nav_split = args['nav_split']
    nav = pd.DataFrame(nav_split['data'], index=pd.to_datetime(nav_split['index']), columns=nav_split['columns'])
    up_to = pd.to_datetime(args['date'])
    nav = nav.loc[nav.index <= up_to]
    stype = args['stype']
    model = args['model'] or {}
    window_mode = model.get('window_mode') or 'all'
    data_len = model.get('data_len')
    nav_fit = slice_fit_data(nav, up_to, window_mode, data_len)
    if len(nav_fit) < 2 or (window_mode.lower() == 'rollingn' and data_len and len(nav_fit) < int(data_len)):
        raise ValueError('样本不足，无法计算调仓权重')
    asset_names = list(nav_fit.columns)
    if stype == 'risk_budget':
        budgets = args['budgets']
        risk_cfg = {'metric': model.get('risk_metric') or 'vol'}
        if model.get('days') is not None:
            risk_cfg['days'] = int(model.get('days'))
        if model.get('window') is not None:
            risk_cfg['window'] = int(model.get('window'))
        if model.get('confidence') is not None:
            risk_cfg['confidence'] = float(model.get('confidence'))
        w = compute_risk_budget_weights(nav_fit, risk_cfg, budgets, window_len=None)
        return {'date': args['date'], 'weights': [float(x) for x in w]}
    else:
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
            nav_fit, ret_cfg, risk_cfg,
            target=str(model.get('target') or 'min_risk'),
            window_len=None, window_mode=None,
            single_limits=single_limits, group_limits=group_limits,
            risk_free_rate=float(model.get('risk_free_rate') or 0.0),
            target_return=model.get('target_return'), target_risk=model.get('target_risk'),
            use_exploration=False,
        )
        return {'date': args['date'], 'weights': [float(x) for x in w]}


@router.post("/compute-schedule-weights")
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
    class_names = [c.name for c in req.strategy.classes]
    nav_wide = nav_wide[class_names].dropna(how='all').dropna(axis=0)
    if nav_wide.empty:
        return JSONResponse(status_code=400, content={"detail": "净值数据不足，无法计算权重"})

    asset_names = list(nav_wide.columns)
    model_cfg = _effective_model_config(req.strategy)
    cache_spec = _build_schedule_spec(req.alloc_name, req.start_date, req.strategy, model_cfg)
    cache_key = _schedule_cache_key(cache_spec)

    rb = req.strategy.rebalance or {}
    try:
        if not rb.get('enabled') or not rb.get('recalc'):
            rset = [nav_wide.index[0]]
            rset, first_idx = ensure_valid_rebalance_window(nav_wide, rset, model_cfg)
        else:
            mode = str(rb.get('mode', 'monthly'))
            which = str(rb.get('which', 'nth'))
            N = int(rb.get('N', 1))
            unit = str(rb.get('unit', 'trading'))
            fixed_interval = int(rb.get('fixedInterval', 20)) if mode == 'fixed' else None
            rset = gen_rebalance_dates(nav_wide.index, mode, N=N, which=which, unit=unit, fixed_interval=fixed_interval)
            rset = sorted([d for d in rset if d in nav_wide.index])
            if not rset:
                return JSONResponse(status_code=400, content={"detail": "未找到可用调仓日期"})
            rset, first_idx = ensure_valid_rebalance_window(nav_wide, rset, model_cfg)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    nav_split = {'index': [d.isoformat() for d in nav_wide.index], 'columns': asset_names, 'data': nav_wide.values.tolist()}
    tasks: List[Dict[str, Any]] = []
    if req.strategy.type == 'risk_budget':
        budgets = [float(c.budget or 0.0) for c in req.strategy.classes]
        for d in rset:
            tasks.append({
                'date': d.date().isoformat(),
                'nav_split': nav_split,
                'stype': 'risk_budget',
                'model': model_cfg,
                'budgets': budgets,
            })
    else:
        for d in rset:
            tasks.append({
                'date': d.date().isoformat(),
                'nav_split': nav_split,
                'stype': 'target',
                'model': model_cfg,
                'budgets': None,
            })

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
        results = [_compute_weight_for_date(t) for t in tasks]
    results.sort(key=lambda x: x['date'])
    weight_map = {r['date']: r['weights'] for r in results}
    _store_schedule_cache(cache_key, {
        'spec': cache_spec,
        'asset_names': asset_names,
        'weights': weight_map,
    })
    return {
        "asset_names": asset_names,
        "dates": [r['date'] for r in results],
        "weights": [r['weights'] for r in results],
        "cache_key": cache_key,
    }
