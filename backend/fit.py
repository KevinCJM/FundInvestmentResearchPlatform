from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable
import numpy as np
import pandas as pd
from pathlib import Path
import math


@dataclass
class ETFSpec:
    code: str
    name: str
    weight: float  # percentage 0-100


@dataclass
class ClassSpec:
    id: str
    name: str
    etfs: List[ETFSpec]


def _code_nosfx(x: str) -> str:
    xs = (x or "").strip()
    if "." in xs:
        return xs.split(".")[0]
    return xs


def _load_adj_nav(data_dir: Path, codes: Iterable[str], names: Iterable[str]) -> pd.DataFrame:
    """高效读取：只取所需列，尽量用 filters 过滤所需 code/name（pyarrow 引擎）。"""
    pq = data_dir / "etf_daily_df.parquet"
    if not pq.exists():
        raise FileNotFoundError("data/etf_daily_df.parquet 不存在")
    cols = ["ts_code", "name", "date", "adj_nav"]
    # 使用 pyarrow 过滤（OR 组合）
    codes = [c for c in set(codes) if c]
    names = [n for n in set(names) if n]
    df: pd.DataFrame
    try:
        filters = None
        # pandas.read_parquet(filters=...) 需要 engine='pyarrow'
        filt_list = []
        if codes:
            filt_list.append([("ts_code", "in", codes)])
        if names:
            filt_list.append([("name", "in", names)])
        if filt_list:
            filters = filt_list  # 这是 OR 逻辑
        df = pd.read_parquet(pq, columns=cols, engine="pyarrow", filters=filters)
        if df.empty and not (codes or names):
            # 兜底情况
            df = pd.read_parquet(pq, columns=cols, engine="pyarrow")
    except Exception:
        # 回退：读取列后再 pandas 过滤
        df = pd.read_parquet(pq, columns=cols)
        if codes:
            df = df[df["ts_code"].astype(str).isin(codes)]
        if names:
            df = pd.concat([df, df[df["name"].astype(str).isin(names)]], axis=0).drop_duplicates()
    # 规范
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"parquet 缺少必要列：{c}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "adj_nav"]).sort_values("date")
    return df


def _returns_from_adj_nav(series: pd.Series) -> pd.Series:
    r = series.sort_index().pct_change().dropna()
    return r


def _pick_series(df: pd.DataFrame, code: str, name: str) -> Optional[pd.Series]:
    code_ns = _code_nosfx(code)
    ts = df["ts_code"].astype(str)
    nm = df["name"].astype(str)
    sub = df[(ts == code) | (ts.str.split(".").str[0] == code_ns) | (nm == name)]
    if sub.empty and code_ns:
        sub = df[ts.str.contains(code_ns, na=False)]
    if sub.empty:
        return None
    s = sub.groupby("date")["adj_nav"].last().sort_index()
    if s.empty:
        return None
    return s


def compute_classes_nav(
    data_dir: Path,
    classes: List[ClassSpec],
    start_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    返回：(nav_df, corr_df, metrics_df)
    - nav_df: index=日期, columns=大类名称, 值=虚拟净值（起始=1）
    - corr_df: 各大类收益率相关系数矩阵
    - metrics_df: 各大类年化收益、年化波动、夏普
    """
    # 收集所需 code/name 以做过滤
    want_codes = []
    want_names = []
    for c in classes:
        for e in c.etfs:
            if e.weight is None:
                continue
            want_codes.append(e.code)
            want_names.append(e.name)
    df = _load_adj_nav(data_dir, want_codes, want_names)

    # 先构造宽表（日期 x ts_code）便于矢量化
    piv = df.pivot_table(index="date", columns="ts_code", values="adj_nav", aggfunc="last").sort_index()
    ret_wide = piv.pct_change().iloc[1:]  # 第一行为 NaN 去掉
    # 过滤开始日期
    ret_wide = ret_wide.loc[ret_wide.index >= start_date]
    ret_wide = ret_wide.fillna(0.0)

    # 为每个输入 ETF 找到对应的 ts_code 列名
    ts_list = list(ret_wide.columns.astype(str))
    def map_to_ts(code: str, name: str) -> Optional[str]:
        if code in ts_list:
            return code
        ns = _code_nosfx(code)
        if ns in ts_list:
            return ns
        # 回退：用 name 匹配（只能在原 df 里找一次）
        sub = df[(df["name"].astype(str) == name)]
        if not sub.empty:
            return str(sub.iloc[0]["ts_code"])
        # 再回退：包含关系
        cand = [c for c in ts_list if ns and ns in str(c)]
        return cand[0] if cand else None

    # 矢量化计算每个大类收益：R * w
    class_ret: Dict[str, pd.Series] = {}
    for c in classes:
        if not c.etfs:
            continue
        cols = []
        wgt = []
        for e in c.etfs:
            if e.weight is None:
                continue
            col = map_to_ts(str(e.code), str(e.name))
            if col is None or col not in ret_wide.columns:
                continue
            cols.append(col)
            wgt.append(max(0.0, float(e.weight)))
        if not cols:
            continue
        w = np.array(wgt, dtype=float)
        s = w.sum()
        if s <= 0:
            continue
        w = w / s  # 归一化到 1
        sub = ret_wide[cols]
        # matrix dot: 每列按权重相加
        agg = pd.Series(sub.to_numpy() @ w, index=sub.index)
        class_ret[c.name] = agg

    if not class_ret:
        raise ValueError("没有可用的大类收益率：请检查权重或数据匹配。")

    # 对齐所有大类的收益率（内联），并据此计算 nav/corr/metrics
    R = pd.DataFrame(class_ret).sort_index()
    if R.empty:
        raise ValueError("没有可用的大类收益率：请检查权重或数据匹配。")
    R = R.fillna(0.0)
    # nav 从 1 开始
    NAV = (1 + R).cumprod()
    NAV.iloc[0] = 1.0

    # 相关系数
    corr = R.corr().fillna(0.0).clip(-1.0, 1.0)

    # 指标（年化）
    ann_factor = 252.0
    mean = R.mean() * ann_factor
    vol = R.std(ddof=1) * np.sqrt(ann_factor)
    sharpe = mean.divide(vol.replace(0, np.nan))
    # 风险指标（基于日收益）
    q01 = R.quantile(0.01)
    var99 = -q01  # 损失为正值
    es99_vals: List[float] = []
    for col in R.columns:
        q = q01[col]
        tail = R[col][R[col] <= q]
        es = -float(tail.mean()) if len(tail) > 0 else float("nan")
        es99_vals.append(es)
    es99 = pd.Series(es99_vals, index=R.columns)
    # 最大回撤与卡玛（基于 NAV）
    roll_max = NAV.cummax()
    drawdown = NAV / roll_max - 1.0
    max_dd = drawdown.min()  # 负值
    calmar = mean.divide(max_dd.abs().replace(0, np.nan))
    metrics = pd.DataFrame({
        "年化收益率": mean,
        "年化波动率": vol,
        "夏普比率": sharpe,
        "99%VaR(日)": var99,
        "99%ES(日)": es99,
        "最大回撤": max_dd,
        "卡玛比率": calmar,
    })
    # 清理潜在的 inf/NaN（前端/JSON 不接受 NaN/inf）
    metrics = metrics.replace([np.inf, -np.inf], np.nan)
    return NAV, corr, metrics


def compute_rolling_corr(
    data_dir: Path,
    etfs: List[ETFSpec],
    start_date: pd.Timestamp,
    window: int,
    target_code: str,
    target_name: str,
) -> Tuple[pd.DatetimeIndex, Dict[str, np.ndarray], List[Dict[str, float]]]:
    """
    以研究对象（target）为主，计算其余各个 ETF 与它的滚动相关系数。
    返回： (dates, series_map, metrics_list)
      - dates: 对齐后的日期索引
      - series_map: {other_label: rolling_corr_array}
      - metrics_list: 每个 other 的统计 {name, sum, mean, median, std, skew, kurtosis}
    """
    if window <= 1:
        raise ValueError("window 必须 > 1")
    # 过滤有权重或至少存在的 etf 列表
    codes = [e.code for e in etfs]
    names = [e.name for e in etfs]
    df = _load_adj_nav(data_dir, codes, names)
    piv = df.pivot_table(index="date", columns="ts_code", values="adj_nav", aggfunc="last").sort_index()
    ret = piv.pct_change().iloc[1:]
    ret = ret.loc[ret.index >= start_date]
    ret = ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 确定 target 列
    ts_cols = list(ret.columns.astype(str))
    tgt = None
    if target_code in ts_cols:
        tgt = target_code
    elif _code_nosfx(target_code) in ts_cols:
        tgt = _code_nosfx(target_code)
    else:
        sub = df[df["name"].astype(str) == target_name]
        if not sub.empty:
            tgt = str(sub.iloc[0]["ts_code"])
    if (tgt is None) or (tgt not in ret.columns):
        raise ValueError("未能找到研究对象的收益序列")

    base = ret[tgt]
    # 计算滚动相关：对每个其他列进行 rolling.corr(base)
    series_map: Dict[str, np.ndarray] = {}
    metrics_list: List[Dict[str, float]] = []
    for col in ret.columns:
        if col == tgt:
            continue
        rc = ret[col].rolling(window).corr(base)
        rc = rc.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        series_map[col] = rc.to_numpy()
        arr = rc.to_numpy()
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            metrics = {"name": col, "sum": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0, "skew": 0.0, "kurtosis": 0.0}
        else:
            mean = float(np.mean(finite))
            median = float(np.median(finite))
            std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            # 偏度、峰度（过量峰度）
            if finite.size > 2:
                m3 = float(((finite - mean) ** 3).mean())
                m2 = float(((finite - mean) ** 2).mean())
                skew = m3 / (m2 ** 1.5 + 1e-12)
                m4 = float(((finite - mean) ** 4).mean())
                kurt = m4 / (m2 ** 2 + 1e-12) - 3.0
            else:
                skew = 0.0
                kurt = 0.0
            metrics = {
                "name": col,
                "sum": float(np.sum(finite)),
                "mean": mean,
                "median": median,
                "std": std,
                "skew": float(skew),
                "kurtosis": float(kurt),
            }
        metrics_list.append(metrics)
    return ret.index, series_map, metrics_list


def compute_rolling_corr_classes(
    data_dir: Path,
    classes: List[ClassSpec],
    start_date: pd.Timestamp,
    window: int,
    target_class_name: str,
) -> Tuple[pd.DatetimeIndex, Dict[str, np.ndarray], List[Dict[str, float]]]:
    if window <= 1:
        raise ValueError("window 必须 > 1")
    # 收集所需代码和名称以便过滤
    want_codes: List[str] = []
    want_names: List[str] = []
    for c in classes:
        for e in c.etfs:
            want_codes.append(e.code)
            want_names.append(e.name)
    df = _load_adj_nav(data_dir, want_codes, want_names)
    piv = df.pivot_table(index="date", columns="ts_code", values="adj_nav", aggfunc="last").sort_index()
    ret_wide = piv.pct_change().iloc[1:]
    ret_wide = ret_wide.loc[ret_wide.index >= start_date].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ts_cols = list(ret_wide.columns.astype(str))
    def map_to_ts(code: str, name: str) -> Optional[str]:
        if code in ts_cols:
            return code
        ns = _code_nosfx(code)
        if ns in ts_cols:
            return ns
        sub = df[df["name"].astype(str) == name]
        if not sub.empty:
            return str(sub.iloc[0]["ts_code"])
        cand = [c for c in ts_cols if ns and ns in str(c)]
        return cand[0] if cand else None

    # 构建每个大类的收益序列（矢量化）
    class_ret: Dict[str, pd.Series] = {}
    for c in classes:
        cols: List[str] = []
        wgt: List[float] = []
        for e in c.etfs:
            col = map_to_ts(str(e.code), str(e.name))
            if col is None or col not in ret_wide.columns:
                continue
            cols.append(col)
            wgt.append(max(0.0, float(e.weight)))
        if not cols:
            continue
        w = np.array(wgt, dtype=float)
        s = w.sum()
        if s <= 0:
            continue
        w = w / s
        sub = ret_wide[cols]
        agg = pd.Series(sub.to_numpy() @ w, index=sub.index)
        class_ret[c.name] = agg

    if not class_ret:
        raise ValueError("没有可用的大类收益率：请检查权重或数据匹配。")

    R = pd.DataFrame(class_ret).sort_index().fillna(0.0)
    if target_class_name not in R.columns:
        raise ValueError("研究对象大类未找到")
    base = R[target_class_name]
    series_map: Dict[str, np.ndarray] = {}
    metrics_list: List[Dict[str, float]] = []
    for name, s in R.items():
        if name == target_class_name:
            continue
        rc = s.rolling(window).corr(base)
        rc = rc.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        series_map[name] = rc.to_numpy()
        arr = rc.to_numpy()
        finite = arr[np.isfinite(arr)]
        overall = float(s.corr(base))
        if not math.isfinite(overall):
            overall = 0.0
        if finite.size == 0:
            metrics = {"name": name, "overall": overall, "mean": 0.0, "median": 0.0, "std": 0.0, "skew": 0.0, "kurtosis": 0.0}
        else:
            mean = float(np.mean(finite))
            median = float(np.median(finite))
            std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            if finite.size > 2:
                mu = mean
                m3 = float(((finite - mu) ** 3).mean())
                m2 = float(((finite - mu) ** 2).mean())
                skew = m3 / (m2 ** 1.5 + 1e-12)
                m4 = float(((finite - mu) ** 4).mean())
                kurt = m4 / (m2 ** 2 + 1e-12) - 3.0
            else:
                skew = 0.0
                kurt = 0.0
            metrics = {"name": name, "overall": overall, "mean": mean, "median": median, "std": std, "skew": float(skew), "kurtosis": float(kurt)}
        metrics_list.append(metrics)
    return R.index, series_map, metrics_list


def compute_class_consistency(
    data_dir: Path,
    classes: List[ClassSpec],
    start_date: pd.Timestamp,
) -> List[Dict[str, float]]:
    """计算每个大类内部 ETF 的一致性：成对相关系数均值、第一主成分解释度。
    返回列表：[ {name, mean_corr, pca_evr1} ]
    """
    # 准备一次性读取所有需要的 ETF 数据
    codes: List[str] = []
    names: List[str] = []
    for c in classes:
        for e in c.etfs:
            codes.append(e.code)
            names.append(e.name)
    df = _load_adj_nav(data_dir, codes, names)
    piv = df.pivot_table(index="date", columns="ts_code", values="adj_nav", aggfunc="last").sort_index()
    ret_wide = piv.pct_change().iloc[1:]
    ret_wide = ret_wide.loc[ret_wide.index >= start_date]
    ret_wide = ret_wide.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ts_cols = list(ret_wide.columns.astype(str))

    def map_to_ts(code: str, name: str) -> Optional[str]:
        if code in ts_cols:
            return code
        ns = _code_nosfx(code)
        if ns in ts_cols:
            return ns
        sub = df[df["name"].astype(str) == name]
        if not sub.empty:
            return str(sub.iloc[0]["ts_code"])
        cand = [c for c in ts_cols if ns and ns in str(c)]
        return cand[0] if cand else None

    out: List[Dict[str, float]] = []
    for c in classes:
        cols: List[str] = []
        for e in c.etfs:
            col = map_to_ts(str(e.code), str(e.name))
            if col is not None and col in ret_wide.columns:
                cols.append(col)
        if len(cols) < 2:
            out.append({"name": c.name, "mean_corr": float("nan"), "pca_evr1": float("nan")})
            continue
        sub = ret_wide[cols]
        # 成对相关系数均值
        C = sub.corr().to_numpy()
        n = C.shape[0]
        iu = np.triu_indices(n, 1)
        vals = C[iu]
        if vals.size == 0 or not np.isfinite(vals).any():
            mean_corr = float("nan")
        else:
            mean_corr = float(np.nanmean(vals))
        # 第一主成分解释度（基于协方差矩阵）
        S = np.cov(sub.to_numpy(), rowvar=False, ddof=1)
        try:
            eigvals = np.linalg.eigvalsh(S)
            total = float(np.sum(eigvals))
            p1 = float(np.max(eigvals)) if eigvals.size else 0.0
            evr1 = (p1 / total) if total > 1e-12 else float("nan")
        except Exception:
            evr1 = float("nan")
        out.append({"name": c.name, "mean_corr": mean_corr, "pca_evr1": evr1})
def compute_class_consistency(
    data_dir: Path,
    classes: List[ClassSpec],
    start_date: pd.Timestamp,
) -> List[Dict[str, float]]:
    """计算每个大类内部 ETF 的一致性：成对相关系数均值、第一主成分解释度、最大跟踪误差。
    返回列表：[ {name, mean_corr, pca_evr1, max_te} ]
    """
    # 准备一次性读取所有需要的 ETF 数据
    codes: List[str] = []
    names: List[str] = []
    for c in classes:
        for e in c.etfs:
            codes.append(e.code)
            names.append(e.name)
    df = _load_adj_nav(data_dir, codes, names)
    piv = df.pivot_table(index="date", columns="ts_code", values="adj_nav", aggfunc="last").sort_index()
    ret_wide = piv.pct_change().iloc[1:]
    ret_wide = ret_wide.loc[ret_wide.index >= start_date]
    ret_wide = ret_wide.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ts_cols = list(ret_wide.columns.astype(str))

    def map_to_ts(code: str, name: str) -> Optional[str]:
        if code in ts_cols:
            return code
        ns = _code_nosfx(code)
        if ns in ts_cols:
            return ns
        sub = df[df["name"].astype(str) == name]
        if not sub.empty:
            return str(sub.iloc[0]["ts_code"])
        cand = [c for c in ts_cols if ns and ns in str(c)]
        return cand[0] if cand else None

    out: List[Dict[str, float]] = []
    for c in classes:
        cols: List[str] = []
        for e in c.etfs:
            col = map_to_ts(str(e.code), str(e.name))
            if col is not None and col in ret_wide.columns:
                cols.append(col)
        if len(cols) < 2:
            out.append({"name": c.name, "mean_corr": float("nan"), "pca_evr1": float("nan"), "max_te": float("nan")})
            continue
        sub = ret_wide[cols]
        # 1) 成对相关系数均值
        C = sub.corr().to_numpy()
        n = C.shape[0]
        iu = np.triu_indices(n, 1)
        vals = C[iu]
        mean_corr = float(np.nanmean(vals)) if vals.size > 0 and np.isfinite(vals).any() else float("nan")
        
        # 2) 第一主成分解释度（基于协方差矩阵）
        S = np.cov(sub.to_numpy(), rowvar=False, ddof=1)
        try:
            eigvals = np.linalg.eigvalsh(S)
            total = float(np.sum(eigvals))
            p1 = float(np.max(eigvals)) if eigvals.size else 0.0
            evr1 = (p1 / total) if total > 1e-12 else float("nan")
        except Exception:
            evr1 = float("nan")

        # 3) 最大跟踪误差（年化）
        r_avg = sub.mean(axis=1)
        diffs = sub.sub(r_avg, axis=0)
        # 年化跟踪误差 TE = std(r_i - r_benchmark) * sqrt(252)
        tracking_errors = diffs.std(ddof=1) * np.sqrt(252)
        max_te = float(tracking_errors.max())

        out.append({"name": c.name, "mean_corr": mean_corr, "pca_evr1": evr1, "max_te": max_te})
    return out
