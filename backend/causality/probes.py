"""因果性探针。

判据只有一条：**输出在时点 t 的取值，只能是输入 x[0..t] 的函数**。
等价地——改变 t 之后的输入，不允许改变 t 及之前的输出。

P1 尾部扰动是这条判据的直接翻译，也是首选探针。它相对「截断重算」的关键
优势是**序列长度不变**：截断会同时改变预热条件，把「预热不足」和「未来函数」
混成同一个信号（Freqtrade 正因如此不得不把 lookahead-analysis 与
recursive-analysis 拆成两个命令）。扰动法天然把两者分开，P3 再单独去测预热。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Mapping, Sequence

import numpy as np

#: 一致：截断/重排会改变浮点求和顺序，1e-15 级的差异不是泄露。
TIGHT_RTOL, TIGHT_ATOL = 1e-9, 1e-12
#: 灰区上界：超过它才判泄露。真实泄露是宏观的，结论对该阈值不敏感。
LOOSE_RTOL, LOOSE_ATOL = 1e-6, 1e-9
#: P3 预热偏差阈值（相对）。
WARMUP_RTOL = 1e-6

#: 尾部扰动的缩放系数，**全部取正**以保号——净值类输入必须保持为正，否则
#: ``log``/``sqrt``/``drawdown_series`` 会直接抛错，审计结果退化成一片 UNKNOWN。
#:
#: 用多种扰动而不是一种，是因为单一扰动会产生数据相关的假阴性：
#: ``max_consecutive_true`` 只在最长真值段恰好落在头部时才「看起来」不依赖尾部，
#: 换一种扰动就露馅。任何一种扰动改变了结论，结论就是被改变了。
PERTURB_STYLES: tuple[tuple[str, bool, float], ...] = (
    # (名称, 是否倒序, 缩放系数)
    ("reverse", True, 1.9),
    ("amplify", False, 6.0),
    ("shrink", True, 0.013),
)


class Verdict(StrEnum):
    """裁决。``UNKNOWN`` 绝不静默降级为通过。"""

    CAUSAL = "causal"
    WINDOW_CONSUMING = "window_consuming"
    LEAK = "leak"
    WARMUP_SENSITIVE = "warmup_sensitive"
    UNKNOWN = "unknown"


#: 合并多个探针结果时谁说了算。越靠前越严重。
_SEVERITY = (
    Verdict.LEAK,
    Verdict.UNKNOWN,
    Verdict.WARMUP_SENSITIVE,
    Verdict.WINDOW_CONSUMING,
    Verdict.CAUSAL,
)


def merge_verdicts(verdicts: Sequence[Verdict]) -> Verdict:
    for candidate in _SEVERITY:
        if candidate in verdicts:
            return candidate
    return Verdict.UNKNOWN


@dataclass(frozen=True)
class ProbeOutcome:
    verdict: Verdict
    detail: str
    decision_index: int | None = None
    first_mismatch: int | None = None
    baseline_value: float | None = None
    perturbed_value: float | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"verdict": str(self.verdict), "detail": self.detail}
        for key in ("decision_index", "first_mismatch", "baseline_value", "perturbed_value"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload


# --------------------------------------------------------------------------
# 扰动
# --------------------------------------------------------------------------

def perturb_tail(
    values: np.ndarray, decision_index: int, style: str = "reverse"
) -> np.ndarray | None:
    """把 ``decision_index`` 之后的数据换掉，保持长度与定义域不变。

    做法是**沿时间轴倒序再乘正系数**：
    * 倒序改变次序 —— ``argmax``/``rank``/``lag`` 这类只看位置的泄露会显影；
    * 乘正系数改变量级 —— ``max``/``mean``/``sum`` 这类只看数值的泄露会显影，
      并且在尾部只剩一个元素时（倒序等于没动）仍然有效；
    * 系数为正 —— 保号，净值类输入不会变负而让算子直接抛错。

    布尔掩码没有量级，改用取反 / 全真 / 全假。

    返回 ``None`` 表示无法构造有效扰动（尾部为空，或扰动后与原值无异），
    此时该探针在这个决策日上没有证据力，调用方必须跳过而不是判定通过。
    """

    array = np.asarray(values)
    if array.ndim == 0 or array.shape[0] <= decision_index + 1:
        return None
    reverse, scale = next(
        ((flip, factor) for name, flip, factor in PERTURB_STYLES if name == style),
        (True, PERTURB_STYLES[0][2]),
    )
    tail = array[decision_index + 1 :]
    perturbed = array.copy()
    if array.dtype == np.bool_:
        if style == "amplify":
            perturbed[decision_index + 1 :] = True
        elif style == "shrink":
            perturbed[decision_index + 1 :] = False
        else:
            perturbed[decision_index + 1 :] = ~tail[::-1]
    else:
        perturbed[decision_index + 1 :] = (tail[::-1] if reverse else tail) * scale
    if np.array_equal(perturbed, array):
        return None
    return perturbed


# --------------------------------------------------------------------------
# 比较
# --------------------------------------------------------------------------

def _closeness(baseline: np.ndarray, perturbed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tight = np.isclose(baseline, perturbed, rtol=TIGHT_RTOL, atol=TIGHT_ATOL, equal_nan=True)
    loose = np.isclose(baseline, perturbed, rtol=LOOSE_RTOL, atol=LOOSE_ATOL, equal_nan=True)
    return tight, loose


def compare_prefix(
    baseline: Any,
    perturbed: Any,
    *,
    decision_index: int,
    input_length: int,
    time_axis: int = 0,
) -> ProbeOutcome | None:
    """比对两次计算在决策日之前的输出。

    输出沿时间轴被缩短时按**右对齐**解释：本 DSL 里 ``difference(x, n)`` 返回
    ``x[n:] - x[:-n]``，即输出第 j 项属于时点 ``j + n``；``lag(x, n)`` 返回
    ``x[:-n]``，同样是「n 期之前的值」对齐到时点 ``j + n``。若按左对齐比较，
    ``difference`` 会被误判成读取未来——这是本模块最容易出的假阳性。

    返回 ``None`` 表示该决策日没有可比区间（扰动落在输出覆盖范围之外）。
    """

    base = np.asarray(baseline, dtype=np.float64)
    pert = np.asarray(perturbed, dtype=np.float64)
    if base.shape != pert.shape:
        return ProbeOutcome(
            Verdict.LEAK,
            f"扰动未来数据改变了输出形状：{base.shape} → {pert.shape}",
            decision_index=decision_index,
        )
    if base.ndim == 0 or not 0 <= time_axis < base.ndim:
        return None

    # ``transpose`` 会把时间轴换到别的位置，按 0 号轴切会得到一个荒谬的偏移，
    # 于是每个决策日都「没有可比区间」——沉默的假阴性。
    offset = input_length - base.shape[time_axis]
    cutoff = decision_index - offset + 1
    if cutoff <= 0:
        return None
    cutoff = min(cutoff, base.shape[time_axis])

    head = (slice(None),) * time_axis + (slice(0, cutoff),)
    head_base, head_pert = base[head], pert[head]
    tight, loose = _closeness(head_base, head_pert)
    if bool(tight.all()):
        return ProbeOutcome(
            Verdict.CAUSAL,
            f"决策日 {decision_index}：前 {cutoff} 个输出未被未来数据改变",
            decision_index=decision_index,
        )

    # 灰区时 loose 全通过，失配位置要到 tight 里找。
    failing = tight if bool(loose.all()) else loose
    position = np.unravel_index(int(np.argmin(failing.reshape(-1))), head_base.shape)
    baseline_value = float(head_base[position])
    perturbed_value = float(head_pert[position])
    verdict = Verdict.UNKNOWN if bool(loose.all()) else Verdict.LEAK
    detail = (
        f"决策日 {decision_index}：第 {position} 个输出在未来数据被改动后 "
        f"由 {baseline_value!r} 变为 {perturbed_value!r}"
    )
    if verdict is Verdict.UNKNOWN:
        detail += "（落在浮点灰区，需人工确认是数值问题还是泄露）"
    return ProbeOutcome(
        verdict,
        detail,
        decision_index=decision_index,
        first_mismatch=int(position[time_axis]),
        baseline_value=baseline_value,
        perturbed_value=perturbed_value,
    )


# --------------------------------------------------------------------------
# P1 / P3
# --------------------------------------------------------------------------

def default_decision_dates(length: int, count: int = 8) -> tuple[int, ...]:
    """在预热之后均匀取若干决策日，末端留出扰动空间。"""

    warmup = max(8, length // 8)
    last = length - 2
    if last <= warmup:
        return (max(0, length - 2),) if length >= 2 else ()
    step = max(1, (last - warmup) // max(count - 1, 1))
    return tuple(sorted({index for index in range(warmup, last + 1, step)})[:count])


def run_tail_probe(
    evaluate: Callable[[list[Any]], Any],
    arrays: Sequence[Any],
    *,
    time_indices: Sequence[int],
    input_length: int,
    tail_offsets: Mapping[int, int] | None = None,
    decision_dates: Sequence[int] | None = None,
    time_axis: int = 0,
) -> ProbeOutcome:
    """P1 的通用形态：多个入参一起扰动。

    ``tail_offsets`` 给出某个入参相对时间下标的额外偏移。复权净值的类型符号是
    ``L``（比 ``T`` 多一个起点），它在时点 t 之后的第一个下标是 ``t + 2``，
    所以偏移为 1。漏掉这一位会让扰动落在决策日之内，把因果公式误判成泄露。
    """

    dates = tuple(decision_dates) if decision_dates is not None else default_decision_dates(input_length)
    if not dates:
        return ProbeOutcome(Verdict.UNKNOWN, "序列太短，无法构造决策日")
    offsets = dict(tail_offsets or {})

    try:
        baseline = evaluate(list(arrays))
    except Exception as exc:  # noqa: BLE001 - 算子自己的校验错误也是审计结论的一部分
        return ProbeOutcome(Verdict.UNKNOWN, f"基线计算失败：{type(exc).__name__}: {exc}")

    outcomes: list[ProbeOutcome] = []
    for decision_index in dates:
        for style, _, _ in PERTURB_STYLES:
            perturbed_args = list(arrays)
            perturbed_any = False
            for index in time_indices:
                candidate = perturb_tail(
                    np.asarray(arrays[index]), decision_index + offsets.get(index, 0), style
                )
                if candidate is None:
                    continue
                perturbed_args[index] = candidate
                perturbed_any = True
            if not perturbed_any:
                continue
            try:
                perturbed = evaluate(perturbed_args)
            except Exception as exc:  # noqa: BLE001
                outcomes.append(
                    ProbeOutcome(
                        Verdict.UNKNOWN,
                        f"决策日 {decision_index}（{style}）：扰动后计算失败：{type(exc).__name__}: {exc}",
                        decision_index=decision_index,
                    )
                )
                continue
            outcome = compare_prefix(
                baseline,
                perturbed,
                decision_index=decision_index,
                input_length=input_length,
                time_axis=time_axis,
            )
            if outcome is not None:
                outcomes.append(outcome)

    if not outcomes:
        return ProbeOutcome(Verdict.UNKNOWN, "没有任何决策日产生可比区间")
    for severity in (Verdict.LEAK, Verdict.UNKNOWN):
        for outcome in outcomes:
            if outcome.verdict is severity:
                return outcome
    return ProbeOutcome(Verdict.CAUSAL, f"{len(outcomes)} 个决策日全部通过尾部扰动检验")


def tail_dependence(
    evaluate: Callable[[list[Any]], Any],
    arrays: Sequence[Any],
    *,
    time_indices: Sequence[int],
    input_length: int,
    tail_offsets: Mapping[int, int] | None = None,
    decision_dates: Sequence[int] | None = None,
) -> ProbeOutcome:
    """输出不带时间轴时：它到底吃不吃窗口的后半段？

    沿时间轴归约的算子按定义是整窗的函数，本来就不该用 P1 判通过/失败。但
    「输出没有时间轴」并不等于「消费整窗」——``first(x)`` 只用 ``x[0]``，改动
    尾部它纹丝不动，把它归进 WINDOW_CONSUMING 只会让真正需要确认窗口右端的
    算子淹没在噪声里。所以这里实际测一次：改了未来，输出变不变。
    """

    dates = tuple(decision_dates) if decision_dates is not None else default_decision_dates(input_length)
    if not dates:
        return ProbeOutcome(Verdict.UNKNOWN, "序列太短，无法构造决策日")
    offsets = dict(tail_offsets or {})

    try:
        baseline = np.asarray(evaluate(list(arrays)), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(Verdict.UNKNOWN, f"基线计算失败：{type(exc).__name__}: {exc}")

    probed = 0
    for decision_index in dates:
        for style, _, _ in PERTURB_STYLES:
            perturbed_args = list(arrays)
            perturbed_any = False
            for index in time_indices:
                candidate = perturb_tail(
                    np.asarray(arrays[index]), decision_index + offsets.get(index, 0), style
                )
                if candidate is None:
                    continue
                perturbed_args[index] = candidate
                perturbed_any = True
            if not perturbed_any:
                continue
            try:
                perturbed = np.asarray(evaluate(perturbed_args), dtype=np.float64)
            except Exception as exc:  # noqa: BLE001
                return ProbeOutcome(
                    Verdict.UNKNOWN,
                    f"决策日 {decision_index}（{style}）：扰动后计算失败：{type(exc).__name__}: {exc}",
                    decision_index=decision_index,
                )
            probed += 1
            if perturbed.shape != baseline.shape:
                return ProbeOutcome(Verdict.WINDOW_CONSUMING, "输出形状随窗口变化")
            tight, _ = _closeness(baseline, perturbed)
            if not bool(tight.all()):
                return ProbeOutcome(
                    Verdict.WINDOW_CONSUMING,
                    "沿时间轴归约：输出是整个窗口的函数，须在公式层确认窗口右端不晚于决策日",
                    decision_index=decision_index,
                )

    if probed == 0:
        return ProbeOutcome(Verdict.UNKNOWN, "没有任何决策日产生有效扰动")
    return ProbeOutcome(Verdict.CAUSAL, f"输出不依赖决策日之后的数据（{probed} 个决策日）")


def tail_perturbation(
    compute: Callable[[np.ndarray], Any],
    values: np.ndarray,
    *,
    decision_dates: Sequence[int] | None = None,
) -> ProbeOutcome:
    """P1：改变未来，检查过去是否被改写。单入参的常用形态。"""

    array = np.asarray(values)
    if array.ndim == 0:
        return ProbeOutcome(Verdict.UNKNOWN, "标量输入没有时间轴")
    return run_tail_probe(
        lambda args: compute(args[0]),
        [array],
        time_indices=(0,),
        input_length=int(array.shape[0]),
        decision_dates=decision_dates,
    )


def warmup_sensitivity(
    compute: Callable[[np.ndarray], Any],
    values: np.ndarray,
    *,
    lengths: Sequence[int] = (20, 40, 80, 160, 320),
) -> ProbeOutcome:
    """P3：历史长度是否影响最新值。

    这**不是**泄露。递归型算子（EMA 一类）按设计就会在这里亮灯，它说明的是
    回测值与实盘值会因可得历史长度不同而分叉——一个必须被记录的可复现性性质。
    """

    array = np.asarray(values, dtype=np.float64)
    try:
        full = np.asarray(compute(array), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(Verdict.UNKNOWN, f"基线计算失败：{type(exc).__name__}: {exc}")
    if full.ndim == 0 or full.size == 0:
        return ProbeOutcome(Verdict.UNKNOWN, "输出没有时间轴，预热探针不适用")

    reference = float(full.reshape(-1)[-1])
    worst = 0.0
    for length in lengths:
        if length >= array.shape[0]:
            continue
        try:
            trimmed = np.asarray(compute(array[-length:]), dtype=np.float64)
        except Exception:  # noqa: BLE001 - 历史不足是预热探针的正常结果
            continue
        if trimmed.size == 0:
            continue
        candidate = float(trimmed.reshape(-1)[-1])
        if np.isnan(candidate) and np.isnan(reference):
            continue
        scale = max(abs(reference), 1e-12)
        worst = max(worst, abs(candidate - reference) / scale)

    if worst <= WARMUP_RTOL:
        return ProbeOutcome(Verdict.CAUSAL, f"最新值对历史长度不敏感（最大相对偏差 {worst:.2e}）")
    return ProbeOutcome(
        Verdict.WARMUP_SENSITIVE,
        f"最新值随可得历史长度变化，最大相对偏差 {worst:.2e}；回测与实盘会分叉，需声明最小预热长度",
    )
