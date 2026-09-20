"""One candidate drives product, fee, cash and funding evidence."""

from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess
import numba
from numba import njit, types
import numpy as np
from backend.custom_indicators.errors import IndicatorDomainError
from backend.sensitivity.repository import digest_json, file_hash
from . import risk, risk_kernels, costs, cost_kernels, funding, paths, path_kernels


def execution_manifest():
    root = Path(__file__).resolve().parents[2]
    paths = sorted(
        p
        for p in (root / "backend").rglob("*.py")
        if not {"tests", "__pycache__", ".venv", "venv"}.intersection(
            p.relative_to(root).parts
        )
    )
    hashes = {str(p.relative_to(root)): file_hash(p) for p in paths}
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout
        )
    except (OSError, subprocess.SubprocessError):
        head, dirty = None, None
    return {
        "source_fingerprints": hashes,
        "source_hash": digest_json(hashes),
        "git_head": head,
        "working_tree_dirty": dirty,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba.__version__,
        },
        "risk": risk_kernels.audit(),
        "costs": cost_kernels.audit(),
        "funding": funding.goals.execution_audit(),
        "product_paths": path_kernels.audit(),
    }


def check(
    identifier,
    title,
    status,
    reason,
    *,
    scope="implementation",
    enforcement="hard",
    **details,
):
    return {
        "check_id": identifier,
        "title": title,
        "status": status,
        "reason": reason,
        "scope": scope,
        "enforcement": enforcement,
        "contract_version": "implementation-validation/1",
        "unit": None,
        "horizon": None,
        "confidence": None,
        "comparison": None,
        "limit": None,
        "tolerance": 1e-8,
        **details,
    }


def unavailable(identifier, title, error, **kwargs):
    return check(
        identifier,
        title,
        "unavailable",
        getattr(error, "message", str(error)),
        error_code=getattr(error, "code", str(error)),
        **kwargs,
    )


def evaluate(service, candidate, candidate_hash, *, validation=False):
    risk_kernels.require_ready()
    cost_kernels.require_ready()
    funding.goals.require_ready()
    checks, arrays, models = [], {}, []
    result = {
        "candidate_hash": candidate_hash,
        "checks": checks,
        "models": models,
        "scope": "product_moment_research_and_declared_costs",
        "execution": execution_manifest(),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "independent_simulation": False,
        "validation_mode": "frozen_candidate_validation" if validation else "preview",
        "historical_pit_eligibility": False,
        "independent_review": "unavailable_no_authenticated_identity",
        "limitations": [
            "复权净值是收益代理，不是可成交价格；费用为声明费率的研究估计。",
            "D=0 是前瞻模型假设，历史残差正交不证明未来正交。",
            "历史 PIT 与认证独立复核未验证，研究定稿不代表交易授权。",
        ],
    }
    try:
        source = service.sources.resolve(candidate.source, candidate.as_of)
        result["dependencies"] = source["refs"]
        result["frozen_source"] = {
            "saved": source["saved"],
            "baseline": source["baseline"],
            "target": source["target"],
            "assumptions": source["assumptions"],
        }
        checks.append(
            check(
                "source",
                "来源与当前资格",
                "passed",
                "已核对政策、目标、CMA、产品域及映射版本。",
                scope="research",
            )
        )
    except IndicatorDomainError as exc:
        checks.append(unavailable("source", "来源与当前资格", exc, scope="research"))
        return summarize(result), arrays
    names = [a["id"] for a in source["baseline"]["assets"]]
    if any(p.asset_class_id not in names for p in candidate.products):
        checks.append(
            check(
                "budget", "大类预算", "failed", "产品引用未知大类。", scope="research"
            )
        )
        return summarize(result), arrays
    weights = np.asarray([p.weight for p in candidate.products], dtype=np.float64)
    classes = np.asarray(
        [names.index(p.asset_class_id) for p in candidate.products], dtype=np.int64
    )
    target = np.asarray([source["target"][a] for a in names], dtype=np.float64)
    actual, valid = risk_kernels.budget_kernel(
        weights,
        classes,
        target,
        np.asarray([p.max_weight for p in candidate.products], dtype=np.float64),
    )
    checks.append(
        check(
            "budget",
            "大类预算与产品上限",
            "passed" if valid else "failed",
            "产品权重须合计 100%，并逐类匹配来源预算。",
            scope="research",
            actual=actual.tolist(),
            target=target.tolist(),
            asset_ids=names,
        )
    )
    try:
        result["confirmed_balance"] = costs.balance(candidate)
        checks.append(
            check(
                "balance",
                "资金余额对账",
                "passed",
                "各资金项目与确认总价值一致，组合外储备未重复扣除。",
                scope="research",
            )
        )
    except IndicatorDomainError as exc:
        checks.append(
            check("balance", "资金余额对账", "failed", str(exc), scope="research")
        )
    panel = None
    if valid:
        try:
            factors, products, metadata = risk.load_panel(
                service.strategic, source, candidate
            )
            beta, residual_cov, fit = risk.fit_joint(
                factors, products, candidate, source
            )
            models, model_arrays = risk.evaluate_models(
                source, candidate, beta, residual_cov
            )
            arrays.update(
                class_returns=factors,
                product_returns=products,
                beta=beta,
                residual_covariance=residual_cov,
                **model_arrays,
            )
            panel = (factors, products, metadata)
            result.update(models=models, exposure_evidence=fit, data=metadata)
            checks.append(
                check(
                    "exposure",
                    "联合暴露与残差",
                    "passed",
                    "同轴联合回归、独立时间留出和完整残差协方差已计算。",
                )
            )
            for row in models:
                checks.append(
                    check(
                        "risk:" + row["model_id"],
                        row["name"] + "：产品风险",
                        row["status"],
                        "；".join(row["violations"])
                        or "产品波动、相对 SAA 总主动风险及适用基准约束通过。",
                        scope="research" if row["enforced"] else "diagnostic",
                        enforcement="hard" if row["enforced"] else "information",
                        model_ref=row["model_id"],
                        metric="annual_product_risk",
                        unit="annual_decimal",
                        horizon="annual",
                    )
                )
        except (IndicatorDomainError, ValueError, np.linalg.LinAlgError) as exc:
            invalid_mapping = getattr(exc, "code", "") in (
                "IMPLEMENTATION_PRODUCT_MEMBERSHIP",
                "IMPLEMENTATION_CASH_CLASS",
            )
            row = unavailable(
                "exposure",
                "联合暴露与残差",
                exc,
                scope="research" if invalid_mapping else "implementation",
            )
            if invalid_mapping:
                row["status"] = "failed"
            checks.append(row)
    checks.append(
        check(
            "return_basis",
            "收益与费用口径",
            "passed" if candidate.return_basis_confirmed else "unavailable",
            (
                "已声明代理、CMA 和产品收益口径一致；历史截距不作为未来主动收益。"
                if candidate.return_basis_confirmed
                else "请确认收益口径；确认前只显示风险，不生成资金成功率。"
            ),
        )
    )
    trade = None
    if valid and any(
        x["check_id"] == "balance" and x["status"] == "passed" for x in checks
    ):
        try:
            trade = costs.transition(candidate)
            result["transition"] = trade
            checks.append(
                check(
                    "costs",
                    "费用与自融资",
                    "passed",
                    "首次建仓和每边成交收费，扣费后金额与目标权重已对账。",
                    unit="currency",
                )
            )
            cash = costs.cash_calendar(candidate, trade)
            result["cash_calendar"] = cash
            checks.append(
                check(
                    "cash_calendar",
                    "近期可支付现金",
                    cash["status"],
                    (
                        "存在付款日现金缺口。"
                        if cash["status"] == "failed"
                        else (
                            "存在未知到账日。"
                            if cash["status"] == "unavailable"
                            else "已登记事件未出现日末资金缺口。"
                        )
                    ),
                )
            )
            checks.append(
                check(
                    "execution_calendar",
                    "实际执行与全期付款日历",
                    "unavailable",
                    "已登记日期与模型月末路径分别有效；尚未覆盖全期实际交收、盘中截止、申赎暂停、整数份额及成交容量。",
                )
            )
            if panel:
                buy, sell, cash_flags = costs.fee_arrays(candidate)
                dates = panel[2]["dates"]
                reset = np.asarray(
                    [
                        int(
                            candidate.future_weight_rule == "monthly_rebalance"
                            and (
                                i + 1 == len(dates) or dates[i][:7] != dates[i + 1][:7]
                            )
                        )
                        for i in range(len(dates))
                    ],
                    dtype=np.int64,
                )
                replay = cost_kernels.net_replay_kernel(
                    panel[1], weights, buy, sell, cash_flags, reset
                )
                arrays["net_replay"] = replay
                result["historical_replay"] = {
                    "gross_terminal": float(replay[-1, 0]),
                    "net_terminal": float(replay[-1, 1]),
                    "weight_rule": (
                        "monthly_rebalance"
                        if candidate.future_weight_rule == "monthly_rebalance"
                        else "buy_and_hold"
                    ),
                    "scope": "static_selected_target_historical_replay_not_dynamic_taa",
                    "cost_basis": "current_declared_rates_sensitivity_not_historical_fee_evidence",
                    "initial_cost": float(replay[0, 2]),
                }
        except (IndicatorDomainError, ValueError) as exc:
            checks.append(unavailable("costs", "费用与自融资", exc))
    prepared = None
    try:
        prepared = funding.prepare(source["policy"]["mandate"], candidate.state)
        if prepared:
            result["funding_state"] = {
                k: v
                for k, v in prepared.items()
                if k not in ("inflows", "outflows", "plan")
            }
            arrays.update(
                remaining_inflows=prepared["inflows"],
                remaining_outflows=prepared["outflows"],
            )
            checks.append(
                check(
                    "reconciliation",
                    "剩余支付核对",
                    "passed",
                    "原预算发生额已核对，逾期和漏付事实独立保留。",
                    scope="research",
                )
            )
            if prepared["original_plan_missed_payment"]:
                checks.append(
                    check(
                        "original_plan",
                        "原计划已有漏付",
                        "failed",
                        "续算成功率不能消除过去漏付；当前研究须保留此事实。",
                        scope="history",
                        enforcement="information",
                    )
                )
        else:
            checks.append(
                check(
                    "funding",
                    "剩余资金续算",
                    "not_applicable",
                    "冻结投资目标没有资金预算，未新增概率要求。",
                    enforcement="information",
                )
            )
    except (IndicatorDomainError, ValueError) as exc:
        checks.append(
            unavailable("reconciliation", "剩余支付核对", exc, scope="research")
        )
    if prepared and prepared["remaining_months"] == 0:
        diagnosis = funding.diagnose(
            prepared,
            0.0,
            0.0,
            float(candidate.state.confirmed_investable_value),
            0.0,
            candidate.paths,
            candidate.validation_seed,
        )
        result["funding_results"] = [
            {
                "model_id": "terminal_balance",
                "name": "到期余额核对",
                "enforced": True,
                **diagnosis,
            }
        ]
        checks.append(
            check(
                "funding:terminal",
                "到期余额与漏付核对",
                diagnosis["status"],
                "剩余 0 期，直接核对已确认余额、原目标与未付款，不生成模拟成功率。",
                scope="research",
            )
        )
    if prepared and prepared["remaining_months"] > 0:
        supported = (
            trade is not None
            and candidate.return_basis_confirmed
            and bool(models)
            and candidate.horizon_stationarity_acknowledged
        )
        if candidate.future_weight_rule != "frozen_scalar_proxy":
            supported = False
        if prepared["remaining_months"] * candidate.paths * len(models) > 20_000_000:
            supported = False
            checks.append(
                check(
                    "funding_budget",
                    "模拟计算预算",
                    "unavailable",
                    "逐模型续算超过 2000 万路径月，请减少模拟路径数。",
                )
            )
        if supported:
            fee = float(prepared["plan"]["annual_fee"])
            # Two distinct declared wealth charges combine multiplicatively, once.
            fee = combined_fee(fee, candidate.annual_additional_fee)
            initial = (
                candidate.state.confirmed_investable_value
                if candidate.state.transition_cost_in_balance
                else trade["post_cost_value"]
            )
            rows = []
            for model in models:
                if model["expected_return"] is None:
                    continue
                diagnosis = funding.diagnose(
                    prepared,
                    model["expected_return"],
                    model["volatility"],
                    initial,
                    fee,
                    candidate.paths,
                    candidate.validation_seed if validation else candidate.search_seed,
                )
                rows.append(
                    {
                        "model_id": model["model_id"],
                        "name": model["name"],
                        "enforced": model["enforced"],
                        **diagnosis,
                    }
                )
                if model["enforced"]:
                    checks.append(
                        check(
                            "funding:" + model["model_id"],
                            model["name"] + "：未来条件成功率",
                            diagnosis["status"],
                            "任意剩余整月的标量代理；逐模型 Wilson 区间不是全部模型联合置信。",
                            scope="research",
                            enforcement=(
                                "hard"
                                if prepared["probability_required"]
                                else "information"
                            ),
                            model_ref=model["model_id"],
                            confidence="per_model_95pct",
                            horizon=prepared["remaining_months"],
                            unit="probability",
                        )
                    )
            result["funding_results"] = rows
            result["independent_simulation"] = validation and bool(rows)
        else:
            checks.append(
                check(
                    "funding",
                    "剩余资金续算",
                    "unavailable",
                    "需完整费用与暴露、收益和期限口径确认，并选择月度标量代理。逐产品路径规则需单独验证。",
                )
            )
    if (
        prepared
        and prepared["remaining_months"] > 0
        and trade
        and models
        and candidate.return_basis_confirmed
        and candidate.horizon_stationarity_acknowledged
        and candidate.future_weight_rule != "frozen_scalar_proxy"
    ):
        try:
            path_result = paths.diagnose(
                candidate,
                prepared,
                trade,
                models,
                arrays,
                combined_fee(
                    float(prepared["plan"]["annual_fee"]),
                    candidate.annual_additional_fee,
                ),
                validation=validation,
            )
            result["product_paths"] = path_result
            result["independent_simulation"] = validation and bool(
                path_result["results"]
            )
            checks[:] = [x for x in checks if x["check_id"] != "funding"]
            checks.append(
                check(
                    "funding",
                    "月度标量代理",
                    "not_applicable",
                    "本次由逐产品路径承接资金检验。",
                    enforcement="information",
                )
            )
            checks.append(
                check(
                    "path_cash",
                    "全期产品路径与结算",
                    path_result["status"],
                    "按明确的模型月末结算条款，逐产品计算收益、漂移、费用、现金支付和永久漏付标记。",
                    scope="research",
                    unit="probability",
                    horizon=prepared["remaining_months"],
                )
            )
        except (IndicatorDomainError, ValueError, np.linalg.LinAlgError) as exc:
            checks.append(unavailable("path_cash", "全期产品路径与结算", exc))
    else:
        checks.append(
            check(
                "path_cash",
                "全期产品路径与结算",
                "unavailable",
                "选择逐产品持有规则并补齐现金项目、费用及结算条款后可运行；标量续算不替代此验证。",
            )
        )
    checks.append(
        check(
            "independent_review",
            "认证独立复核",
            "unavailable",
            "本地具名研究记录不构成机构认证独立审批。",
            scope="governance",
            enforcement="review",
        )
    )
    checks.append(
        check(
            "historical_pit",
            "历史可得性",
            "unavailable",
            "本次冻结来源不能证明历史时点已可得。",
            scope="history",
            enforcement="information",
        )
    )
    if candidate.scenario_release_ids:
        # Explicitly selected releases must be resolved by the registered adapter;
        # a missing provider cannot silently become a completed stress result.
        if service.scenarios is None:
            checks.append(
                check(
                    "scenarios",
                    "已发布情景",
                    "unavailable",
                    "当前进程尚未接入匹配此产品轴的已发布情景适配。",
                )
            )
        else:
            result["scenarios"] = service.scenarios(candidate, source, candidate_hash)
            checks.extend(result["scenarios"]["checks"])
    return summarize(result), arrays


@njit((types.float64, types.float64), cache=True, nogil=True)
def combined_fee(first, second):
    return 1.0 - (1.0 - first) * (1.0 - second)


combined_fee.disable_compile()


def summarize(result):
    checks = result["checks"]
    # Numeric failures never become passing evidence through a user waiver.
    research_blockers = [
        x["check_id"]
        for x in checks
        if x["enforcement"] == "hard"
        and (
            x["status"] == "failed"
            or (x["scope"] == "research" and x["status"] == "unavailable")
        )
    ]
    implementation_blockers = [
        x["check_id"]
        for x in checks
        if x["enforcement"] == "hard" and x["status"] in ("failed", "unavailable")
    ]
    result.update(
        research_ready=not research_blockers,
        research_blockers=research_blockers,
        implementation_eligibility=(
            "eligible" if not implementation_blockers else "conditions_incomplete"
        ),
        implementation_blockers=implementation_blockers,
    )
    for item in checks:
        item.update(
            evaluated_subject_hash=result["candidate_hash"],
            checked_at=result["checked_at"],
            execution_version=result["execution"]["source_hash"],
        )
    return result
