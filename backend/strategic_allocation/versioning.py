"""投前产物统一版本与依赖状态：只从已保存记录派生，不改写冻结字节或内容 hash。

规则见 docs/pre-investment/versioning.md。
"""
from typing import Iterable

from backend.custom_indicators.errors import IndicatorDomainError
from backend.product_pools.errors import ProductPoolError
from .scope_facts import scope_weight_limits

CURRENT, SUPERSEDED, RETIRED, MISSING = "current", "superseded", "retired", "missing"
READY, STALE, BLOCKED = "ready", "stale", "blocked"


def version_index(records: Iterable[dict], parent_key: str, retired: Iterable[str] = ()) -> dict[str, dict]:
    """按 supersedes 链为每个 id 给出系列、版本号、状态与同系列当前版本。"""
    parent = {item["id"]: item.get(parent_key) for item in records}
    superseded, retired = {value for value in parent.values() if value}, set(retired)
    index = {}
    for identifier in parent:
        root, number = identifier, 1
        while parent.get(root) in parent:
            root, number = parent[root], number + 1
        status = RETIRED if identifier in retired else SUPERSEDED if identifier in superseded else CURRENT
        index[identifier] = {"lineage_id": root, "number": number, "status": status}
    latest = {}
    for identifier, info in index.items():
        if info["status"] != SUPERSEDED and info["number"] > latest.get(info["lineage_id"], (None, 0))[1]:
            latest[info["lineage_id"]] = (identifier, info["number"])
    for info in index.values():
        head = latest.get(info["lineage_id"])
        # 系列当前版本被删除时，整个系列停止引用，没有可升级的目标。
        alive = head and index[head[0]]["status"] == CURRENT
        info.update(latest_id=head[0] if alive else None, latest_number=head[1] if alive else None)
    return index


def single_version(status: str = CURRENT, lineage_id: str | None = None, number: int = 1) -> dict:
    """不可修改的产物：每份即一个系列。"""
    return {"lineage_id": lineage_id, "number": number, "status": status,
            "latest_id": lineage_id if status == CURRENT else None,
            "latest_number": number if status == CURRENT else None}


def reference(kind: str, identifier: str | None, name: str | None, version: dict | None,
              usable: dict | None = None, equivalent: bool = False) -> dict:
    """下游看到的一条上游：钉住的版本、最新版本与状态；读不到即 missing。

    equivalent 表示旧版本与当前版本的经济配置相同（如只改了说明），下游无需更新。
    """
    version = version or {}
    return {"kind": kind, "id": identifier, "name": name, "number": version.get("number"),
            "latest_id": version.get("latest_id"), "latest_number": version.get("latest_number"),
            "status": version.get("status", MISSING), "equivalent": equivalent,
            "usable": (usable or {}).get("status", BLOCKED if version.get("status") == SUPERSEDED
                                        and version.get("latest_id") is None else READY)}


def usability(version: dict, upstream: Iterable[dict]) -> dict:
    """自身与直接上游共同决定能否接入新的下游工作；上游自身的可用性沿链传递。"""
    reasons = []
    if (version["status"] in (RETIRED, MISSING)
            or version["status"] == SUPERSEDED and version.get("latest_id") is None):
        reasons.append({"code": "self_retired"})
    elif version["status"] == SUPERSEDED:
        reasons.append({"code": "self_superseded", "latest_number": version.get("latest_number")})
    for ref in upstream:
        if ref["status"] in (RETIRED, MISSING) or ref["usable"] == BLOCKED:
            reasons.append({"code": "upstream_deleted", "kind": ref["kind"], "name": ref["name"]})
        elif (ref["status"] == SUPERSEDED and not ref.get("equivalent")) or ref["usable"] == STALE:
            reasons.append({"code": "upstream_superseded", "kind": ref["kind"], "name": ref["name"],
                            "number": ref["number"], "latest_number": ref["latest_number"]})
    blocked = any(item["code"] in ("self_retired", "upstream_deleted") for item in reasons)
    return {"status": BLOCKED if blocked else STALE if reasons else READY, "reasons": reasons}


# 类型：(artifact_type, 修改链字段, 删除记录类型, 删除记录中的 id 字段)
STRATEGIC_TYPES = {
    "mandate": ("investment_mandate", "supersedes_mandate_id", "investment_mandate_retirement", "mandate_id"),
    "strategic_scope": ("strategic_universe", "supersedes_universe_id", "strategic_universe_retirement", "universe_id"),
    "cma": ("capital_market_assumptions", "supersedes_cma_id", "cma_retirement", "cma_id"),
}
SCOPE_KINDS = {"strategy_first": "strategic_scope", "product_first": "product_scope"}


class ResearchVersions:
    """一次请求内的版本快照：战略侧存储只扫描一次，产品范围按需加载。"""

    def __init__(self, artifacts, product_repository=None):
        """product_repository 为返回产品池存储的函数，仅在遇到产品范围时调用。"""
        from backend.product_pools.scope_lifecycle import scope_mandate_fields
        self.artifacts, self._product_repository, self._bound = artifacts, product_repository, scope_mandate_fields
        wanted = {value[0]: key for key, value in STRATEGIC_TYPES.items()}
        by_kind = {key: [] for key in STRATEGIC_TYPES}
        self.records = {}
        for summary in artifacts.list("series"):
            kind = wanted.get(summary.get("artifact_type"))
            if kind:
                item = artifacts.get(summary["id"], "series")
                self.records[item["id"]] = item
                by_kind[kind].append(item)
        retired = {key: set() for key in STRATEGIC_TYPES}
        by_retirement = {value[2]: (key, value[3]) for key, value in STRATEGIC_TYPES.items()}
        for summary in artifacts.list("retirement"):
            item = artifacts.get(summary["id"], "retirement")
            kind, field = by_retirement.get(item.get("artifact_type"), (None, None))
            if kind and item.get(field):
                retired[kind].add(item[field])
        self.retired = retired
        self.index = {key: version_index(by_kind[key], STRATEGIC_TYPES[key][1], retired[key]) for key in STRATEGIC_TYPES}
        self._scopes: dict = {}

    def _product(self) -> None:
        if "product_scope" not in self.index:
            repository = self._product_repository() if self._product_repository else None
            snapshots = repository.list_universe_snapshots() if repository else []
            retired = {str(item.get("snapshot_id")) for item in (repository.list_universe_retirements() if repository else [])}
            snapshots = [{**item, "id": str(item["id"])} for item in snapshots]
            self.records.update({item["id"]: item for item in snapshots})
            self.index["product_scope"] = version_index(snapshots, "supersedes_snapshot_id", retired)

    def version(self, kind: str, identifier: str | None) -> dict | None:
        if kind == "product_scope":
            self._product()
        return self.index[kind].get(identifier) if identifier else None

    def mandate(self, identifier: str | None) -> dict:
        record = self.records.get(identifier) or {}
        return reference("mandate", identifier, record.get("name"), self.version("mandate", identifier))

    def scope(self, path: str | None, identifier: str | None) -> tuple[dict, str | None]:
        """范围引用及其绑定的目标 id；读不到的范围按缺失处理。"""
        key = (path, identifier)
        if key not in self._scopes:
            kind = SCOPE_KINDS.get(path, "strategic_scope")
            version = self.version(kind, identifier)
            record = self.records.get(identifier) if version else None
            try:
                bound = self._bound(self.artifacts.root, "strategic" if kind == "strategic_scope" else "product",
                                    record) if record else {}
            except (IndicatorDomainError, ProductPoolError):  # 关联不可读即视为范围不可用
                record, bound = None, {}
            name = (record or {}).get("name")
            latest = self.records.get((version or {}).get("latest_id")) if record else None
            # 收益估计指纹不包含权重边界；生命周期等价还须核对冻结的配置约束。
            # 读取时核对可修复已有记录，不改写其历史 hash。
            equivalent = bool(record and latest and version["status"] == SUPERSEDED and record.get("scope_fingerprint")
                              and record.get("scope_fingerprint") == latest.get("scope_fingerprint")
                              and scope_weight_limits(record["definition"]["assets"])
                              == scope_weight_limits(latest["definition"]["assets"]))
            self._scopes[key] = (reference(kind, identifier, name, version if record else None, equivalent=equivalent),
                                 bound.get("mandate_id"))
        return self._scopes[key]

    def scope_state(self, path: str | None, identifier: str | None) -> dict:
        """范围自身的版本、上游目标与可用性。"""
        ref, mandate_id = self.scope(path, identifier)
        upstream = [self.mandate(mandate_id)] if mandate_id else []
        return {"version": self.version(ref["kind"], identifier) or single_version(MISSING),
                "upstream": upstream,
                "usable": usability(self.version(ref["kind"], identifier) or single_version(MISSING), upstream)}

    def cma_state(self, item: dict) -> dict:
        """LTCMA 的版本、上游（目标经范围推导）与可用性。"""
        definition = item["definition"]
        scope_id = definition.get("strategic_universe_id")
        path = "strategy_first" if scope_id else "product_first" if definition.get("alloc_name") else None
        if path == "product_first":
            scope_id = item.get("source_snapshot", {}).get("universe_snapshot_id")
        upstream = []
        if scope_id:  # 早期产品路径 LTCMA 只有大类名、没有范围记录，按未关联处理。
            scope, mandate_id = self.scope(path, scope_id)
            upstream = ([self.mandate(mandate_id)] if mandate_id else []) + [scope]
        version = self.version("cma", item["id"]) or single_version(MISSING)
        return {"research_path": path, "version": version, "upstream": upstream,
                "usable": usability(version, upstream)}


def research_versions(strategic_root, universe_dir) -> ResearchVersions:
    """TAA、研究包等不持有战略服务的模块，从同一存储根建立版本快照。"""
    from pathlib import Path
    from backend.product_pools.repository import ProductPoolRepository
    from backend.product_pools.scope_lifecycle import strategic_artifacts_root
    from backend.sensitivity.repository import ArtifactRepository
    return ResearchVersions(ArtifactRepository(strategic_artifacts_root(strategic_root)),
                            lambda: ProductPoolRepository(Path(universe_dir) / "product_pools.json"))


def policy_state(lineage: ResearchVersions, baseline: dict) -> dict:
    """SAA 方案不可修改：版本恒为 v1，可用性由钉住的目标与 LTCMA 决定（范围经 LTCMA 传递）。"""
    version = single_version(lineage_id=baseline["id"])
    policy = baseline.get("policy")
    if not policy:  # 早期手工基线没有冻结上游，不虚构依赖。
        return {"version": version, "upstream": [], "usable": usability(version, [])}
    multi = policy.get("multi_cma") or {}
    ids = [ref["cma_id"] for ref in multi.get("refs") or multi.get("sources") or []] or [policy.get("cma_id")]
    cmas = []
    for identifier in ids:
        record = lineage.records.get(identifier)
        state = lineage.cma_state(record) if record else None
        cmas.append(reference("cma", identifier, record and record["name"], state and state["version"],
                              state and state["usable"]))
    upstream = [lineage.mandate(policy.get("mandate_id")), *cmas]
    return {"version": version, "upstream": upstream, "usable": usability(version, upstream)}


def decision_state(lineage: ResearchVersions, decision: dict, baseline: dict | None) -> dict:
    """TAA 决策不可修改：可用性跟随其 SAA 方案。"""
    version = single_version(lineage_id=decision["id"])
    policy = policy_state(lineage, baseline) if baseline else None
    upstream = [reference("saa_policy", decision.get("baseline_id"), baseline and baseline.get("name"),
                          policy and policy["version"], policy and policy["usable"])]
    return {"version": version, "upstream": upstream, "usable": usability(version, upstream)}


def source_state(lineage: ResearchVersions, kind: str, identifier: str, baselines: dict, decisions: dict) -> dict:
    """研究包来源（SAA 方案或 TAA 决策）的引用与可用性。"""
    if kind == "taa_decision":
        decision = decisions.get(identifier)
        state = decision_state(lineage, decision, baselines.get(decision.get("baseline_id"))) if decision else None
        name = decision and decision.get("name")
    else:
        baseline = baselines.get(identifier)
        state = policy_state(lineage, baseline) if baseline else None
        name = baseline and baseline.get("name")
    return reference(kind, identifier, name, state and state["version"], state and state["usable"])


if __name__ == "__main__":
    index = version_index([{"id": "a"}, {"id": "b", "p": "a"}, {"id": "c", "p": "b"}, {"id": "x"}], "p", {"x"})
    assert index["a"] == {"lineage_id": "a", "number": 1, "status": SUPERSEDED, "latest_id": "c", "latest_number": 3}
    assert index["c"]["status"] == CURRENT and index["x"]["latest_id"] is None
    old = reference("mandate", "a", "目标", index["a"])
    assert usability(index["c"], [old])["status"] == STALE
    assert usability(index["c"], [reference("scope", "gone", None, None)])["status"] == BLOCKED
