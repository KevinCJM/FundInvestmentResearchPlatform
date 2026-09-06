"""Small integration seam for the existing Tushare acquisition coordinator."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from .models import Pagination
from .presets import default_interfaces, default_source
from .runtime import ConfiguredTushareClient
from .store import DEFAULT_ROOT, SourceStore


def configuration_fingerprint(root: Path = DEFAULT_ROOT) -> str:
    path = Path(root) / "data_sources.sqlite3"
    if path.exists():
        store = SourceStore(root)
        source = store.get("source", "tushare")["config"]
        interfaces = [item["config"] for item in store.list("interface") if item["config"]["source_id"] == "tushare"]
    else:
        source = default_source().model_dump(mode="json")
        interfaces = [item.model_dump(mode="json") for item in default_interfaces()]
    return hashlib.sha256(json.dumps({"source": source, "interfaces": sorted(interfaces, key=lambda item: item["id"])}, sort_keys=True).encode()).hexdigest()


def page_size(pro, api: str, fallback: int) -> int:
    pagination = getattr(getattr(pro, api), "pagination_config", None)
    return pagination.page_size if isinstance(pagination, Pagination) and pagination.mode != "none" else fallback


def create_client(credential: str, args) -> ConfiguredTushareClient:
    # Smoke runs never put candidate data in the live workspace.
    root = Path(args.output_dir) if getattr(args, "smoke", False) else DEFAULT_ROOT
    client = ConfiguredTushareClient(credential, root=root, capture=True)
    args.source_configuration_hash = client.configuration_hash
    args.max_workers = min(getattr(args, "max_workers", 16), client.source.policy.max_concurrency)
    for api, attribute in (("fund_basic", "max_fund_basic_pages"), ("fund_nav", "max_fund_nav_pages"), ("fund_manager", "max_fund_manager_pages")):
        configured = client.interfaces[api].pagination.max_pages
        setattr(args, attribute, min(getattr(args, attribute, configured), configured))
    configured = client.interfaces["fund_nav"].pagination.page_size
    args.fund_nav_page_size = min(getattr(args, "fund_nav_page_size", configured), configured)
    return client


def pagination_defaults(root: Path = DEFAULT_ROOT) -> dict[str, str]:
    if (Path(root) / "data_sources.sqlite3").exists():
        store = SourceStore(root)
        configs = {api: store.get("interface", "tushare." + api)["config"]["pagination"] for api in ("fund_basic", "fund_nav", "fund_manager")}
    else:
        configs = {item.api_name: item.pagination.model_dump() for item in default_interfaces()}
    return {"basic_pages": str(configs["fund_basic"]["max_pages"]), "nav_pages": str(configs["fund_nav"]["max_pages"]), "nav_page_size": str(configs["fund_nav"]["page_size"]), "manager_pages": str(configs["fund_manager"]["max_pages"])}
