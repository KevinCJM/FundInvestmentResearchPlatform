"""Immutable indicator input snapshots, written only while saving a definition."""
import copy
import json

import numpy as np
import pandas as pd

from backend.sensitivity.repository import ArtifactRepository, digest_json
from backend.custom_indicators.errors import IndicatorDomainError as StorageError
from custom_indicators.errors import ValidationError
from .data import DataBundle, resolve_target


def source_identity(spec):
    return {key: value for key, value in spec.items()
            if key not in {"name", "data_fingerprint", "indicator_data_snapshot"}}


def freeze(spec, bundle, root):
    repository = ArtifactRepository(root / "historical_regime_indicator_sources")
    frame = bundle.frame
    arrays = {
        "observation_dates": frame["observation_date"].to_numpy(dtype="datetime64[ns]").view(np.int64),
        "available_dates": frame["available_at"].to_numpy(dtype="datetime64[ns]").view(np.int64),
        "values": frame["value"].to_numpy(dtype=np.float64),
    }
    # Non-numeric provenance is serialized once, outside the calculation chain.
    encoded = json.loads(frame.drop(columns=["observation_date", "available_at", "value"]).to_json(orient="split", date_format="iso"))
    metadata = {column: [row[index] for row in encoded["data"]]
                for index, column in enumerate(encoded["columns"])}
    fields = {"source": source_identity(spec), "snapshot": copy.deepcopy(bundle.snapshot), "columns": metadata}
    fields["cache_key"] = digest_json(fields)
    try:
        with repository.governance_lock.locked():
            artifact = repository.find("series", fields["cache_key"])
            if artifact is None:
                artifact = repository.save("series", fields, arrays)
    except StorageError as exc:
        raise ValidationError(exc.code, exc.message, "indicator_data_snapshot") from exc
    return {**copy.deepcopy(bundle.snapshot), "frozen_series": {
        "id": artifact["id"], "content_hash": artifact["content_hash"],
    }}


def read(spec, root):
    binding = (spec.get("indicator_data_snapshot") or {}).get("frozen_series")
    if not binding:
        return None
    repository = ArtifactRepository(root / "historical_regime_indicator_sources")
    try:
        artifact = repository.get(binding.get("id", ""), "series")
        arrays = repository.arrays(artifact["id"])
    except StorageError as exc:
        # This shared repository uses backend.* imports; expose the domain error
        # understood by the historical service's bare-package HTTP entry point.
        raise ValidationError(exc.code, exc.message, "indicator_data_snapshot") from exc
    if (artifact["content_hash"] != binding.get("content_hash")
            or artifact["source"] != source_identity(spec)
            or artifact["snapshot"].get("fingerprint") != spec.get("data_fingerprint")):
        raise ValidationError("INDICATOR_SOURCE_BINDING_MISMATCH", "指标冻结序列与所选版本或参数不匹配，请显式重新绑定后保存新修订。")
    # Serialized artifact buffers are read-only. The existing graph kernels have
    # fixed writable-array signatures, so decode into owned input buffers once
    # at this I/O boundary (3 * N * 8 bytes, at most 120 KB for 5,000 points).
    # Subsequent ports/slices reuse these buffers; no persisted/cache bytes change.
    arrays = {name: values.copy() for name, values in arrays.items()}
    frame = pd.DataFrame({**artifact["columns"],
        "observation_date": arrays["observation_dates"].view("datetime64[ns]"),
        "available_at": arrays["available_dates"].view("datetime64[ns]"),
        "value": arrays["values"],
    }, copy=False)
    return DataBundle(frame=frame, snapshot=copy.deepcopy(artifact["snapshot"]))


def resolve_unfrozen(spec, mode, market_root, indicator_service):
    """Replay a legacy binding at its original boundary, verifying exact values.

    Appending observations does not change its window. Revisions inside that
    window still fail closed; unavailable old values are never invented.
    """
    snapshot = spec.get("indicator_data_snapshot") or {}
    cutoff = snapshot.get("latest_available_at") or snapshot.get("last_observation_date")
    bundle = resolve_target(spec, mode, cutoff, market_root, indicator_service)
    expected = spec.get("data_fingerprint")
    if expected and bundle.snapshot.get("fingerprint") != expected:
        raise ValidationError("INDICATOR_DATA_FINGERPRINT_MISMATCH",
                              "原指标窗口的数据已修订，无法还原旧结果。请在数据源中重新选择指标并保存新修订。",
                              "data_fingerprint")
    return bundle
