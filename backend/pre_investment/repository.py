"""Append-only package revisions on the existing guarded artifact store."""

from __future__ import annotations

import uuid
from backend.custom_indicators.errors import (
    ConflictError,
    NotFoundError,
    ValidationError,
)
from backend.sensitivity.repository import ArtifactRepository, digest_json


class PackageRepository:
    def __init__(self, root):
        self.artifacts = ArtifactRepository(root)

    def history(self, package_id):
        return sorted(
            (
                self.artifacts.get(x["id"], "series")
                for x in self.artifacts.list("series")
                if x.get("artifact_type") == "implementation_package"
                and x.get("scheme_id") == package_id
            ),
            key=lambda x: x["revision"],
        )

    def current(self, package_id):
        versions = self.history(package_id)
        if not versions:
            raise NotFoundError("PACKAGE_NOT_FOUND", "未找到研究包，请重新选择。")
        return versions[-1]

    def list(self):
        ids = dict.fromkeys(
            x["scheme_id"]
            for x in self.artifacts.list("series")
            if x.get("artifact_type") == "implementation_package"
        )
        return [self.current(key) for key in ids]

    def append(self, package_id, expected_revision, fields, *, key):
        request_hash = digest_json(
            {
                "package_id": package_id,
                "expected_revision": expected_revision,
                "fields": fields,
            }
        )
        with self.artifacts.governance_lock.locked():
            replay = self.artifacts.idempotent_result(key, request_hash)
            if replay is not None:
                return replay
            prior = self.current(package_id) if package_id else None
            if expected_revision != (prior["revision"] if prior else 0):
                raise ConflictError(
                    "PACKAGE_REVISION", "研究包已更新，请刷新后再操作；旧版本仍保留。"
                )
            if prior and prior["stage"] == "finalized":
                raise ConflictError(
                    "PACKAGE_FINALIZED", "已定稿版本不可修改，请复制重研。"
                )
            return self.artifacts.save(
                "series",
                {
                    **fields,
                    "artifact_type": "implementation_package",
                    "scheme_id": package_id or f"package-{uuid.uuid4().hex}",
                    "revision": expected_revision + 1,
                    "parent_id": prior["id"] if prior else None,
                },
                idempotency_key=key,
                request_hash=request_hash,
            )

    def report(self, identifier):
        item = self.artifacts.get(identifier, "run")
        if item.get("artifact_type") != "implementation_validation":
            raise ValidationError("PACKAGE_REPORT_TYPE", "所选成果不是实施验证报告。")
        self.artifacts.arrays(identifier)
        return item

    @staticmethod
    def require_report(candidate_hash, report):
        if report.get("candidate_hash") != candidate_hash:
            raise ConflictError(
                "PACKAGE_REPORT_STALE",
                "输入已变化，旧报告不适用于当前候选，请重新验证。",
            )
