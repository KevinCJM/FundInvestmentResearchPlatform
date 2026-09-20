"""Package identity, stale evidence and idempotence are independent of UI state."""

import pytest
from backend.custom_indicators.errors import ConflictError
from backend.pre_investment.repository import PackageRepository
from backend.sensitivity.repository import digest_json


def test_package_revisions_are_immutable_and_idempotent(tmp_path):
    repo = PackageRepository(tmp_path / "packages")
    fields = {
        "name": "研究方案",
        "stage": "draft",
        "candidate_hash": digest_json({"weight": 0.5}),
    }
    first = repo.append(None, 0, fields, key="create-package-1")
    assert repo.append(None, 0, fields, key="create-package-1") == first
    second = repo.append(
        first["scheme_id"], 1, {**fields, "name": "修改名称"}, key="update-package-1"
    )
    assert second["revision"] == 2
    assert repo.history(first["scheme_id"])[0] == first
    with pytest.raises(ConflictError):
        repo.append(first["scheme_id"], 1, fields, key="stale-package-1")
    with pytest.raises(ConflictError):
        repo.append(None, 0, {**fields, "name": "不同输入"}, key="create-package-1")


def test_old_report_cannot_approve_new_weights_or_costs():
    old = digest_json({"weights": [0.5, 0.5], "cost": 0.001})
    for changed in (
        {"weights": [0.6, 0.4], "cost": 0.001},
        {"weights": [0.5, 0.5], "cost": 0.002},
    ):
        with pytest.raises(ConflictError):
            PackageRepository.require_report(
                digest_json(changed), {"candidate_hash": old}
            )


def test_finalized_package_requires_clone(tmp_path):
    repo = PackageRepository(tmp_path / "packages")
    first = repo.append(
        None, 0, {"name": "已定稿", "stage": "finalized"}, key="final-package-1"
    )
    with pytest.raises(ConflictError):
        repo.append(first["scheme_id"], 1, {"stage": "draft"}, key="reopen-package-1")
