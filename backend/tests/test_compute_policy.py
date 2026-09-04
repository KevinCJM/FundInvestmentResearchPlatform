import sys
from pathlib import Path

import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from compute_policy import (
    ComputePolicyError,
    OptimizedThirdPartyModelDeclaration,
    validate_execution_audit,
)


def test_accepts_only_proven_fixed_signature_nopython_audit():
    result = validate_execution_audit({
        "backend": "numba_njit_fixed_signature",
        "nopython": True,
        "python_fallback": 0,
        "kernel_signatures": {"kernel": ["(Array(float64, 1, C),)"]},
    })

    assert result["njit_required"] is True
    assert result["python_fallback"] == 0
    assert result["object_mode"] == 0


def test_njit_audit_accepts_a_predeclared_mutable_and_readonly_signature_set():
    result = validate_execution_audit({
        "backend": "numba_njit_fixed_signature",
        "nopython": True,
        "python_fallback": 0,
        "kernel_signatures": {
            "kernel": ["(Array(float64, mutable),)", "(Array(float64, readonly),)"]
        },
    })

    assert result["request_time_compilation"] == 0


@pytest.mark.parametrize(
    "signatures",
    [
        {"kernel": []},
        {"kernel": ["(float64,)", "(float64,)"]},
        {"kernel": "(float64,)"},
        {"kernel": [""]},
    ],
)
def test_njit_audit_rejects_missing_duplicate_or_malformed_signatures(signatures):
    with pytest.raises(ComputePolicyError, match="固定签名集合"):
        validate_execution_audit({
            "backend": "numba_njit_fixed_signature",
            "nopython": True,
            "python_fallback": 0,
            "kernel_signatures": signatures,
        })


def test_njit_audit_rejects_object_mode():
    with pytest.raises(ComputePolicyError, match="object mode"):
        validate_execution_audit({
            "backend": "numba_njit_fixed_signature",
            "nopython": True,
            "object_mode": 1,
            "python_fallback": 0,
            "kernel_signatures": {"kernel": ["(float64,)"]},
        })


@pytest.mark.parametrize("backend", ["numpy", "pandas", "scipy_slsqp", "python"])
def test_ordinary_third_party_math_is_not_an_exemption(backend: str):
    with pytest.raises(ComputePolicyError):
        validate_execution_audit({"backend": backend, "python_fallback": 0})


def test_optimized_ml_model_isolated_from_njit_may_be_exempt():
    declaration = OptimizedThirdPartyModelDeclaration(
        model_family="machine_learning",
        package="scikit-learn",
        package_version="1.7.0",
        model_name="GradientBoostingClassifier",
        model_version="regime-model-3",
        native_backend="compiled_tree_inference",
        model_fingerprint="sha256:model",
        input_dtype="float64[C]",
        output_dtype="int64[C]",
    )

    result = validate_execution_audit(declaration.audit())

    assert result["execution_backend"] == "optimized_third_party_model"
    assert result["exemption_reason"] == "optimized_ml_dl_nn_model_engine"


def test_third_party_model_cannot_hide_python_callbacks_or_fallback():
    invalid = OptimizedThirdPartyModelDeclaration(
        model_family="neural_network",
        package="example-engine",
        package_version="1",
        model_name="RegimeNet",
        model_version="1",
        native_backend="accelerated-runtime",
        model_fingerprint="sha256:model",
        input_dtype="float32[C]",
        output_dtype="float32[C]",
        python_callback=True,
    ).audit()

    with pytest.raises(ComputePolicyError):
        validate_execution_audit(invalid)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"third_party_package": False}, "自研模型"),
        ({"package": "numpy"}, "不是可豁免"),
        ({"package": "unreviewed-model-engine"}, "登记表"),
        ({"native_backend": "python"}, "原生优化后端"),
        ({"model_engine_scope": "full_pipeline"}, "训练或推理"),
        ({"feature_pipeline_backend": "python"}, "特征工程"),
        ({"postprocess_backend": "pandas"}, "后处理"),
    ],
)
def test_model_exemption_is_isolated_from_all_ordinary_math(override, message):
    audit = OptimizedThirdPartyModelDeclaration(
        model_family="machine_learning",
        package="scikit-learn",
        package_version="1.7.0",
        model_name="GradientBoostingClassifier",
        model_version="regime-model-3",
        native_backend="compiled_tree_inference",
        model_fingerprint="sha256:model",
        input_dtype="float64[C]",
        output_dtype="int64[C]",
    ).audit()
    audit.update(override)

    with pytest.raises(ComputePolicyError, match=message):
        validate_execution_audit(audit)


def test_ordinary_regression_cannot_claim_the_model_exemption():
    invalid = OptimizedThirdPartyModelDeclaration(
        model_family="regression",
        package="statsmodels",
        package_version="1",
        model_name="OLS",
        model_version="1",
        native_backend="linear_algebra",
        model_fingerprint="sha256:ols",
        input_dtype="float64[C]",
        output_dtype="float64[C]",
    ).audit()

    with pytest.raises(ComputePolicyError, match="仅限机器学习"):
        validate_execution_audit(invalid)
