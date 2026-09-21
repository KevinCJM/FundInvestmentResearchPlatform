from __future__ import annotations

"""Fail-closed execution policy for every user-reachable numerical chain."""

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping


NJIT_BACKEND = "numba_njit_fixed_signature"
CPP_AOT_BACKEND = "cpp_aot"
NATIVE_NUMERICAL_BACKENDS = frozenset({NJIT_BACKEND, CPP_AOT_BACKEND})
THIRD_PARTY_BACKEND = "optimized_third_party_model"
ALLOWED_THIRD_PARTY_MODEL_FAMILIES = frozenset(
    {"machine_learning", "deep_learning", "neural_network"}
)
DISALLOWED_MODEL_ENGINE_PACKAGES = frozenset(
    {"", "internal", "self", "python", "numpy", "pandas", "scipy", "numba"}
)
APPROVED_OPTIMIZED_MODEL_PACKAGES = frozenset(
    {
        "catboost",
        "hmmlearn",
        "jax",
        "jaxlib",
        "lightgbm",
        "onnxruntime",
        "scikit-learn",
        "tensorflow",
        "tensorflow-cpu",
        "torch",
        "xgboost",
    }
)
DISALLOWED_NATIVE_MODEL_BACKENDS = frozenset(
    {"", "interpreted", "numpy", "pandas", "python", "scipy", "numba"}
)


class ComputePolicyError(ValueError):
    """Raised when a numerical execution lane cannot prove policy compliance."""


@dataclass(frozen=True)
class OptimizedThirdPartyModelDeclaration:
    model_family: str
    package: str
    package_version: str
    model_name: str
    model_version: str
    native_backend: str
    model_fingerprint: str
    input_dtype: str
    output_dtype: str
    third_party_package: bool = True
    native_optimized: bool = True
    isolated_array_contract: bool = True
    model_engine_scope: str = "training_or_inference_only"
    feature_pipeline_backend: str = NJIT_BACKEND
    postprocess_backend: str = NJIT_BACKEND
    python_callback: bool = False
    python_fallback: int = 0
    execution_backend: str = THIRD_PARTY_BACKEND

    def audit(self) -> dict[str, Any]:
        return asdict(self)


def _backend(audit: Mapping[str, Any]) -> str:
    return str(audit.get("backend") or audit.get("execution_backend") or "")


def _signature_groups(audit: Mapping[str, Any]) -> Mapping[str, Any]:
    value = audit.get("kernel_signatures") or audit.get("compiled_signatures") or {}
    return value if isinstance(value, Mapping) else {}


def _validate_cpp_aot(audit: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "audit_schema": "cpp-aot-execution-1", "engine": "calmetrics_engine",
        "operator_registry_version": "canonical-native-1", "typed_ir_version": "cpp-typed-ir-1",
        "input_dtype": "float64", "output_dtype": "float64",
    }
    if any(audit.get(key) != value for key, value in required.items()):
        raise ComputePolicyError("C++ AOT 执行契约或引擎不受支持")
    if audit.get("native_aot") is not True:
        raise ComputePolicyError("C++ AOT 必须使用构建阶段编译的原生内核")
    for key in ("python_fallback", "python_operator_calls", "python_worker_callbacks", "request_time_compilation"):
        if type(audit.get(key)) is not int or audit[key] != 0:
            raise ComputePolicyError("C++ AOT 禁止 Python 数值回调、回退或请求期编译")
    if not isinstance(audit.get("engine_version"), str) or not audit["engine_version"].strip():
        raise ComputePolicyError("C++ AOT 缺少引擎版本")
    if not re.fullmatch(r"[0-9a-f]{64}", str(audit.get("engine_build_id", ""))):
        raise ComputePolicyError("C++ AOT 缺少有效构建身份")
    if not re.fullmatch(r"native-[1-9][0-9]*-[0-9a-f]{32}", str(audit.get("plan_fingerprint", ""))):
        raise ComputePolicyError("C++ AOT 缺少有效计划身份")
    cpu, tokens = audit.get("cpu_budget"), audit.get("cpu_tokens")
    if type(cpu) is not int or type(tokens) is not int or not 1 <= tokens <= cpu <= 1024:
        raise ComputePolicyError("C++ AOT CPU 执行证明无效")
    if audit.get("result_lifetime") not in {"independent", "borrowed_until_next_run"}:
        raise ComputePolicyError("C++ AOT 必须声明结果所有权")
    return {**audit, "execution_backend": CPP_AOT_BACKEND, "njit_required": False}


def validate_execution_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one numerical lane and return a normalized immutable audit.

    Ordinary numerical code uses fixed-signature Numba or the audited C++ AOT
    engine.  A third-party exception is
    intentionally narrow and only covers optimized ML/DL/neural-network model
    engines behind an isolated array contract.
    """

    if not isinstance(audit, Mapping):
        raise ComputePolicyError("执行证明必须是对象")
    if audit.get("backend") and audit.get("execution_backend") and audit["backend"] != audit["execution_backend"]:
        raise ComputePolicyError("执行后端声明冲突")
    backend = _backend(audit)
    if backend == CPP_AOT_BACKEND:
        return _validate_cpp_aot(audit)
    fallback = audit.get("python_fallback")
    if fallback != 0:
        raise ComputePolicyError("数值计算链路禁止 Python 慢速回退")

    if backend == NJIT_BACKEND:
        if audit.get("nopython") is not True:
            raise ComputePolicyError("NJIT 链路必须证明 nopython=true")
        if audit.get("object_mode", 0) != 0:
            raise ComputePolicyError("NJIT 链路禁止 object mode")
        if audit.get("request_time_compilation", 0) != 0:
            raise ComputePolicyError("NJIT 链路禁止请求期间编译")
        signatures = _signature_groups(audit)
        if not signatures or any(
            not isinstance(values, (list, tuple))
            or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(set(values)) != len(values)
            for values in signatures.values()
        ):
            raise ComputePolicyError("NJIT 链路必须声明非空、无重复的预编译固定签名集合")
        normalized = dict(audit)
        normalized["execution_backend"] = NJIT_BACKEND
        normalized["njit_required"] = True
        normalized["python_fallback"] = 0
        normalized["object_mode"] = 0
        normalized["request_time_compilation"] = 0
        return normalized

    if backend == THIRD_PARTY_BACKEND:
        family = str(audit.get("model_family") or "")
        if family not in ALLOWED_THIRD_PARTY_MODEL_FAMILIES:
            raise ComputePolicyError("第三方豁免仅限机器学习、深度学习或神经网络模型")
        required_text = (
            "package",
            "package_version",
            "model_name",
            "model_version",
            "native_backend",
            "model_fingerprint",
            "input_dtype",
            "output_dtype",
        )
        missing = [field for field in required_text if not str(audit.get(field) or "").strip()]
        if missing:
            raise ComputePolicyError(f"第三方模型执行声明缺少字段：{', '.join(missing)}")
        if audit.get("third_party_package") is not True:
            raise ComputePolicyError("自研模型或算法不能申请第三方优化引擎豁免")
        package = str(audit.get("package") or "").strip().lower()
        if package in DISALLOWED_MODEL_ENGINE_PACKAGES:
            raise ComputePolicyError("普通 Python/NumPy/SciPy/Numba 数学库不是可豁免的模型引擎")
        if package not in APPROVED_OPTIMIZED_MODEL_PACKAGES:
            raise ComputePolicyError("第三方模型包尚未进入受审计的优化引擎登记表")
        native_backend = str(audit.get("native_backend") or "").strip().lower()
        if native_backend in DISALLOWED_NATIVE_MODEL_BACKENDS:
            raise ComputePolicyError("第三方模型必须由独立的原生优化后端执行")
        if audit.get("native_optimized") is not True:
            raise ComputePolicyError("第三方模型必须声明其原生优化执行后端")
        if audit.get("isolated_array_contract") is not True:
            raise ComputePolicyError("第三方模型必须通过独立 ndarray 契约隔离")
        if audit.get("python_callback") is not False:
            raise ComputePolicyError("第三方模型计算期间禁止 Python 数值回调")
        if audit.get("model_engine_scope") != "training_or_inference_only":
            raise ComputePolicyError("第三方豁免只能覆盖模型训练或推理本身")
        if audit.get("feature_pipeline_backend") not in NATIVE_NUMERICAL_BACKENDS:
            raise ComputePolicyError("第三方模型的特征工程与缩放必须使用固定签名 NJIT 或 C++ AOT")
        if audit.get("postprocess_backend") not in NATIVE_NUMERICAL_BACKENDS:
            raise ComputePolicyError("第三方模型的路径、统计和归因后处理必须使用固定签名 NJIT 或 C++ AOT")
        normalized = dict(audit)
        normalized["execution_backend"] = THIRD_PARTY_BACKEND
        normalized["njit_required"] = False
        normalized["exemption_reason"] = "optimized_ml_dl_nn_model_engine"
        normalized["python_fallback"] = 0
        return normalized

    raise ComputePolicyError(
        "普通数学、矩阵、回归、导数、统计、组合和回测仅允许 fixed-signature NJIT 或 C++ AOT；"
        "当前执行后端不合规"
    )


def validate_execution_graph(*audits: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not audits:
        raise ComputePolicyError("计算链路没有提供执行审计")
    return [validate_execution_audit(audit) for audit in audits]


__all__ = [
    "ALLOWED_THIRD_PARTY_MODEL_FAMILIES",
    "APPROVED_OPTIMIZED_MODEL_PACKAGES",
    "ComputePolicyError",
    "DISALLOWED_MODEL_ENGINE_PACKAGES",
    "DISALLOWED_NATIVE_MODEL_BACKENDS",
    "NJIT_BACKEND",
    "OptimizedThirdPartyModelDeclaration",
    "THIRD_PARTY_BACKEND",
    "validate_execution_audit",
    "validate_execution_graph",
]
