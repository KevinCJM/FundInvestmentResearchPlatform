from __future__ import annotations

import pytest

from cal_indicators.typed_types import (
    SCALAR,
    ValueType,
    TypedDslError,
    elementwise_result,
    require_same_type,
    type_from_axes,
    user_type_label,
)


def test_typed_dsl_error_serializes_stable_fields() -> None:
    error = TypedDslError(
        "TYPE_MISMATCH",
        "类型不一致。",
        node_id=3,
        details={"expected": "scalar"},
    )

    assert error.to_dict() == {
        "code": "TYPE_MISMATCH",
        "message": "类型不一致。",
        "node_id": 3,
        "details": {"expected": "scalar"},
    }


@pytest.mark.parametrize(
    "factory",
    (
        lambda: ValueType("unknown"),
        lambda: ValueType("scalar", dtype="float32"),
        lambda: ValueType("scalar", ("time",), ("T",)),
        lambda: ValueType("series", ("asset",), ("N",)),
        lambda: ValueType("vector", ("time",), ("T",)),
        lambda: ValueType.matrix(("time", "unknown"), ("T", "N")),
        lambda: ValueType.vector(0),
    ),
)
def test_value_type_rejects_invalid_nominal_definitions(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_value_type_round_trip_and_axis_constructor() -> None:
    matrix = ValueType.matrix(("asset", "time"), (5, 12))

    assert ValueType.from_dict(matrix.to_dict()) == matrix
    assert type_from_axes((), ()) == SCALAR
    assert type_from_axes(("time",), (12,)) == ValueType.series(12)
    assert type_from_axes(("asset",), (5,)) == ValueType.vector(5)
    assert type_from_axes(("asset", "time"), (5, 12)) == matrix

    with pytest.raises(TypedDslError) as caught:
        type_from_axes(("time", "asset", "time"), (2, 3, 4))
    assert caught.value.code == "RANK_MISMATCH"


def test_value_type_preserves_semantic_dimension_price_basis_and_masks() -> None:
    adjusted_nav = ValueType.series(
        semantic_dimension="adjusted_nav",
        price_basis="adjusted_nav",
    )
    restored = ValueType.from_dict(adjusted_nav.to_dict())

    assert restored.semantic_dimension == "adjusted_nav"
    assert restored.price_basis == "adjusted_nav"
    assert restored.dtype == "float64"

    mask = ValueType.mask(("time",), ("T",))
    assert mask.is_mask
    assert not mask.is_numeric
    assert mask.to_dict()["display"] == "mask<time>[T]"
    assert ValueType.from_dict(mask.to_dict()).is_mask


def test_value_type_has_chinese_labels_for_user_facing_diagnostics() -> None:
    assert user_type_label(ValueType.scalar()) == "有限标量"
    assert user_type_label(ValueType.series()) == "时间序列"
    assert user_type_label(ValueType.vector()) == "资产向量"
    assert user_type_label(ValueType.matrix()) == "时间—资产矩阵"
    assert user_type_label(ValueType.matrix(("asset", "asset"), ("N", "N"))) == "资产方阵"
    assert user_type_label(ValueType.mask(("time",), ("T",))) == "时间序列布尔掩码"


def test_legacy_semantic_aliases_are_normalized_but_invalid_masks_fail() -> None:
    assert ValueType.series(semantic_dimension="return").semantic_dimension == (
        "return_decimal"
    )
    assert ValueType.scalar(semantic_dimension="number").semantic_dimension == (
        "dimensionless"
    )

    with pytest.raises(ValueError):
        ValueType.scalar(dtype="bool", semantic_dimension="dimensionless")
    with pytest.raises(ValueError):
        ValueType.series(semantic_dimension="unknown_dimension")


def test_elementwise_rules_only_allow_scalar_broadcast_or_exact_tensor() -> None:
    series = ValueType.series("T")
    assert elementwise_result(SCALAR, series) == series
    assert elementwise_result(series, SCALAR) == series
    assert elementwise_result(series, series) == series

    with pytest.raises(TypedDslError) as axis_error:
        elementwise_result(series, ValueType.vector("T"))
    assert axis_error.value.code == "AXIS_MISMATCH"

    with pytest.raises(TypedDslError) as shape_error:
        elementwise_result(ValueType.series("T"), ValueType.series("M"))
    assert shape_error.value.code == "SHAPE_MISMATCH"

    with pytest.raises(TypedDslError) as same_error:
        require_same_type(ValueType.vector("N"), ValueType.vector("M"), "dot")
    assert same_error.value.code == "SHAPE_MISMATCH"

    with pytest.raises(TypedDslError) as mask_error:
        elementwise_result(series, ValueType.mask(("time",), ("T",)))
    assert mask_error.value.code == "TYPE_MISMATCH"
