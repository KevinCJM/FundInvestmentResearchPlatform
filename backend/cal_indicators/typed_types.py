"""Nominal tensor types used by the typed indicator DSL.

The type system deliberately distinguishes a time series from an asset vector.
Both are one-dimensional NumPy arrays at runtime, but they cannot be combined
unless an operator explicitly declares how their axes relate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, TypeAlias


Dimension: TypeAlias = str | int
SUPPORTED_AXES = frozenset({"time", "asset", "window"})
SUPPORTED_KINDS = frozenset({"scalar", "series", "vector", "matrix", "window", "record"})
SUPPORTED_DTYPES = frozenset({"float64", "bool"})
DEFAULT_SEMANTIC_DIMENSION = "dimensionless"
MASK_SEMANTIC_DIMENSION = "mask"
SUPPORTED_SEMANTIC_DIMENSIONS = frozenset(
    {
        DEFAULT_SEMANTIC_DIMENSION,
        "return_decimal",
        "rate_decimal",
        "adjusted_nav",
        "reported_nav",
        "raw_market_price",
        "volume",
        "currency_amount",
        "count",
        "calendar_days",
        "date",
        MASK_SEMANTIC_DIMENSION,
    }
)
LEGACY_SEMANTIC_DIMENSION_ALIASES = {
    "number": DEFAULT_SEMANTIC_DIMENSION,
    "return": "return_decimal",
    "weight": DEFAULT_SEMANTIC_DIMENSION,
    "price": "raw_market_price",
    "price_change": "raw_market_price",
    "currency": "currency_amount",
}


def normalize_semantic_dimension(value: str) -> str:
    normalized = LEGACY_SEMANTIC_DIMENSION_ALIASES.get(value, value)
    if normalized in SUPPORTED_SEMANTIC_DIMENSIONS:
        return normalized
    # Derived dimensions are deterministic compiler metadata, not user-defined
    # executable code. Keeping them namespaced avoids an unbounded public enum.
    if normalized.startswith(("derived:", "squared:", "inverse:")):
        return normalized
    raise ValueError(f"不支持的 semantic_dimension: {value}")


class TypedDslError(ValueError):
    """Stable, machine-readable failure raised by typed DSL compilation/runtime."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        node_id: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.node_id = node_id
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.node_id is not None:
            payload["node_id"] = self.node_id
        if self.details:
            payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True)
class ValueType:
    """A numeric value or mask with nominal axes and semantic provenance.

    ``semantic_dimension`` and ``price_basis`` do not participate in structural
    equality so v2.0 callers that compare ``ValueType.series()`` keep working.
    Operators still inspect both fields explicitly where dimensional
    compatibility matters (notably addition, subtraction and comparisons).
    """

    kind: str
    axes: tuple[str, ...] = ()
    shape: tuple[Dimension, ...] = ()
    dtype: str = "float64"
    semantic_dimension: str = field(default=DEFAULT_SEMANTIC_DIMENSION, compare=False)
    price_basis: str | None = field(default=None, compare=False)
    fields: tuple[tuple[str, "ValueType"], ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in SUPPORTED_KINDS:
            raise ValueError(f"不支持的类型 kind: {self.kind}")
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"typed DSL 仅支持 dtype: {sorted(SUPPORTED_DTYPES)}")
        if not self.semantic_dimension or not self.semantic_dimension.strip():
            raise ValueError("semantic_dimension 不能为空")
        normalized_dimension = normalize_semantic_dimension(self.semantic_dimension)
        object.__setattr__(self, "semantic_dimension", normalized_dimension)
        if self.dtype == "bool" and self.semantic_dimension != MASK_SEMANTIC_DIMENSION:
            raise ValueError("bool 类型必须使用 mask 语义量纲")
        if self.dtype == "bool" and self.price_basis is not None:
            raise ValueError("mask 不能携带价格基准")
        if self.price_basis is not None and not self.price_basis.strip():
            raise ValueError("price_basis 不能为空字符串")
        if len(self.axes) != len(self.shape):
            raise ValueError("axes 与 shape 的维数必须一致")
        if any(axis not in SUPPORTED_AXES for axis in self.axes):
            raise ValueError(f"仅支持命名轴: {sorted(SUPPORTED_AXES)}")
        if any(isinstance(dim, int) and dim <= 0 for dim in self.shape):
            raise ValueError("具体 shape 维度必须为正整数")
        if self.kind == "record":
            names = [name for name, _ in self.fields]
            if not names or len(names) != len(set(names)) or any(not name.isidentifier() or name.startswith('_') for name in names):
                raise ValueError("结构化中间状态需要唯一字段名")
            if any(not value.is_scalar or not value.is_numeric for _, value in self.fields):
                raise ValueError("当前结构化中间状态仅支持数值标量字段")
        elif self.fields:
            raise ValueError("只有 record 可以声明结构化中间状态字段")
        expected_rank = {
            "record": 0,
            "scalar": 0,
            "series": 1,
            "vector": 1,
            "matrix": 2,
            "window": 2,
        }[self.kind]
        if len(self.axes) != expected_rank:
            raise ValueError(f"{self.kind} 必须是 {expected_rank} 维")
        if self.kind == "series" and self.axes != ("time",):
            raise ValueError("series 必须使用 time 轴")
        if self.kind == "vector" and self.axes != ("asset",):
            raise ValueError("vector 必须使用 asset 轴")
        if self.kind == "window" and self.axes != ("time", "window"):
            raise ValueError("window 必须使用 time,window 轴")

    @classmethod
    def scalar(
        cls,
        *,
        dtype: str = "float64",
        semantic_dimension: str = DEFAULT_SEMANTIC_DIMENSION,
        price_basis: str | None = None,
    ) -> "ValueType":
        return cls(
            "scalar",
            dtype=dtype,
            semantic_dimension=semantic_dimension,
            price_basis=price_basis,
        )

    @classmethod
    def series(
        cls,
        length: Dimension = "T",
        *,
        dtype: str = "float64",
        semantic_dimension: str = DEFAULT_SEMANTIC_DIMENSION,
        price_basis: str | None = None,
    ) -> "ValueType":
        return cls(
            "series",
            ("time",),
            (length,),
            dtype=dtype,
            semantic_dimension=semantic_dimension,
            price_basis=price_basis,
        )

    @classmethod
    def vector(
        cls,
        length: Dimension = "N",
        *,
        dtype: str = "float64",
        semantic_dimension: str = DEFAULT_SEMANTIC_DIMENSION,
        price_basis: str | None = None,
    ) -> "ValueType":
        return cls(
            "vector",
            ("asset",),
            (length,),
            dtype=dtype,
            semantic_dimension=semantic_dimension,
            price_basis=price_basis,
        )

    @classmethod
    def matrix(
        cls,
        axes: tuple[str, str] = ("time", "asset"),
        shape: tuple[Dimension, Dimension] = ("T", "N"),
        *,
        dtype: str = "float64",
        semantic_dimension: str = DEFAULT_SEMANTIC_DIMENSION,
        price_basis: str | None = None,
    ) -> "ValueType":
        return cls(
            "matrix",
            axes,
            shape,
            dtype=dtype,
            semantic_dimension=semantic_dimension,
            price_basis=price_basis,
        )

    @classmethod
    def window(
        cls,
        time_length: Dimension = "T",
        window_length: Dimension = "W",
        *,
        semantic_dimension: str = DEFAULT_SEMANTIC_DIMENSION,
        price_basis: str | None = None,
    ) -> "ValueType":
        """Logical causal rolling-window collection; never materialized in production."""

        return cls(
            "window",
            ("time", "window"),
            (time_length, window_length),
            semantic_dimension=semantic_dimension,
            price_basis=price_basis,
        )

    @classmethod
    def mask(
        cls,
        axes: tuple[str, ...] = (),
        shape: tuple[Dimension, ...] = (),
    ) -> "ValueType":
        """Construct a boolean mask with the requested nominal axes."""

        return type_from_axes(
            axes,
            shape,
            dtype="bool",
            semantic_dimension=MASK_SEMANTIC_DIMENSION,
        )

    @property
    def rank(self) -> int:
        return len(self.axes)

    @property
    def is_scalar(self) -> bool:
        return self.kind == "scalar"

    @property
    def is_mask(self) -> bool:
        return self.dtype == "bool"

    @property
    def is_numeric(self) -> bool:
        # ``window`` is a logical compiler state, not a materialized numeric tensor.
        return self.dtype == "float64" and self.kind not in {"record", "window"}

    def with_semantics(
        self,
        semantic_dimension: str,
        price_basis: str | None = None,
    ) -> "ValueType":
        return ValueType(
            self.kind,
            self.axes,
            self.shape,
            dtype=self.dtype,
            semantic_dimension=semantic_dimension,
            price_basis=price_basis,
        )

    def as_mask(self) -> "ValueType":
        return ValueType(
            self.kind,
            self.axes,
            self.shape,
            dtype="bool",
            semantic_dimension=MASK_SEMANTIC_DIMENSION,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            **({"fields": {name: value.to_dict() for name, value in self.fields}} if self.kind == "record" else {}),
            "dtype": self.dtype,
            "axes": list(self.axes),
            "shape": list(self.shape),
            "semantic_dimension": self.semantic_dimension,
            "price_basis": self.price_basis,
            "is_mask": self.is_mask,
            "display": str(self),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValueType":
        dtype = str(payload.get("dtype", "float64"))
        return cls(
            kind=str(payload["kind"]),
            fields=tuple((str(name), cls.from_dict(value)) for name, value in payload.get("fields", {}).items()),
            dtype=dtype,
            axes=tuple(str(axis) for axis in payload.get("axes", [])),
            shape=tuple(payload.get("shape", [])),
            semantic_dimension=str(
                payload.get(
                    "semantic_dimension",
                    MASK_SEMANTIC_DIMENSION
                    if dtype == "bool"
                    else DEFAULT_SEMANTIC_DIMENSION,
                )
            ),
            price_basis=(
                str(payload["price_basis"])
                if payload.get("price_basis") is not None
                else None
            ),
        )

    def __str__(self) -> str:
        if self.kind == "record":
            return "record{" + ", ".join(name for name, _ in self.fields) + "}"
        if self.is_scalar:
            return "mask" if self.is_mask else "scalar"
        axes = ",".join(self.axes)
        shape = ",".join(str(dim) for dim in self.shape)
        kind = "mask" if self.is_mask else self.kind
        return f"{kind}<{axes}>[{shape}]"


def user_type_label(value_type: ValueType) -> str:
    """Return a stable Chinese label for user-facing diagnostics.

    ``ValueType.__str__`` remains the machine-readable DSL representation used
    by API clients and persisted plans.  User messages must not require people
    to understand nominal-axis syntax such as ``series<time>[T]``.
    """

    if value_type.kind == "record":
        fields = tuple(name for name, _ in value_type.fields)
        return "线性拟合中间结果（需提取字段）" if fields and fields[0] == "slope" else "回撤区间（需提取位置）"
    if value_type.is_mask:
        if value_type.axes == ("time",):
            return "时间序列布尔掩码"
        if value_type.axes == ("asset",):
            return "资产布尔掩码"
        if set(value_type.axes) == {"time", "asset"}:
            return "时间—资产布尔掩码"
        return "布尔值"
    if value_type.kind == "scalar":
        return "有限标量"
    if value_type.kind == "series":
        return "时间序列"
    if value_type.kind == "vector":
        return "资产向量"
    if value_type.kind == "window":
        return "滚动窗口集合（中间结果）"
    if value_type.axes == ("asset", "asset"):
        return "资产方阵"
    if set(value_type.axes) == {"time", "asset"}:
        return "时间—资产矩阵"
    return "矩阵"


SCALAR = ValueType.scalar()
TIME_SERIES = ValueType.series()
ASSET_VECTOR = ValueType.vector()
TIME_ASSET_MATRIX = ValueType.matrix()
ASSET_ASSET_MATRIX = ValueType.matrix(("asset", "asset"), ("N", "N"))


def type_from_axes(
    axes: Iterable[str],
    shape: Iterable[Dimension],
    *,
    dtype: str = "float64",
    semantic_dimension: str = DEFAULT_SEMANTIC_DIMENSION,
    price_basis: str | None = None,
) -> ValueType:
    """Construct the nominal kind implied by named axes."""

    axes_tuple = tuple(axes)
    shape_tuple = tuple(shape)
    if not axes_tuple:
        return ValueType.scalar(
            dtype=dtype,
            semantic_dimension=semantic_dimension,
            price_basis=price_basis,
        )
    if axes_tuple == ("time",):
        return ValueType.series(
            shape_tuple[0],
            dtype=dtype,
            semantic_dimension=semantic_dimension,
            price_basis=price_basis,
        )
    if axes_tuple == ("asset",):
        return ValueType.vector(
            shape_tuple[0],
            dtype=dtype,
            semantic_dimension=semantic_dimension,
            price_basis=price_basis,
        )
    if axes_tuple == ("time", "window"):
        if dtype != "float64":
            raise TypedDslError("TYPE_MISMATCH", "滚动窗口中间结果只支持 float64。")
        return ValueType.window(
            shape_tuple[0],
            shape_tuple[1],
            semantic_dimension=semantic_dimension,
            price_basis=price_basis,
        )
    if len(axes_tuple) == 2:
        return ValueType.matrix(  # type: ignore[arg-type]
            axes_tuple,
            shape_tuple,
            dtype=dtype,
            semantic_dimension=semantic_dimension,
            price_basis=price_basis,
        )
    raise TypedDslError(
        "RANK_MISMATCH",
        "typed DSL v2 仅支持 scalar、一维序列/向量和二维矩阵。",
        details={"axes": list(axes_tuple), "shape": list(shape_tuple)},
    )


def elementwise_result(lhs: ValueType, rhs: ValueType) -> ValueType:
    """Apply scalar broadcasting and otherwise require exact nominal shape."""

    if lhs.is_mask or rhs.is_mask:
        raise TypedDslError(
            "TYPE_MISMATCH",
            "数值逐元素运算不能直接使用 mask；请使用 where 或逻辑算子。",
            details={"left": lhs.to_dict(), "right": rhs.to_dict()},
        )

    if lhs.is_scalar:
        return rhs
    if rhs.is_scalar:
        return lhs
    if lhs.axes != rhs.axes:
        raise TypedDslError(
            "AXIS_MISMATCH",
            f"逐元素运算的命名轴不一致: {lhs} 与 {rhs}",
            details={"left": lhs.to_dict(), "right": rhs.to_dict()},
        )
    if lhs.shape != rhs.shape:
        raise TypedDslError(
            "SHAPE_MISMATCH",
            f"逐元素运算的 shape 不一致: {lhs} 与 {rhs}",
            details={"left": lhs.to_dict(), "right": rhs.to_dict()},
        )
    return lhs


def require_same_type(lhs: ValueType, rhs: ValueType, operator: str) -> ValueType:
    if (
        lhs.kind != rhs.kind
        or lhs.axes != rhs.axes
        or lhs.shape != rhs.shape
        or lhs.dtype != rhs.dtype
    ):
        code = "AXIS_MISMATCH" if lhs.axes != rhs.axes else "SHAPE_MISMATCH"
        raise TypedDslError(
            code,
            f"{operator} 要求两个输入具有相同类型，实际为 {lhs} 与 {rhs}",
            details={"left": lhs.to_dict(), "right": rhs.to_dict()},
        )
    return lhs


def symbolic_elements(value_type: ValueType) -> str:
    if value_type.is_scalar:
        return "1"
    return "*".join(str(dim) for dim in value_type.shape)
