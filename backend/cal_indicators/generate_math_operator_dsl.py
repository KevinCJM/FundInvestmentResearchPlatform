import ast
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

MODULE_NAME = "cal_indicators.numba_finance_math"
CURRENT_DIR = Path(__file__).resolve().parent
TARGET_FILE = CURRENT_DIR / "numba_finance_math.py"
OUTPUT_FILE = CURRENT_DIR / "numba_finance_math_dsl.json"
DSL_VERSION = "1.0.0"


def escape_identifier(identifier: str) -> str:
    return identifier.replace("_", r"\_")


def format_argument(name: str) -> str:
    return r"\mathrm{" + escape_identifier(name) + "}"


def latex_operator_name(name: str) -> str:
    return escape_identifier(name)


def _binary(op: str) -> Callable[[List[str]], str]:
    def formatter(args: List[str]) -> str:
        if len(args) < 2:
            raise ValueError("binary operator requires two arguments")
        return f"\\left({args[0]}\\right) {op} \\left({args[1]}\\right)"

    return formatter


def _unary(prefix: str, suffix: str = "") -> Callable[[List[str]], str]:
    def formatter(args: List[str]) -> str:
        if not args:
            raise ValueError("unary operator requires one argument")
        return f"{prefix}{args[0]}{suffix}"

    return formatter


LATEX_PATTERNS: Dict[str, Callable[[List[str]], str]] = {
    "add": _binary("+"),
    "subtract": _binary("-"),
    "multiply": _binary(r"\times"),
    "divide": lambda args: f"\\frac{{{args[0]}}}{{{args[1]}}}",
    "negate": _unary("-\\left(", "\\right)"),
    "absolute": _unary("\\left|", "\\right|"),
    "power": lambda args: f"\\left({args[0]}\\right)^{{{args[1]}}}",
    "maximum": lambda args: f"\\max\\left({args[0]}, {args[1]}\\right)",
    "minimum": lambda args: f"\\min\\left({args[0]}, {args[1]}\\right)",
    "reciprocal": lambda args: f"\\frac{{1}}{{{args[0]}}}",
    "clip": lambda args: f"\\operatorname{{clip}}\\left({args[0]}, {args[1]}, {args[2]}\\right)",
    "sequence_sum": lambda args: f"\\sum\\left({args[0]}\\right)",
    "sequence_mean": lambda args: f"\\overline{{{args[0]}}}",
    "sequence_prod": lambda args: f"\\prod\\left({args[0]}\\right)",
    "sequence_std": lambda args: f"\\operatorname{{std}}\\left({args[0]}, {args[1]}\\right)",
    "sequence_variance": lambda args: f"\\operatorname{{var}}\\left({args[0]}, {args[1]}\\right)",
    "sequence_min": lambda args: f"\\min\\left({args[0]}\\right)",
    "sequence_max": lambda args: f"\\max\\left({args[0]}\\right)",
    "sequence_cumsum": lambda args: f"\\operatorname{{cumsum}}\\left({args[0]}\\right)",
    "sequence_cumprod": lambda args: f"\\operatorname{{cumprod}}\\left({args[0]}\\right)",
    "sequence_drawdown": lambda args: f"\\operatorname{{drawdown}}\\left({args[0]}\\right)",
    "sequence_excess_return": _binary("-"),
    "sequence_geometric_mean": lambda args: f"\\operatorname{{gmean}}\\left({args[0]}\\right)",
    "sequence_compound_annual_growth_rate": lambda args: f"\\operatorname{{CAGR}}\\left({args[0]}, {args[1]}\\right)",
    "sequence_add": _binary("+"),
    "sequence_subtract": _binary("-"),
    "sequence_multiply": _binary(r"\times"),
    "sequence_divide": lambda args: f"\\frac{{{args[0]}}}{{{args[1]}}}",
    "cumulative_simple_returns": lambda args: f"\\operatorname{{cumret}}\\left({args[0]}\\right)",
    "cumulative_net_value": lambda args: f"\\operatorname{{net}}\\left({args[0]}, {args[1]}\\right)",
    "max_drawdown_rate_and_recovery_days": lambda args: f"\\operatorname{{MDD}}\\left({args[0]}\\right)",
}


def generate_latex_expression(name: str, arg_names: List[str]) -> str:
    placeholders = [format_argument(arg) for arg in arg_names]
    pattern = LATEX_PATTERNS.get(name)
    try:
        if pattern:
            return pattern(placeholders)
    except Exception:
        pass
    operator_label = latex_operator_name(name)
    inside = ", ".join(placeholders) if placeholders else ""
    return r"\operatorname{{{0}}}\left({1}\right)".format(operator_label, inside)


def extract_signature_strings(node: ast.FunctionDef) -> List[str]:
    """
    从 njit 或 vectorize 装饰器中提取类型签名字符串列表。
    """
    signatures: List[str] = []
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        func = decorator.func
        if isinstance(func, ast.Name):
            if func.id == "njit":
                if decorator.args and isinstance(decorator.args[0], ast.Constant):
                    value = decorator.args[0].value
                    if isinstance(value, str):
                        signatures.append(value.strip())
            elif func.id == "vectorize":
                if decorator.args:
                    first_arg = decorator.args[0]
                    if isinstance(first_arg, (ast.List, ast.Tuple)):
                        for element in first_arg.elts:
                            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                                signatures.append(element.value.strip())
                    elif isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                        signatures.append(first_arg.value.strip())
    return signatures


def has_vectorize_decorator(node: ast.FunctionDef) -> bool:
    """
    判断函数是否使用了 numba.vectorize 装饰器。
    """
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
            if decorator.func.id == "vectorize":
                return True
    return False


def split_signature_arguments(arg_section: str) -> List[str]:
    """
    在保持括号平衡的情况下拆分参数类型。
    """
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    for char in arg_section:
        if char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
    if current:
        part = "".join(current).strip()
        if part:
            parts.append(part)
    return parts


def parse_signature(signature: str) -> Tuple[str, List[str]]:
    """
    将 numba 类型签名拆解为返回类型和参数类型列表。
    """
    signature = signature.strip()
    if not signature or "(" not in signature or not signature.endswith(")"):
        return signature, []
    arg_start = signature.rfind("(")
    return_type = signature[:arg_start].strip()
    args_section = signature[arg_start + 1: -1].strip()
    if not args_section:
        return return_type, []
    return return_type, split_signature_arguments(args_section)


def clean_docstring(docstring: Optional[str]) -> str:
    """
    统一清洗 docstring 文本。
    """
    if not docstring:
        return ""
    return textwrap.dedent(docstring).strip()


def build_inputs(node: ast.FunctionDef, arg_types: List[str]) -> Tuple[List[Dict[str, object]], List[str]]:
    """
    构建输入参数描述，并返回参数名称列表。
    """
    inputs: List[Dict[str, object]] = []
    arg_names: List[str] = []
    for index, param in enumerate(node.args.args):
        arg_name = param.arg
        arg_names.append(arg_name)
        arg_type = arg_types[index] if index < len(arg_types) else ""
        inputs.append(
            {
                "name": arg_name,
                "type": arg_type,
                "required": True,
            }
        )
    return inputs, arg_names


def build_operator_entry(node: ast.FunctionDef) -> Optional[Dict[str, object]]:
    """
    将函数节点转换为 DSL 算子条目。
    """
    signatures = extract_signature_strings(node)
    if not signatures:
        return None

    primary_signature = signatures[0]
    return_type, arg_types = parse_signature(primary_signature)
    docstring = clean_docstring(ast.get_docstring(node))

    inputs, arg_names = build_inputs(node, arg_types)
    latex_expr = generate_latex_expression(node.name, arg_names)

    entry: Dict[str, object] = {
        "name": node.name,
        "aliases": [],
        "category": None,
        "description": docstring,
        "signature": primary_signature,
        "signatures": signatures,
        "is_ufunc": has_vectorize_decorator(node),
        "inputs": inputs,
        "output": {
            "type": return_type,
            "description": "",
        },
        "latex": latex_expr,
        "examples": [],
        "notes": [],
    }
    return entry


def build_operators() -> Tuple[List[Dict[str, object]], Dict[str, str]]:
    """
    解析 numba_finance_math.py，构建算子 DSL 列表。
    """
    source = TARGET_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TARGET_FILE))
    operators: List[Dict[str, object]] = []
    latex_map: Dict[str, str] = {}

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        entry = build_operator_entry(node)
        if entry is None:
            continue
        operators.append(entry)
        latex_map[entry["name"]] = entry.get("latex", "")

    return operators, latex_map


def main() -> None:
    """
    构建算子 DSL 并写出到 JSON 文件。
    """
    operators, _ = build_operators()
    metadata = {
        "module": MODULE_NAME,
        "version": DSL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "operator_count": len(operators),
    }
    dsl_payload = {
        "metadata": metadata,
        "operators": operators,
    }
    OUTPUT_FILE.write_text(
        json.dumps(dsl_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"DSL 已写入: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
