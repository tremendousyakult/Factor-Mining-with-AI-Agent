from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from .utils import cs_rank, cs_zscore, delta_asset, groupby_asset, lag_asset, rolling_mean_asset, rolling_std_asset, winsorize_cs


class UnsafeExpressionError(ValueError):
    pass


@dataclass(slots=True)
class ExpressionEngine:
    panel: pd.DataFrame

    def evaluate(self, expression: str) -> pd.Series:
        tree = ast.parse(expression, mode="eval")
        result = self._visit(tree.body)
        if isinstance(result, (int, float)):
            return pd.Series(float(result), index=self.panel.index)
        if not isinstance(result, pd.Series):
            raise UnsafeExpressionError(f"Expression did not evaluate to a pandas Series: {expression}")
        return result.reindex(self.panel.index)

    def _visit(self, node: ast.AST) -> Any:
        if isinstance(node, ast.BinOp):
            left = self._visit(node.left)
            right = self._visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            raise UnsafeExpressionError(f"Unsupported binary operator: {ast.dump(node.op)}")
        if isinstance(node, ast.UnaryOp):
            value = self._visit(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
            raise UnsafeExpressionError(f"Unsupported unary operator: {ast.dump(node.op)}")
        if isinstance(node, ast.Name):
            if node.id not in self.panel.columns:
                raise UnsafeExpressionError(f"Unknown variable: {node.id}")
            return self.panel[node.id]
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise UnsafeExpressionError(f"Unsupported constant: {node.value!r}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise UnsafeExpressionError("Only simple function calls are allowed")
            func_name = node.func.id
            args = [self._visit(arg) for arg in node.args]
            kwargs = {kw.arg: self._visit(kw.value) for kw in node.keywords}
            return self._functions()[func_name](*args, **kwargs)
        raise UnsafeExpressionError(f"Unsupported AST node: {ast.dump(node)}")

    def _functions(self) -> dict[str, Callable[..., Any]]:
        def _window(n: float) -> int:
            if not math.isfinite(n):
                raise UnsafeExpressionError("Window must be finite")
            return max(1, int(round(float(n))))
        def _lag(x: pd.Series, n: float = 1.0) -> pd.Series:
            return lag_asset(x, _window(n))
        def _delta(x: pd.Series, n: float = 1.0) -> pd.Series:
            return delta_asset(x, _window(n))
        def _rolling_mean(x: pd.Series, n: float) -> pd.Series:
            return rolling_mean_asset(x, _window(n), min_periods=_window(n))
        def _rolling_std(x: pd.Series, n: float) -> pd.Series:
            return rolling_std_asset(x, _window(n), min_periods=_window(n))
        def _rolling_sum(x: pd.Series, n: float) -> pd.Series:
            w = _window(n)
            return groupby_asset(x).transform(lambda s: s.rolling(w, min_periods=w).sum())
        def _cs_rank(x: pd.Series) -> pd.Series:
            return cs_rank(x)
        def _cs_zscore(x: pd.Series) -> pd.Series:
            return cs_zscore(x)
        def _ts_zscore(x: pd.Series, n: float) -> pd.Series:
            w = _window(n)
            mean = rolling_mean_asset(x, w, min_periods=w)
            std = rolling_std_asset(x, w, min_periods=w)
            return (x - mean) / (std + 1e-12)
        def _winsor(x: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
            return winsorize_cs(x, float(lower), float(upper))
        def _abs(x: pd.Series) -> pd.Series:
            return x.abs()
        def _sign(x: pd.Series) -> pd.Series:
            return np.sign(x)
        def _sqrt(x: pd.Series) -> pd.Series:
            return np.sqrt(np.maximum(x, 0))
        def _clip(x: pd.Series, lower: float = -np.inf, upper: float = np.inf) -> pd.Series:
            return x.clip(lower=float(lower), upper=float(upper))
        def _log1p(x: pd.Series) -> pd.Series:
            return np.log1p(np.maximum(x, -0.999999))
        return {
            "lag": _lag, "delta": _delta, "rolling_mean": _rolling_mean, "rolling_std": _rolling_std,
            "rolling_sum": _rolling_sum, "cs_rank": _cs_rank, "cs_zscore": _cs_zscore,
            "ts_zscore": _ts_zscore, "winsor": _winsor, "abs": _abs, "sign": _sign,
            "sqrt": _sqrt, "clip": _clip, "log1p": _log1p,
        }
