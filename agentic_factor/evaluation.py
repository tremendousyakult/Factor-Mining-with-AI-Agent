from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .engine import ExpressionEngine
from .recipe import FactorRecipe
from .utils import annualized_return, annualized_sharpe, annualized_sortino, bucketize, calmar_ratio, cs_zscore, max_drawdown, winsorize_cs


@dataclass(slots=True)
class FactorEvaluation:
    recipe: FactorRecipe
    score: pd.Series
    ic_by_date: pd.Series
    ls_returns: pd.Series
    metrics: dict[str, Any]
    decision: str


class FactorEvaluator:
    def __init__(self, panel: pd.DataFrame, config: dict[str, Any]):
        self.panel = panel
        self.config = config
        self.engine = ExpressionEngine(panel)

    def evaluate(self, recipe: FactorRecipe) -> FactorEvaluation:
        raw = self.engine.evaluate(recipe.expression)
        score = cs_zscore(winsorize_cs(raw, lower=float(self.config["preprocessing"].get("factor_winsor_lower", 0.01)), upper=float(self.config["preprocessing"].get("factor_winsor_upper", 0.99))))
        ic_by_date = self._daily_rank_ic(score)
        ls_returns = self._half_split_returns(score)
        metrics = {
            "ic_mean": float(ic_by_date.mean()),
            "ic_std": float(ic_by_date.std(ddof=1)),
            "t_ic": self._t_stat(ic_by_date),
            "icir": self._ir(ic_by_date),
            "sharpe": float(annualized_sharpe(ls_returns)),
            "sortino": float(annualized_sortino(ls_returns)),
            "ann_return": float(annualized_return(ls_returns)),
            "max_dd": float(max_drawdown(ls_returns)),
            "calmar": float(calmar_ratio(ls_returns)),
            "monotonicity": float(self._decile_monotonicity(score)),
            "coverage_days": int(ls_returns.dropna().shape[0]),
        }
        return FactorEvaluation(recipe, score, ic_by_date, ls_returns, metrics, self._gate(metrics))

    def _daily_rank_ic(self, score: pd.Series) -> pd.Series:
        joined = pd.DataFrame({"score": score, "ret_fwd_1": self.panel["ret_fwd_1"]}).dropna()
        def _corr(g: pd.DataFrame) -> float:
            if g.shape[0] < 8:
                return np.nan
            return g["score"].corr(g["ret_fwd_1"], method="spearman")
        return joined.groupby(level="date", sort=False).apply(_corr)

    def _half_split_returns(self, score: pd.Series) -> pd.Series:
        joined = pd.DataFrame({"score": score, "ret_fwd_1": self.panel["ret_fwd_1"]}).dropna()
        def _spread(g: pd.DataFrame) -> float:
            if g.shape[0] < 10:
                return np.nan
            rank_pct = g["score"].rank(pct=True, method="first")
            long_ret = g.loc[rank_pct > 0.5, "ret_fwd_1"].mean()
            short_ret = g.loc[rank_pct <= 0.5, "ret_fwd_1"].mean()
            return long_ret - short_ret
        return joined.groupby(level="date", sort=False).apply(_spread)

    def _decile_monotonicity(self, score: pd.Series) -> float:
        joined = pd.DataFrame({"score": score, "ret_fwd_1": self.panel["ret_fwd_1"]}).dropna()
        joined = joined.assign(bucket=bucketize(joined["score"], q=10))
        decile_means = joined.groupby("bucket")["ret_fwd_1"].mean()
        if decile_means.shape[0] < 5:
            return float("nan")
        x = pd.Series(decile_means.index.astype(int).to_numpy())
        y = pd.Series(decile_means.to_numpy())
        return float(x.corr(y, method="spearman"))

    @staticmethod
    def _t_stat(series: pd.Series) -> float:
        clean = series.dropna()
        if clean.shape[0] < 3:
            return float("nan")
        std = clean.std(ddof=1)
        if pd.isna(std) or std == 0:
            return float("nan")
        return float(clean.mean() / (std / np.sqrt(clean.shape[0])))

    @staticmethod
    def _ir(series: pd.Series) -> float:
        clean = series.dropna(); std = clean.std(ddof=1)
        if clean.empty or pd.isna(std) or std == 0:
            return float("nan")
        return float(clean.mean() / std)

    def _gate(self, metrics: dict[str, Any]) -> str:
        ev = self.config["evaluation"]
        tau_sig = float(ev.get("tau_sig", 3.0)); tau_econ = float(ev.get("tau_econ", 1.0)); tau_fail = float(ev.get("tau_fail", 1.0))
        require_monotonicity = bool(ev.get("require_positive_monotonicity", True))
        t_ic = abs(float(metrics.get("t_ic", np.nan))); sharpe = float(metrics.get("sharpe", np.nan)); mono = float(metrics.get("monotonicity", np.nan))
        mono_ok = (not require_monotonicity) or (np.isnan(mono) or mono > 0)
        if t_ic >= tau_sig and sharpe >= tau_econ and mono_ok:
            return "promote"
        if t_ic < tau_fail or np.isnan(sharpe):
            return "retire"
        return "hold"


def evaluate_library(recipes: list[FactorRecipe], panel: pd.DataFrame, config: dict[str, Any]) -> list[FactorEvaluation]:
    evaluator = FactorEvaluator(panel, config)
    return [evaluator.evaluate(recipe) for recipe in recipes]
