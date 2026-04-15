from __future__ import annotations

import pandas as pd

from agentic_factor.evaluation import FactorEvaluation, FactorEvaluator
from agentic_factor.pipeline import AgenticFactorPipeline
from agentic_factor.recipe import FactorRecipe


def _dummy_recipe(fid: str) -> FactorRecipe:
    return FactorRecipe(
        factor_id=fid,
        name=fid,
        expression="lag_ret_1",
        rationale="test",
        family="test",
        round_created=1,
    )


def test_gate_respects_min_coverage_days() -> None:
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2020-01-01", "2020-01-02"]), ["A", "B"]],
        names=["date", "asset"],
    )
    panel = pd.DataFrame({"ret_fwd_1": [0.01, 0.02, -0.01, 0.00], "lag_ret_1": [0.0, 0.0, 0.0, 0.0]}, index=idx)
    cfg = {"preprocessing": {}, "evaluation": {"min_coverage_days": 5}}
    evaluator = FactorEvaluator(panel, cfg)
    assert evaluator._gate({"coverage_days": 2, "t_ic": 100.0, "sharpe": 10.0, "monotonicity": 1.0}) == "retire"


def test_promoted_factor_correlation_dedup() -> None:
    cfg = {"agent": {"max_corr": 0.90, "corr_method": "spearman"}}
    pipe = AgenticFactorPipeline(cfg)
    idx = pd.RangeIndex(8)
    ev1 = FactorEvaluation(_dummy_recipe("f_1"), pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=idx), pd.Series(dtype=float), pd.Series(dtype=float), {"t_ic": 4.0}, "promote")
    ev2 = FactorEvaluation(_dummy_recipe("f_2"), pd.Series([2, 4, 6, 8, 10, 12, 14, 16], index=idx), pd.Series(dtype=float), pd.Series(dtype=float), {"t_ic": 3.0}, "promote")
    ev3 = FactorEvaluation(_dummy_recipe("f_3"), pd.Series([8, 7, 6, 5, 4, 3, 2, 1], index=idx), pd.Series(dtype=float), pd.Series(dtype=float), {"t_ic": 2.0}, "promote")
    keep = pipe._dedupe_promoted_by_correlation([ev1, ev2, ev3], [ev1.recipe, ev2.recipe, ev3.recipe])
    assert [r.factor_id for r in keep] == ["f_1"]
