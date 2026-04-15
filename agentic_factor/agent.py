from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .recipe import FactorRecipe


@dataclass
class MemoryState:
    family_scores: dict[str, float] = field(default_factory=dict)
    family_successes: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    family_failures: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    survivors: list[FactorRecipe] = field(default_factory=list)

    def update(self, metrics: list[dict]) -> None:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for rec in metrics:
            grouped[rec["family"]].append(rec)
        for family, rows in grouped.items():
            promoted = [r for r in rows if r["decision"] == "promote"]
            retired = [r for r in rows if r["decision"] == "retire"]
            holds = [r for r in rows if r["decision"] == "hold"]
            self.family_successes[family] += len(promoted)
            self.family_failures[family] += len(retired)
            score = 1.0
            if promoted:
                score += sum(max(0.0, r.get("sharpe", 0.0)) for r in promoted) / len(promoted)
            score += 0.15 * len(holds)
            score -= 0.35 * len(retired)
            self.family_scores[family] = max(0.2, score)

    def weighted_families(self, families: Iterable[str]) -> tuple[list[str], list[float]]:
        fams = list(families)
        weights = [float(self.family_scores.get(f, 1.0)) for f in fams]
        return fams, weights


class HeuristicAgent:
    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)
        self.memory = MemoryState()
        self._counter = 0
        self.templates = {
            "flow": [
                ("Flow shock", "cs_rank(volume_growth_1)", "One-day volume growth proxies attention or order-flow surprise that may not be fully incorporated immediately."),
                ("Demand pressure", "cs_zscore(relative_turnover + delta(relative_turnover, 1))", "Current relative turnover plus its change captures concentrated demand pressure and liquidity imbalance."),
                ("Flow momentum", "cs_zscore(rolling_mean(relative_turnover + delta(relative_turnover, 1), {w1}))", "Smoothed turnover pressure separates sustained flow from one-day noise."),
                ("Order-flow persistence", "cs_rank(rolling_mean(relative_turnover + delta(relative_turnover, 1), {w1}))", "Persistent order flow can create short-run continuation or delayed incorporation."),
            ],
            "turnover": [
                ("Relative turnover", "cs_rank(relative_turnover)", "Current turnover relative to own history measures the intensity of trading interest."),
                ("Liquidity anomaly", "cs_zscore(relative_turnover)", "Cross-sectional turnover tails flag names with unusual liquidity demand or neglect."),
                ("Turnover divergence", "cs_zscore(relative_turnover - rolling_mean(relative_turnover, {w1}))", "A divergence from recent turnover baseline may indicate transient crowding or fresh information arrival."),
                ("Turnover extremity", "cs_zscore(abs(relative_turnover - 1.0))", "Extreme deviations from normal turnover emphasize supply-demand stress."),
            ],
            "crowding": [
                ("Short-horizon crowding", "cs_rank(rolling_mean(relative_turnover, 7))", "One-week average relative turnover captures short-horizon crowding and repeated attention."),
                ("Medium-horizon crowding", "cs_rank(rolling_mean(relative_turnover, 9))", "Persistent turnover over about two weeks acts as a crowding proxy."),
                ("Crowding acceleration", "cs_zscore(rolling_mean(relative_turnover, {w1}) + delta(relative_turnover, 1))", "Sustained elevated activity plus fresh acceleration can identify unstable crowding."),
            ],
            "price_vol": [
                ("Price-gap and volatility", "cs_zscore(-(price_to_ma_20 - 1.0) - realized_vol_20)", "A stock below its recent trend with subdued realized volatility can proxy for mean reversion or risk-adjusted mispricing."),
                ("Risk-adjusted price level", "cs_rank(-(price_to_ma_20 - 1.0) / (realized_vol_20 + 1e-6))", "Price dislocation scaled by volatility favors stable deviations over noisy ones."),
                ("Price/flow tension", "cs_zscore(-(price_to_ma_20 - 1.0) + rolling_mean(relative_turnover, {w1}))", "Price below trend combined with persistent flow may indicate unresolved information."),
            ],
            "liquidity": [
                ("Friction-adjusted flow shock", "cs_zscore(volume_growth_1 / (spread_proxy + 1e-4))", "A flow shock net of trading frictions prioritizes signals that can survive execution costs."),
                ("Spread-aware turnover", "cs_rank(relative_turnover / (spread_proxy + 1e-4))", "Turnover scaled by spread rewards attention supported by cheaper execution conditions."),
                ("Liquidity squeeze", "cs_zscore(relative_turnover - spread_proxy)", "High activity against a tight spread can signal absorptive liquidity; against a wide spread it may indicate stress."),
            ],
        }
        self.window_choices = [3, 5, 7, 9, 12, 15, 20]

    def generate(self, round_idx: int, n_candidates: int) -> list[FactorRecipe]:
        families, weights = self.memory.weighted_families(self.templates.keys())
        candidates: list[FactorRecipe] = []
        while len(candidates) < n_candidates:
            family = self.random.choices(families, weights=weights, k=1)[0]
            name, expr_template, rationale = self.random.choice(self.templates[family])
            w1 = self.random.choice(self.window_choices)
            expr = expr_template.format(w1=w1)
            name2 = f"{name} [{family}:{w1}]" if "{w1}" in expr_template else name
            candidates.append(self._make_recipe(name2, expr, rationale, family, round_idx))
        if self.memory.survivors:
            survivors = self.random.sample(self.memory.survivors, k=min(len(self.memory.survivors), max(1, n_candidates // 4)))
            for parent in survivors:
                expr = self._mutate_expression(parent.expression)
                candidates.append(self._make_recipe(f"Mutated {parent.name}", expr, parent.rationale + " This variant is a local mutation around a previously successful family.", parent.family, round_idx, parent.factor_id))
        seen = set(); deduped = []
        for rec in candidates:
            if rec.expression not in seen:
                deduped.append(rec); seen.add(rec.expression)
        return deduped[:n_candidates]

    def absorb_round(self, round_metrics: list[dict], promoted_recipes: list[FactorRecipe]) -> None:
        self.memory.update(round_metrics)
        self.memory.survivors = promoted_recipes[:]

    def _make_recipe(self, name: str, expression: str, rationale: str, family: str, round_idx: int, parent_id: str | None = None) -> FactorRecipe:
        self._counter += 1
        return FactorRecipe(factor_id=f"f_{self._counter:04d}", name=name, expression=expression, rationale=rationale, family=family, round_created=round_idx, parent_id=parent_id, tags=[family], metadata={})

    def _mutate_expression(self, expression: str) -> str:
        windows = re.findall(r"\b(3|5|7|9|12|15|20)\b", expression)
        new_expr = expression
        for w in windows:
            if self.random.random() < 0.7:
                new_expr = new_expr.replace(w, str(self.random.choice(self.window_choices)), 1)
        replacements = {"cs_rank": "cs_zscore", "cs_zscore": "cs_rank", "rolling_mean": "rolling_std"}
        if self.random.random() < 0.35:
            src = self.random.choice(list(replacements.keys()))
            if src in new_expr:
                new_expr = new_expr.replace(src, replacements[src], 1)
        if self.random.random() < 0.25 and "spread_proxy" not in new_expr:
            new_expr = f"{new_expr} / (spread_proxy + 1e-4)"
        return new_expr


def paper_seed_library() -> list[FactorRecipe]:
    specs = [
        ("Demand pressure", "cs_zscore(relative_turnover + delta(relative_turnover, 1))", "turnover"),
        ("Flow momentum", "cs_zscore(rolling_mean(relative_turnover + delta(relative_turnover, 1), 3))", "flow"),
        ("Flow shock", "cs_rank(volume_growth_1)", "flow"),
        ("Liquidity anomaly", "cs_zscore(relative_turnover)", "turnover"),
        ("Medium-horizon crowding", "cs_rank(rolling_mean(relative_turnover, 9))", "crowding"),
        ("Order-flow persistence", "cs_rank(rolling_mean(relative_turnover + delta(relative_turnover, 1), 3))", "flow"),
        ("Price-gap and volatility", "cs_zscore(-(price_to_ma_20 - 1.0) - realized_vol_20)", "price_vol"),
        ("Relative turnover", "cs_rank(relative_turnover)", "turnover"),
        ("Risk-adjusted price level", "cs_rank(-(price_to_ma_20 - 1.0) / (realized_vol_20 + 1e-6))", "price_vol"),
        ("Short-horizon crowding", "cs_rank(rolling_mean(relative_turnover, 7))", "crowding"),
        ("Turnover divergence", "cs_zscore(relative_turnover - rolling_mean(relative_turnover, 20))", "turnover"),
        ("Turnover extremity", "cs_zscore(abs(relative_turnover - 1.0))", "turnover"),
    ]
    out = []
    for idx, (name, expr, family) in enumerate(specs, start=1):
        out.append(FactorRecipe(factor_id=f"paper_{idx:02d}", name=name, expression=expr, rationale="Approximate reproduction inferred from Table IX. The paper does not publish the exact symbolic formula, so this package implements a transparent approximation.", family=family, round_created=0, tags=["paper_seed", family]))
    return out


def traditional_baseline_library() -> list[FactorRecipe]:
    specs = [
        ("One-shot lag return", "cs_rank(lag_ret_1)", "baseline"),
        ("One-shot market beta proxy", "cs_rank(market_ret)", "baseline"),
        ("One-shot price gap", "cs_zscore(price_to_ma_20 - 1.0)", "baseline"),
        ("One-shot volume growth", "cs_rank(volume_growth_1)", "baseline"),
        ("One-shot realized volatility", "cs_zscore(realized_vol_20)", "baseline"),
        ("One-shot turnover", "cs_rank(relative_turnover)", "baseline"),
    ]
    return [FactorRecipe(factor_id=f"baseline_{i:02d}", name=name, expression=expr, rationale="Static one-shot baseline without memory update or economic reflection.", family=family, round_created=0, tags=["traditional_baseline"]) for i, (name, expr, family) in enumerate(specs, start=1)]
