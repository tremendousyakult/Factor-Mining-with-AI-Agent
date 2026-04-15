from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from .utils import annualized_return, annualized_sharpe, bucketize, max_drawdown


@dataclass(slots=True)
class CompositeResult:
    name: str
    score: pd.Series
    decile_returns: pd.DataFrame
    ls_returns: pd.Series
    turnover: pd.Series
    net_ls_returns: pd.Series
    summary: dict[str, Any]


@dataclass(slots=True)
class ModelArtifacts:
    feature_names: list[str]
    model: Any | None = None


def make_feature_frame(evaluations, panel: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=panel.index)
    for ev in evaluations:
        frame[ev.recipe.factor_id] = ev.score
    frame["ret_fwd_1"] = panel["ret_fwd_1"]
    if "market_ret" in panel.columns:
        frame["market_ret"] = panel["market_ret"]
    return frame


def linear_composite(feature_frame: pd.DataFrame, feature_names: list[str]):
    return feature_frame[feature_names].mean(axis=1, skipna=True), ModelArtifacts(feature_names=feature_names, model=None)


def lgbm_composite(feature_frame_is: pd.DataFrame, feature_frame_oos: pd.DataFrame, feature_names: list[str], config: dict[str, Any]):
    cfg = config["aggregation"]["lgbm"]
    train = feature_frame_is.dropna(subset=feature_names + ["ret_fwd_1"]).copy()
    model = LGBMRegressor(
        objective="regression", n_estimators=int(cfg.get("n_estimators", 300)), learning_rate=float(cfg.get("learning_rate", 0.05)),
        num_leaves=int(cfg.get("num_leaves", 31)), max_depth=int(cfg.get("max_depth", -1)), min_child_samples=int(cfg.get("min_child_samples", 50)),
        subsample=float(cfg.get("subsample", 0.8)), colsample_bytree=float(cfg.get("colsample_bytree", 0.8)), reg_alpha=float(cfg.get("reg_alpha", 0.0)),
        reg_lambda=float(cfg.get("reg_lambda", 1.0)), random_state=int(config["agent"].get("seed", 42)), verbosity=-1, n_jobs=1, force_col_wise=True
    )
    model.fit(train[feature_names], train["ret_fwd_1"])
    preds = pd.Series(model.predict(feature_frame_oos[feature_names].fillna(0.0)), index=feature_frame_oos.index)
    return preds, ModelArtifacts(feature_names=feature_names, model=model)


def compute_decile_returns(score: pd.Series, forward_returns: pd.Series, n_quantiles: int = 10) -> pd.DataFrame:
    joined = pd.DataFrame({"score": score, "ret_fwd_1": forward_returns}).dropna().assign(bucket=bucketize(score.dropna(), q=n_quantiles))
    deciles = joined.groupby([joined.index.get_level_values("date"), "bucket"], sort=False)["ret_fwd_1"].mean().unstack("bucket")
    for q in range(1, n_quantiles + 1):
        if q not in deciles.columns:
            deciles[q] = np.nan
    deciles = deciles.reindex(sorted(deciles.columns), axis=1)
    deciles.columns = [f"D{int(c)}" for c in deciles.columns]
    return deciles


def build_ls_weights(score: pd.Series, n_quantiles: int = 10) -> dict[pd.Timestamp, pd.Series]:
    joined = pd.DataFrame({"score": score}).dropna().assign(bucket=bucketize(score.dropna(), q=n_quantiles))
    weights = {}
    for dt, g in joined.groupby(level="date", sort=False):
        assets = g.index.get_level_values("asset")
        w = pd.Series(0.0, index=assets)
        long_assets = assets[g["bucket"].to_numpy() == n_quantiles]
        short_assets = assets[g["bucket"].to_numpy() == 1]
        if len(long_assets) > 0:
            w.loc[long_assets] = 1.0 / len(long_assets)
        if len(short_assets) > 0:
            w.loc[short_assets] = -1.0 / len(short_assets)
        weights[pd.Timestamp(dt)] = w.groupby(level=0).sum()
    return weights


def compute_turnover(weight_map: dict[pd.Timestamp, pd.Series]) -> pd.Series:
    prev = pd.Series(dtype=float); rows = {}
    for dt in sorted(weight_map.keys()):
        cur = weight_map[dt]
        union = prev.index.union(cur.index)
        rows[dt] = 0.5 * (cur.reindex(union, fill_value=0.0) - prev.reindex(union, fill_value=0.0)).abs().sum()
        prev = cur
    return pd.Series(rows).sort_index()


def apply_linear_transaction_costs(gross_returns: pd.Series, turnover: pd.Series, one_way_bps: float) -> pd.Series:
    return gross_returns - turnover.reindex(gross_returns.index).fillna(0.0) * (one_way_bps / 10000.0)


def summarize_spread(returns: pd.Series) -> dict[str, Any]:
    clean = returns.dropna()
    return {
        "period_return": float(np.prod(1.0 + clean.to_numpy()) - 1.0) if not clean.empty else float("nan"),
        "ann_return": float(annualized_return(clean)),
        "ann_sharpe": float(annualized_sharpe(clean)),
        "max_dd": float(max_drawdown(clean)),
        "n_days": int(clean.shape[0]),
    }


def evaluate_composite(name: str, score: pd.Series, panel_oos: pd.DataFrame, one_way_bps: float = 3.0) -> CompositeResult:
    deciles = compute_decile_returns(score, panel_oos["ret_fwd_1"], n_quantiles=10)
    ls = deciles["D10"] - deciles["D1"]
    turnover = compute_turnover(build_ls_weights(score, n_quantiles=10)).reindex(ls.index)
    net_ls = apply_linear_transaction_costs(ls, turnover, one_way_bps)
    return CompositeResult(name, score, deciles, ls, turnover, net_ls, {"gross": summarize_spread(ls), "net": summarize_spread(net_ls), "avg_turnover": float(turnover.mean())})


def quarterly_cost_table(result: CompositeResult) -> pd.DataFrame:
    df = pd.DataFrame({"gross_ret": result.ls_returns, "net_ret": result.net_ls_returns, "turnover": result.turnover}).dropna()
    rows = []
    for q, g in df.groupby(df.index.to_period("Q"), sort=True):
        rows.append({"quarter": str(q), "avg_turnover": float(g["turnover"].mean()), "gross_ret": float(np.prod(1.0 + g["gross_ret"].to_numpy()) - 1.0), "net_ret": float(np.prod(1.0 + g["net_ret"].to_numpy()) - 1.0), "gross_sharpe": float(annualized_sharpe(g["gross_ret"])), "net_sharpe": float(annualized_sharpe(g["net_ret"]))})
    return pd.DataFrame(rows)


def risk_adjusted_alpha(portfolio_returns: pd.Series, benchmark_df: pd.DataFrame, groups: dict[str, list[str]], risk_free_col: str | None = None) -> pd.DataFrame:
    rows = []
    y = portfolio_returns.dropna().copy()
    bench = benchmark_df.copy(); bench.index = pd.to_datetime(bench.index)
    if risk_free_col and risk_free_col in bench.columns:
        y = y - bench.loc[y.index, risk_free_col]
    for name, cols in groups.items():
        use_cols = [c for c in cols if c in bench.columns]
        if not use_cols:
            continue
        reg_df = pd.concat([y.rename("y"), bench.loc[y.index, use_cols]], axis=1).dropna()
        if reg_df.shape[0] < 20:
            continue
        model = OLS(reg_df["y"], add_constant(reg_df[use_cols])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        rows.append({"model": name, "alpha_daily": float(model.params["const"]), "alpha_annualized": float(model.params["const"] * 252), "t_stat": float(model.tvalues["const"])})
    return pd.DataFrame(rows)
