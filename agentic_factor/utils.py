from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DATE_LEVEL = "date"
ASSET_LEVEL = "asset"


def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col])
    return out


def ensure_multiindex(df: pd.DataFrame, date_col: str, asset_col: str) -> pd.DataFrame:
    out = df.copy()
    out = ensure_datetime(out, date_col)
    out = out.sort_values([date_col, asset_col]).set_index([date_col, asset_col])
    out.index = out.index.set_names([DATE_LEVEL, ASSET_LEVEL])
    return out


def groupby_date(s: pd.Series):
    return s.groupby(level=DATE_LEVEL, sort=False)


def groupby_asset(s: pd.Series):
    return s.groupby(level=ASSET_LEVEL, sort=False)


def cs_rank(s: pd.Series) -> pd.Series:
    return groupby_date(s).rank(pct=True, method="average")


def cs_zscore(s: pd.Series, eps: float = 1e-12) -> pd.Series:
    g = groupby_date(s)
    mean = g.transform("mean")
    std = g.transform("std").replace(0.0, np.nan)
    return (s - mean) / (std + eps)


def winsorize_cs(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    def _winsor(g: pd.Series) -> pd.Series:
        if g.notna().sum() == 0:
            return g
        lo = g.quantile(lower)
        hi = g.quantile(upper)
        return g.clip(lower=lo, upper=hi)
    return groupby_date(s).apply(_winsor).droplevel(0)


def rolling_mean_asset(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_p = window if min_periods is None else min_periods
    return groupby_asset(s).transform(lambda x: x.rolling(window, min_periods=min_p).mean())


def rolling_std_asset(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_p = window if min_periods is None else min_periods
    return groupby_asset(s).transform(lambda x: x.rolling(window, min_periods=min_p).std())


def lag_asset(s: pd.Series, periods: int = 1) -> pd.Series:
    return groupby_asset(s).shift(periods)


def delta_asset(s: pd.Series, periods: int = 1) -> pd.Series:
    return s - lag_asset(s, periods)


def rolling_std_date(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_p = window if min_periods is None else min_periods
    by_date = groupby_date(s).first().sort_index()
    rolled = by_date.rolling(window, min_periods=min_p).std()
    values = by_date.index.map(rolled.to_dict())
    # broadcast to full panel
    mapping = dict(zip(by_date.index, rolled.to_numpy()))
    out = s.index.get_level_values(DATE_LEVEL).map(mapping)
    return pd.Series(out, index=s.index, dtype=float)


def bucketize(scores: pd.Series, q: int = 10) -> pd.Series:
    pct = groupby_date(scores).rank(pct=True, method="first")
    bucket = np.ceil(pct * q).clip(1, q)
    out = pd.Series(bucket, index=scores.index, dtype="float64")
    out[scores.isna()] = np.nan
    return out.astype("Int64")


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    gross = float(np.prod(1.0 + clean.to_numpy()))
    years = len(clean) / periods_per_year
    if years <= 0 or gross <= 0:
        return float("nan")
    return gross ** (1.0 / years) - 1.0


def annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    clean = returns.dropna()
    if clean.shape[0] < 2:
        return float("nan")
    std = clean.std(ddof=1)
    if pd.isna(std) or std == 0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * clean.mean() / std)


def annualized_sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
    clean = returns.dropna()
    downside = clean[clean < 0].std(ddof=1)
    if clean.empty or pd.isna(downside) or downside == 0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * clean.mean() / downside)


def max_drawdown(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    wealth = (1.0 + clean).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    return float(dd.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    ann = annualized_return(returns, periods_per_year)
    mdd = abs(max_drawdown(returns))
    if pd.isna(ann) or pd.isna(mdd) or mdd == 0:
        return float("nan")
    return float(ann / mdd)


def make_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_jsonl(records: Iterable[dict], path: str | Path) -> None:
    import json
    with Path(path).open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
