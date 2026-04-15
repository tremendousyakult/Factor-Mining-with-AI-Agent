from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import ASSET_LEVEL, DATE_LEVEL, cs_zscore, ensure_multiindex, lag_asset, rolling_mean_asset, rolling_std_asset, rolling_std_date, winsorize_cs


def load_panel(path: str | Path, config: dict[str, Any]) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    cols = config["data"]
    return ensure_multiindex(df, cols["date_col"], cols["asset_col"])


def apply_screens(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = config["data"]
    out = df.copy()
    exchange_col = cfg.get("exchange_col")
    if exchange_col and exchange_col in out.columns and cfg.get("eligible_exchanges"):
        out = out[out[exchange_col].isin(cfg["eligible_exchanges"])]
    share_code_col = cfg.get("share_code_col")
    if share_code_col and share_code_col in out.columns and cfg.get("common_share_codes"):
        out = out[out[share_code_col].isin(cfg["common_share_codes"])]
    if cfg.get("min_price") is not None:
        out = out[out[cfg["price_col"]].abs() >= float(cfg["min_price"])]
    min_history = int(cfg.get("min_history_days", 0) or 0)
    if min_history > 0:
        counts = out.groupby(level=ASSET_LEVEL).size()
        keep = counts[counts >= min_history].index
        out = out[out.index.get_level_values(ASSET_LEVEL).isin(keep)]
    return out.sort_index()


def compute_base_panel(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = config["data"]
    prep = config["preprocessing"]
    ret_col = cfg["return_col"]
    price_col = cfg["price_col"]
    vol_col = cfg["volume_col"]
    market_col = cfg["market_return_col"]
    spread_col = cfg.get("spread_col")
    panel = df.copy()
    for col in [ret_col, price_col, vol_col, market_col]:
        panel[col] = panel[col].astype(float)
    panel[price_col] = panel[price_col].abs()
    if spread_col and spread_col in panel.columns:
        panel[spread_col] = panel[spread_col].astype(float).clip(lower=0)
    else:
        spread_col = "spread_proxy"
        panel[spread_col] = np.nan
    panel["ret_fwd_1"] = panel.groupby(level=ASSET_LEVEL)[ret_col].shift(-1)
    panel["ret_fwd_1"] = winsorize_cs(panel["ret_fwd_1"], lower=float(prep.get("target_winsor_lower", 0.01)), upper=float(prep.get("target_winsor_upper", 0.99)))
    panel["lag_ret_1"] = lag_asset(panel[ret_col], 1)
    panel["market_ret"] = panel[market_col]
    panel["abs_price"] = panel[price_col]
    panel["volume"] = panel[vol_col]
    vol_mean_20 = rolling_mean_asset(panel[vol_col], 20, min_periods=20)
    panel["volume_ratio_20"] = panel[vol_col] / (vol_mean_20 + 1e-12)
    panel["relative_turnover"] = panel["volume_ratio_20"]
    panel["realized_vol_20"] = rolling_std_asset(panel[ret_col], 20, min_periods=20)
    ma_price_20 = rolling_mean_asset(panel[price_col], 20, min_periods=20)
    panel["price_to_ma_20"] = panel[price_col] / (ma_price_20 + 1e-12)
    panel["market_vol_20"] = rolling_std_date(panel[market_col], 20, min_periods=20)
    panel["volume_growth_1"] = panel.groupby(level=ASSET_LEVEL)[vol_col].pct_change()
    panel["spread_proxy"] = panel[spread_col].fillna(panel[ret_col].abs())
    panel["price_gap_20"] = panel["price_to_ma_20"] - 1.0
    panel["turnover_change_1"] = panel["relative_turnover"] - lag_asset(panel["relative_turnover"], 1)
    panel["flow_pressure_base"] = panel["relative_turnover"] + panel["turnover_change_1"]
    panel["ret_cs_z"] = cs_zscore(panel[ret_col])
    cols = [
        "lag_ret_1","market_ret","abs_price","volume","volume_ratio_20","realized_vol_20",
        "price_to_ma_20","market_vol_20","volume_growth_1","spread_proxy","relative_turnover",
        "turnover_change_1","flow_pressure_base","price_gap_20","ret_fwd_1"
    ]
    return panel[cols].sort_index()


def split_is_oos(panel: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = config["split"]
    is_end = pd.Timestamp(split["is_end"])
    oos_start = pd.Timestamp(split["oos_start"])
    dates = panel.index.get_level_values(DATE_LEVEL)
    return panel[dates <= is_end].copy(), panel[dates >= oos_start].copy()
