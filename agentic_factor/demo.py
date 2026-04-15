from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_demo_panel(out_path: str | Path, n_assets: int = 120, start: str = "2018-01-01", end: str = "2024-12-31", seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    assets = [f"A{i:04d}" for i in range(1, n_assets + 1)]
    rows = []
    market_ret = pd.Series(rng.normal(0.0002, 0.008, len(dates)), index=dates)
    market_ret += 0.10 * market_ret.shift(1).fillna(0.0)
    for asset in assets:
        price = 20.0 + rng.uniform(0, 80)
        volume = rng.uniform(1e5, 3e6)
        spread = rng.uniform(0.0005, 0.004)
        flow_state = 0.0; crowd_state = 0.0
        ret_hist = []; vol_hist = []; price_hist = []
        beta = rng.normal(1.0, 0.25); size_noise = rng.normal(0, 0.002); micro_noise_scale = rng.uniform(0.006, 0.02)
        exchange = rng.choice(["NYSE", "NASDAQ", "AMEX"], p=[0.4, 0.45, 0.15]); share_code = 10
        for dt in dates:
            market = float(market_ret.loc[dt])
            volume_shock = rng.normal(0, 0.35) + 0.55 * flow_state
            flow_state = 0.45 * flow_state + rng.normal(0, 0.20)
            crowd_state = 0.88 * crowd_state + 0.20 * volume_shock + rng.normal(0, 0.10)
            recent_vol = np.mean(vol_hist[-20:]) if len(vol_hist) >= 20 else volume
            relative_turnover = volume / max(recent_vol, 1.0)
            recent_price = np.mean(price_hist[-20:]) if len(price_hist) >= 20 else price
            price_to_ma = price / max(recent_price, 1e-6)
            realized_vol = np.std(ret_hist[-20:]) if len(ret_hist) >= 20 else micro_noise_scale
            latent_alpha = 0.0025 * volume_shock + 0.0018 * (relative_turnover - 1.0) + 0.0012 * crowd_state - 0.0016 * (price_to_ma - 1.0) - 0.0010 * realized_vol - 0.0008 * spread * 100 + size_noise
            ret = beta * market + latent_alpha + rng.normal(0, micro_noise_scale)
            price = max(5.0, price * (1.0 + ret))
            volume = max(1_000.0, volume * np.exp(0.30 * volume_shock + rng.normal(0, 0.10)))
            spread = np.clip(spread * np.exp(rng.normal(0, 0.08)) + 0.0003 * abs(ret), 0.0002, 0.03)
            rows.append({"date": dt, "asset": asset, "ret": ret, "close": price, "volume": volume, "market_ret": market, "spread": spread, "exchange": exchange, "share_code": share_code})
            ret_hist.append(ret); vol_hist.append(volume); price_hist.append(price)
    df = pd.DataFrame(rows); out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv": df.to_csv(out_path, index=False)
    else: df.to_parquet(out_path, index=False)
    return out_path
