# Agentic Factor Reproduction Report

- Discovery mode: **agentic**
- Promoted factors: **2**

## Gross / net summary

### LINEAR
- Gross annualized return: `39.10%`
- Gross annualized Sharpe: `1.49`
- Net annualized return: `25.83%`
- Net annualized Sharpe: `1.08`
- Average daily turnover: `132.60%`

### LGBM
- Gross annualized return: `52.57%`
- Gross annualized Sharpe: `1.93`
- Net annualized return: `36.35%`
- Net annualized Sharpe: `1.45`
- Average daily turnover: `148.85%`

## Promoted factor library

- **f_0002 — Liquidity anomaly**
  - Expression: `cs_zscore(relative_turnover)`
  - Family: `turnover`
  - Rationale: Cross-sectional turnover tails flag names with unusual liquidity demand or neglect.
- **f_0004 — Friction-adjusted flow shock**
  - Expression: `cs_zscore(volume_growth_1 / (spread_proxy + 1e-4))`
  - Family: `liquidity`
  - Rationale: A flow shock net of trading frictions prioritizes signals that can survive execution costs.

## Artifacts

- `linear_gross_net.png` / `lgbm_gross_net.png`: gross-vs-net cumulative return charts
- `linear_deciles.png` / `lgbm_deciles.png`: decile fan-out charts
- `factor_metrics_is.csv`: full in-sample factor screening table
- `promoted_library.jsonl`: auditable promoted factor library