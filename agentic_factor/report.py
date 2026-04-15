from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _plot_cumulative(returns: pd.Series, title: str, path: Path) -> None:
    wealth = (1.0 + returns.fillna(0.0)).cumprod() - 1.0
    fig, ax = plt.subplots(figsize=(10, 4.5))
    wealth.plot(ax=ax)
    ax.set_title(title); ax.set_ylabel("Cumulative return"); ax.set_xlabel(""); ax.axhline(0.0, linewidth=0.8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_gross_net(spread_df: pd.DataFrame, title: str, path: Path) -> None:
    gross = (1.0 + spread_df["gross_ls"].fillna(0.0)).cumprod() - 1.0
    net = (1.0 + spread_df["net_ls"].fillna(0.0)).cumprod() - 1.0
    fig, ax = plt.subplots(figsize=(10, 4.5))
    gross.plot(ax=ax, label="gross"); net.plot(ax=ax, label="net")
    ax.set_title(title); ax.set_ylabel("Cumulative return"); ax.set_xlabel(""); ax.axhline(0.0, linewidth=0.8); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_deciles(deciles: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ((1.0 + deciles.fillna(0.0)).cumprod() - 1.0).plot(ax=ax)
    ax.set_title(title); ax.set_ylabel("Cumulative return"); ax.set_xlabel(""); ax.legend(ncol=5, fontsize=8); ax.axhline(0.0, linewidth=0.8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def render_run_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    promoted = pd.read_json(run_dir / "promoted_library.jsonl", lines=True)
    linear_spread = pd.read_csv(run_dir / "linear" / "spread_returns.csv", index_col=0, parse_dates=True)
    linear_deciles = pd.read_csv(run_dir / "linear" / "decile_returns.csv", index_col=0, parse_dates=True)
    lgbm_spread = pd.read_csv(run_dir / "lgbm" / "spread_returns.csv", index_col=0, parse_dates=True)
    lgbm_deciles = pd.read_csv(run_dir / "lgbm" / "decile_returns.csv", index_col=0, parse_dates=True)
    _plot_gross_net(linear_spread, "Linear composite: gross vs net", run_dir / "linear_gross_net.png")
    _plot_deciles(linear_deciles, "Linear composite decile fan-out", run_dir / "linear_deciles.png")
    _plot_gross_net(lgbm_spread, "LightGBM composite: gross vs net", run_dir / "lgbm_gross_net.png")
    _plot_deciles(lgbm_deciles, "LightGBM composite decile fan-out", run_dir / "lgbm_deciles.png")
    _plot_cumulative(linear_spread["gross_ls"], "Linear composite gross cumulative return", run_dir / "linear_gross.png")
    _plot_cumulative(lgbm_spread["gross_ls"], "LGBM composite gross cumulative return", run_dir / "lgbm_gross.png")
    report = run_dir / "report.md"
    lines = ["# Agentic Factor Reproduction Report\n", f"- Discovery mode: **{summary['discovery_mode']}**", f"- Promoted factors: **{summary['n_promoted']}**", "", "## Gross / net summary", ""]
    for model_name in ["linear_summary", "lgbm_summary"]:
        s = summary[model_name]
        lines += [f"### {model_name.replace('_summary', '').upper()}", f"- Gross annualized return: `{s['gross']['ann_return']:.2%}`", f"- Gross annualized Sharpe: `{s['gross']['ann_sharpe']:.2f}`", f"- Net annualized return: `{s['net']['ann_return']:.2%}`", f"- Net annualized Sharpe: `{s['net']['ann_sharpe']:.2f}`", f"- Average daily turnover: `{s['avg_turnover']:.2%}`", ""]
    lines += ["## Promoted factor library", ""]
    if not promoted.empty:
        for _, row in promoted.iterrows():
            lines += [f"- **{row['factor_id']} — {row['name']}**", f"  - Expression: `{row['expression']}`", f"  - Family: `{row['family']}`", f"  - Rationale: {row['rationale']}"]
    lines += ["", "## Artifacts", "", "- `linear_gross_net.png` / `lgbm_gross_net.png`: gross-vs-net cumulative return charts", "- `linear_deciles.png` / `lgbm_deciles.png`: decile fan-out charts", "- `factor_metrics_is.csv`: full in-sample factor screening table", "- `promoted_library.jsonl`: auditable promoted factor library"]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
