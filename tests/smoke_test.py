from __future__ import annotations

import json
from pathlib import Path

from agentic_factor.config import load_config
from agentic_factor.demo import generate_demo_panel
from agentic_factor.pipeline import AgenticFactorPipeline
from agentic_factor.report import render_run_report


def test_smoke(tmp_path: Path) -> None:
    data_path = tmp_path / "demo.parquet"
    run_dir = tmp_path / "run"
    generate_demo_panel(data_path, n_assets=40, start="2019-01-01", end="2024-12-31", seed=1)
    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "demo_small.yaml")
    cfg["agent"]["rounds"] = 2; cfg["agent"]["candidates_per_round"] = 8; cfg["data"]["min_history_days"] = 60
    summary = AgenticFactorPipeline(cfg).run(data_path, run_dir)
    assert summary["n_promoted"] >= 1
    report_path = render_run_report(run_dir)
    assert report_path.exists()
    assert json.loads((run_dir / "summary.json").read_text())["n_promoted"] >= 1
